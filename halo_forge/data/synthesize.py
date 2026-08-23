"""Synthetic data generation (Track D1).

Distilabel-style pipeline that ties three already-shipped halo-forge
pieces together:

  1. **Teacher generation** — an OpenAI-compatible HTTP teacher (defaults
     to halo-forge's own `serve` endpoint at 127.0.0.1:8001). Any
     `(prompt) -> str` callable is pluggable, so users can target
     a hosted API, a local llama.cpp / Ollama / vLLM server, or a
     stub for tests.
  2. **Verification** — every candidate is scored via the V1 verifier
     plugin registry. Pass any registered short name (`execution`,
     `llm_judge`, `bleu`, `json_schema`, …).
  3. **Filter + write** — completions above the reward threshold land
     in the output JSONL in a training-ready shape (prompt + completion
     for SFT, or prompt + chosen + rejected for preference data when
     `output_kind="preference"`).

The whole pipeline is online: each prompt is generated, scored, and
filtered in one pass. No huge intermediate JSONL — just the survivors
and a small per-prompt audit row in the report.

Composition with the rest of the data toolkit:

  halo-forge data synthesize --seeds X.jsonl --output raw.jsonl
  halo-forge data dedup       --input raw.jsonl   --output deduped.jsonl --method fuzzy
  halo-forge data score       --input deduped.jsonl --output train.jsonl --top-k-pct 0.5
  halo-forge sft train        --data train.jsonl ...
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


TeacherCallable = Callable[[str], str]


# ---------- result shape ----------------------------------------------------


@dataclass
class SynthesisRow:
    """One generated (and possibly verified) row from the synthesis run."""

    prompt: str
    completion: str
    reward: float
    accepted: bool
    rejected_reason: Optional[str] = None
    rank_in_group: int = 0  # 0 = best in group, 1 = second-best, ...


@dataclass
class SynthesisResult:
    """Aggregate of a synthesis run."""

    n_seeds: int
    n_generated: int
    n_accepted: int
    n_rejected: int
    avg_reward: float
    threshold: float
    output_path: str
    duration_seconds: float
    teacher_model: Optional[str] = None
    verifier_name: str = ""
    verifier_profile_revision_id: Optional[str] = None
    rows: List[SynthesisRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rows"] = [asdict(r) for r in self.rows[:10]]  # trim for wire size
        return d


# ---------- default teacher (OpenAI-compatible HTTP) ------------------------


def _default_openai_teacher(
    *,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.8,
    timeout_s: float = 60.0,
) -> TeacherCallable:
    """Build a `TeacherCallable` that hits any OpenAI-compatible endpoint.

    Defaults to `http://127.0.0.1:8001/v1` so a `halo-forge serve` running
    locally is the implicit teacher with zero configuration.

    Env-var overrides:
        HALOFORGE_TEACHER_BASE_URL — base url
        HALOFORGE_TEACHER_API_KEY  — bearer token
    """
    resolved_base = base_url or os.environ.get("HALOFORGE_TEACHER_BASE_URL")

    def _call(prompt: str) -> str:
        from halo_forge.data_lab.integrations import configured_teacher

        return configured_teacher(
            prompt,
            {
                "endpoint_type": "openai_compatible",
                "teacher_model": model,
                "base_url": resolved_base or "http://127.0.0.1:8001/v1",
                "api_key": api_key,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout_seconds": timeout_s,
            },
        )

    return _call


# ---------- seed loader -----------------------------------------------------


def load_seeds(source: Any) -> List[str]:
    """Read prompts from a list, a JSONL file path, or a plain text file.

    JSONL files use the `prompt`, `text`, `question`, or `instruction`
    field (whichever is present). Text files are split on newlines.
    """
    if isinstance(source, list):
        return [str(s) for s in source if s]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Seed source not found: {path}")

    if path.suffix.lower() in {".jsonl", ".jl"}:
        out: List[str] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = (
                    row.get("prompt")
                    or row.get("text")
                    or row.get("question")
                    or row.get("instruction")
                )
                if prompt:
                    out.append(str(prompt))
        return out

    # Plain text — one prompt per line.
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ---------- main pipeline ---------------------------------------------------


def synthesize_dataset(
    *,
    seeds: Sequence[str] | str | Path,
    output_path: Path | str,
    teacher: Optional[TeacherCallable] = None,
    teacher_model: str = "default",
    verifier_name: str = "json_structure",
    verifier_kwargs: Optional[Dict[str, Any]] = None,
    verifier: Optional[Any] = None,
    verifier_profile_revision_id: Optional[str] = None,
    n_per_prompt: int = 1,
    reward_threshold: float = 0.5,
    output_kind: str = "sft",  # "sft" | "preference"
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.8,
    progress_log_every: int = 25,
) -> SynthesisResult:
    """Run a teacher → verifier → filter pipeline and write training data.

    Args:
        seeds: Iterable of prompt strings, or path to a JSONL/TXT file.
        output_path: Destination JSONL.
        teacher: Pluggable `(prompt) -> str` callable. Defaults to an
            OpenAI-compatible HTTP client targeting `--base-url`.
        teacher_model: Model name passed to the default OpenAI teacher.
            Ignored when `teacher` is provided.
        verifier_name: V1-registered verifier short name to score with.
        verifier_kwargs: Constructor kwargs for the verifier (e.g.
            `{"references": "..."}` for `bleu`).
        n_per_prompt: How many completions to sample per seed prompt.
            n>1 is the right mode for preference-data generation:
            best becomes `chosen`, worst becomes `rejected`.
        reward_threshold: Below this, completions are dropped.
        output_kind: "sft" writes `{prompt, completion}`. "preference"
            writes `{prompt, chosen, rejected}` and requires `n_per_prompt >= 2`.
        progress_log_every: How often to log progress (0 disables).
    """
    if output_kind not in {"sft", "preference"}:
        raise ValueError(f"output_kind must be 'sft' or 'preference', got {output_kind!r}")
    if output_kind == "preference" and n_per_prompt < 2:
        raise ValueError(
            "output_kind='preference' requires n_per_prompt >= 2 "
            "(best vs worst completion in each group)"
        )

    seed_prompts = list(seeds) if isinstance(seeds, (list, tuple)) else load_seeds(seeds)
    if not seed_prompts:
        raise ValueError("No seed prompts provided")

    # Build the legacy verifier lazily unless the caller supplied an exact
    # profile-backed runtime bridge.
    if verifier is None:
        from halo_forge.rlvr.verifiers import get_verifier

        verifier_cls = get_verifier(verifier_name)
        verifier = verifier_cls(**(verifier_kwargs or {}))

    # Build teacher.
    if teacher is None:
        teacher = _default_openai_teacher(
            model=teacher_model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[SynthesisRow] = []
    n_generated = 0
    n_accepted = 0
    sum_reward = 0.0
    t0 = time.time()

    with out_path.open("w") as out_file:
        for seed_idx, prompt in enumerate(seed_prompts):
            group: List[SynthesisRow] = []
            for k in range(n_per_prompt):
                try:
                    completion = teacher(prompt)
                except Exception as exc:
                    logger.warning("Teacher failed on prompt %d/%d: %s", seed_idx + 1, k + 1, exc)
                    group.append(
                        SynthesisRow(
                            prompt=prompt,
                            completion="",
                            reward=0.0,
                            accepted=False,
                            rejected_reason="teacher_error",
                        )
                    )
                    n_generated += 1
                    continue

                try:
                    result = (
                        verifier.verify(candidate=completion, prompt=prompt)
                        if verifier_profile_revision_id
                        else verifier.verify(completion)
                    )
                    reward = float(result.reward)
                except Exception as exc:
                    logger.warning(
                        "Verifier %s failed on completion: %s",
                        verifier_name,
                        exc,
                    )
                    reward = 0.0

                accepted = reward >= reward_threshold
                row = SynthesisRow(
                    prompt=prompt,
                    completion=completion,
                    reward=reward,
                    accepted=accepted,
                    rejected_reason=None if accepted else "below_threshold",
                )
                group.append(row)
                n_generated += 1
                sum_reward += reward

            # Write surviving rows from this group, shaped by output_kind.
            ranked = sorted(
                enumerate(group),
                key=lambda kv: kv[1].reward,
                reverse=True,
            )
            for rank, (_, row) in enumerate(ranked):
                row.rank_in_group = rank

            if output_kind == "sft":
                accepted_in_group = [r for r in group if r.accepted]
                for row in accepted_in_group:
                    out_file.write(
                        json.dumps(
                            {
                                "prompt": row.prompt,
                                "completion": row.completion,
                            }
                        )
                        + "\n"
                    )
                    n_accepted += 1
            else:  # preference
                # Need at least one above-threshold AND at least one
                # below the best — best→chosen, worst→rejected.
                if len(group) < 2:
                    continue
                ranked_rows = [r for _, r in ranked]
                best = ranked_rows[0]
                worst = ranked_rows[-1]
                if best.reward < reward_threshold:
                    # Even the best wasn't good enough — drop the pair.
                    continue
                if best.reward <= worst.reward:
                    # Tied group — no preference signal.
                    continue
                out_file.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "chosen": best.completion,
                            "rejected": worst.completion,
                            "chosen_reward": best.reward,
                            "rejected_reward": worst.reward,
                        }
                    )
                    + "\n"
                )
                n_accepted += 1

            rows.extend(group)
            if progress_log_every and (seed_idx + 1) % progress_log_every == 0:
                logger.info(
                    "[D1] %d/%d seeds, %d generated, %d kept (%.1f%%)",
                    seed_idx + 1,
                    len(seed_prompts),
                    n_generated,
                    n_accepted,
                    100.0 * n_accepted / max(1, n_generated),
                )

    return SynthesisResult(
        n_seeds=len(seed_prompts),
        n_generated=n_generated,
        n_accepted=n_accepted,
        n_rejected=n_generated - n_accepted,
        avg_reward=sum_reward / max(1, n_generated),
        threshold=reward_threshold,
        output_path=str(out_path),
        duration_seconds=time.time() - t0,
        teacher_model=teacher_model,
        verifier_name=verifier_name,
        verifier_profile_revision_id=verifier_profile_revision_id,
        rows=rows,
    )


__all__ = [
    "SynthesisRow",
    "SynthesisResult",
    "synthesize_dataset",
    "load_seeds",
]
