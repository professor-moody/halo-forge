"""Versioned reliability adapters over the existing verifier registry."""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Type

from halo_forge.rlvr.verifiers.base import Verifier
from halo_forge.rlvr.verifiers.registry import get_verifier, inventory

from .fingerprints import ImplementationFingerprint, fingerprint_verifier_class, runtime_identity
from .observation import (
    RewardContract,
    VerifierObservation,
    normalize_reward_contract,
    normalize_verifier_result,
)


VERIFIER_FAMILIES = {"deterministic", "llm_judge", "reward_model", "chain"}


@dataclass(frozen=True)
class VerifierCapability:
    key: str
    family: str
    adapter_id: str
    adapter_version: str
    implementation: str
    origin: str
    fingerprint: Optional[str]
    qualifiable: bool
    modalities: tuple[str, ...] = ("text",)
    tasks: tuple[str, ...] = (
        "binary",
        "categorical",
        "multi_label",
        "scalar",
        "pairwise",
        "ranking",
    )
    runtime_requirements: Mapping[str, Any] = field(default_factory=dict)
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        payload["tasks"] = list(self.tasks)
        return payload


class ReliabilityInvoker(Protocol):
    def __call__(self, item: Mapping[str, Any]) -> Any: ...


class VerifierReliabilityAdapter:
    """Adapter interface kept separate from the historical ``Verifier`` ABC."""

    adapter_id = "legacy_verifier"
    adapter_version = "1"

    def capability(self) -> VerifierCapability:
        raise NotImplementedError

    def invoke(
        self,
        item: Mapping[str, Any],
        *,
        contract: RewardContract | Mapping[str, Any] | Any,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> VerifierObservation:
        raise NotImplementedError


class RegistryVerifierReliabilityAdapter(VerifierReliabilityAdapter):
    """Wrap one registered ``Verifier`` class without modifying that class."""

    adapter_id = "registered_verifier"
    adapter_version = "1"

    def __init__(
        self,
        name: str,
        *,
        configuration: Optional[Mapping[str, Any]] = None,
        family: Optional[str] = None,
        modalities: Iterable[str] = ("text",),
        tasks: Iterable[str] = ("binary",),
    ) -> None:
        self.name = str(name).strip().lower()
        self.cls: Type[Verifier] = get_verifier(self.name)
        self.configuration = dict(configuration or {})
        self.verdict_error_codes = {
            str(value)
            for value in self.configuration.pop("verdict_error_codes", ())
        }
        self.family = family or ("llm_judge" if self.name == "llm_judge" else "deterministic")
        if self.family not in VERIFIER_FAMILIES - {"chain", "reward_model"}:
            raise ValueError("registry verifier family must be deterministic or llm_judge")
        self.modalities = tuple(sorted({str(value) for value in modalities}))
        self.tasks = tuple(sorted({str(value) for value in tasks}))
        origin_by_name = {str(item["name"]): str(item["origin"]) for item in inventory()}
        self.implementation_fingerprint = fingerprint_verifier_class(
            self.name,
            self.cls,
            origin=origin_by_name.get(self.name),
        )
        self._instance: Optional[Verifier] = None

    def _verifier(self) -> Verifier:
        if self._instance is None:
            self._instance = self.cls(**self.configuration)
        return self._instance

    def capability(self) -> VerifierCapability:
        value = self.implementation_fingerprint
        return VerifierCapability(
            key=self.name,
            family=self.family,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            implementation=value.class_path,
            origin=value.origin,
            fingerprint=value.fingerprint,
            qualifiable=value.qualifiable,
            modalities=self.modalities,
            tasks=self.tasks,
            warning=value.reason,
        )

    def invoke(
        self,
        item: Mapping[str, Any],
        *,
        contract: RewardContract | Mapping[str, Any] | Any,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> VerifierObservation:
        verifier = self._verifier()
        task = self.tasks[0] if len(self.tasks) == 1 else ""
        candidates = list(item.get("candidates") or ())
        if self.family == "llm_judge" and task in {"pairwise", "ranking"} and candidates:
            reward_contract = normalize_reward_contract(contract)
            scoring_contract = RewardContract(
                minimum=reward_contract.minimum,
                maximum=reward_contract.maximum,
                direction=reward_contract.direction,
                threshold=None,
                tie_policy="tie",
                error_behavior=reward_contract.error_behavior,
                probability_semantics=reward_contract.probability_semantics,
            )
            observations = [
                self.invoke(
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "candidates"
                        },
                        "candidate": self._candidate_text(value),
                        "prompt": item.get("prompt", item.get("input")),
                    },
                    contract=scoring_contract,
                    runtime=runtime,
                )
                for value in candidates
            ]
            failed = next((value for value in observations if value.error), None)
            if failed is not None:
                return VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    details={"message": "candidate scoring failed"},
                    component_trace=tuple(
                        {
                            "candidate_index": index,
                            "observation": value.to_dict(),
                        }
                        for index, value in enumerate(observations)
                    ),
                    error=failed.error,
                    runtime_identity=runtime_identity(runtime),
                )
            scores = [float(value.reward) for value in observations if value.reward is not None]
            if len(scores) != len(candidates):
                return VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    error="missing candidate score",
                    runtime_identity=runtime_identity(runtime),
                )
            order = sorted(
                range(len(candidates)), key=lambda index: scores[index], reverse=True
            )
            reward = reward_contract.validate_reward(max(scores))
            try:
                passed = reward_contract.classify(reward)
            except Exception as exc:
                return VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    details={"candidate_scores": scores},
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=runtime_identity(runtime),
                )
            parsed: Any = (
                self._candidate_text(candidates[order[0]])
                if task == "pairwise"
                else [self._candidate_text(candidates[index]) for index in order]
            )
            return VerifierObservation(
                reward=reward,
                passed=passed,
                parsed_value=parsed,
                raw_output={"candidate_scores": scores},
                details={"aggregation": "independent_candidate_scoring"},
                component_trace=tuple(
                    {
                        "candidate_index": index,
                        "observation": value.to_dict(),
                    }
                    for index, value in enumerate(observations)
                ),
                latency_ms=sum(value.latency_ms or 0.0 for value in observations),
                runtime_identity=runtime_identity(runtime),
            )
        candidate = str(
            item.get("candidate", item.get("output", item.get("response", item.get("code", ""))))
        )
        prompt = item.get("prompt", item.get("input"))
        started = time.perf_counter()
        result: Any = None
        for attempt in range(3):
            try:
                if prompt is not None and callable(
                    getattr(verifier, "verify_with_prompt", None)
                ):
                    result = verifier.verify_with_prompt(candidate, prompt=str(prompt))  # type: ignore[attr-defined]
                elif prompt is not None:
                    result = verifier._verify_with_prompt(candidate, str(prompt))
                else:
                    result = verifier.verify(candidate)
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                reward_contract = normalize_reward_contract(contract)
                return VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    details={"message": "verifier invocation failed"},
                    latency_ms=elapsed,
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=runtime_identity(runtime),
                )
            detail_text = str(getattr(result, "details", "") or "").lower()
            rate_limited = getattr(result, "error", None) == "judge_failure" and any(
                token in detail_text
                for token in ("429", "rate limit", "too many requests", "throttl")
            )
            if not rate_limited or attempt == 2:
                break
            time.sleep(float(2**attempt))
        elapsed = (time.perf_counter() - started) * 1000.0
        legacy_error = getattr(result, "error", None)
        operational_by_verifier = {
            "llm_judge": {
                "judge_failure": "provider_error",
                "unparseable_score": "parse_error",
            },
            "json_schema": {"jsonschema_unavailable": "runtime_error"},
        }
        error_kind = operational_by_verifier.get(self.name, {}).get(
            str(legacy_error or "")
        )
        if (
            error_kind is None
            and legacy_error is not None
            and self.implementation_fingerprint.origin in {"user_plugin", "entry_point"}
        ):
            if str(legacy_error) not in self.verdict_error_codes:
                error_kind = "plugin_error"
        if error_kind:
            reward_contract = normalize_reward_contract(contract)
            return VerifierObservation(
                reward=None,
                passed=False if reward_contract.fails_closed else None,
                raw_output=getattr(result, "raw_output", None),
                details={
                    "message": getattr(result, "details", None),
                    "legacy_reward_rejected": getattr(result, "reward", None),
                    "error_kind": error_kind,
                },
                latency_ms=elapsed,
                error=f"{error_kind}:{legacy_error}",
                runtime_identity=runtime_identity(runtime),
            )
        reward_contract = normalize_reward_contract(contract)
        normalization_contract: RewardContract | Mapping[str, Any] | Any = contract
        if self.family == "llm_judge":
            span = reward_contract.maximum - reward_contract.minimum
            normalization_contract = RewardContract(
                minimum=0.0,
                maximum=1.0,
                direction=reward_contract.direction,
                threshold=(
                    None
                    if reward_contract.threshold is None
                    else (reward_contract.threshold - reward_contract.minimum) / span
                ),
                tie_policy=reward_contract.tie_policy,
                error_behavior=reward_contract.error_behavior,
                probability_semantics=reward_contract.probability_semantics,
            )
        observation = normalize_verifier_result(
            result,
            contract=normalization_contract,
            latency_ms=elapsed,
            runtime_identity=runtime_identity(runtime),
        )
        if self.family == "llm_judge" and observation.reward is not None:
            # The historical judge emits a normalized [0, 1] reward. V7's
            # immutable reward contract is the declared public scale, so map
            # the normalized score into that scale before calibration.
            scaled_reward = reward_contract.minimum + float(observation.reward) * (
                reward_contract.maximum - reward_contract.minimum
            )
            scaled_reward = reward_contract.validate_reward(scaled_reward)
            observation = VerifierObservation(
                reward=scaled_reward,
                passed=reward_contract.classify(scaled_reward),
                parsed_value=observation.parsed_value,
                raw_output=observation.raw_output,
                details=observation.details,
                component_trace=observation.component_trace,
                latency_ms=observation.latency_ms,
                error=observation.error,
                runtime_identity=observation.runtime_identity,
            )
        return observation

    @staticmethod
    def _candidate_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("text", "content", "response", "output", "candidate"):
                if value.get(key) is not None:
                    return str(value[key])
        return str(value)


class CallableReliabilityAdapter(VerifierReliabilityAdapter):
    """Runtime adapter for reward-model and hosted judge integrations."""

    adapter_id = "callable_reliability"
    adapter_version = "1"

    def __init__(
        self,
        key: str,
        invoker: ReliabilityInvoker,
        *,
        family: str,
        implementation_fingerprint: str,
        modalities: Iterable[str] = ("text",),
        tasks: Iterable[str] = ("scalar", "pairwise"),
        runtime_requirements: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if family not in {"llm_judge", "reward_model"}:
            raise ValueError("callable family must be llm_judge or reward_model")
        if not implementation_fingerprint:
            raise ValueError("callable implementations require a pinned fingerprint")
        self.key = str(key).strip().lower()
        self.invoker = invoker
        self.family = family
        self.fingerprint = str(implementation_fingerprint)
        self.modalities = tuple(sorted({str(value) for value in modalities}))
        self.tasks = tuple(sorted({str(value) for value in tasks}))
        self.runtime_requirements = dict(runtime_requirements or {})

    def capability(self) -> VerifierCapability:
        return VerifierCapability(
            key=self.key,
            family=self.family,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            implementation=getattr(self.invoker, "__qualname__", type(self.invoker).__qualname__),
            origin="runtime",
            fingerprint=self.fingerprint,
            qualifiable=True,
            modalities=self.modalities,
            tasks=self.tasks,
            runtime_requirements=self.runtime_requirements,
        )

    def invoke(
        self,
        item: Mapping[str, Any],
        *,
        contract: RewardContract | Mapping[str, Any] | Any,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> VerifierObservation:
        started = time.perf_counter()
        result: Any = None
        for attempt in range(3):
            try:
                result = self.invoker(item)
                break
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                rate_limited = any(
                    token in detail.lower()
                    for token in ("429", "rate limit", "too many requests", "throttl")
                )
                if rate_limited and attempt < 2:
                    time.sleep(float(2**attempt))
                    continue
                elapsed = (time.perf_counter() - started) * 1000.0
                reward_contract = normalize_reward_contract(contract)
                return VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    details={"message": "verifier invocation failed"},
                    latency_ms=elapsed,
                    error=detail,
                    runtime_identity=runtime_identity(runtime),
                )
        elapsed = (time.perf_counter() - started) * 1000.0
        return normalize_verifier_result(
            result,
            contract=contract,
            latency_ms=elapsed,
            runtime_identity=runtime_identity(runtime),
        )


class ArtifactRewardModelReliabilityAdapter(VerifierReliabilityAdapter):
    """Lazy local scorer for verified Hugging Face reward-model artifacts."""

    adapter_id = "artifact_reward_model"
    adapter_version = "1"

    def __init__(
        self,
        *,
        key: str,
        model_path: str | Path,
        content_hash: str,
        modality: str = "text",
        task_type: str = "scalar",
        tokenizer_revision: Optional[str] = None,
    ) -> None:
        self.key = str(key)
        self.model_path = Path(model_path).expanduser().resolve()
        self.content_hash = str(content_hash)
        self.modality = str(modality)
        self.task_type = str(task_type)
        self.tokenizer_revision = tokenizer_revision
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def capability(self) -> VerifierCapability:
        return VerifierCapability(
            key=self.key,
            family="reward_model",
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            implementation="transformers.AutoModelForSequenceClassification",
            origin="artifact_library",
            fingerprint=self.content_hash,
            qualifiable=True,
            modalities=(self.modality,),
            tasks=(self.task_type,),
            runtime_requirements={"local_model": True, "accelerator_optional": True},
        )

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._tokenizer, self._model
        if not self.model_path.exists():
            raise FileNotFoundError(f"reward-model artifact is missing: {self.model_path}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_source = self.tokenizer_revision or str(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and bool(getattr(torch.backends, "mps", None)):
            try:
                if torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass
        model.to(device)
        model.eval()
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        return torch, tokenizer, model

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("text", "content", "response", "output", "candidate"):
                if value.get(key) is not None:
                    return str(value[key])
        return str(value)

    def _score_many(self, texts: Sequence[str], *, batch_size: int) -> list[float]:
        torch, tokenizer, model = self._ensure_loaded()
        device = next(model.parameters()).device
        values: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, int(batch_size))):
                batch = list(texts[start : start + max(1, int(batch_size))])
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = model(**encoded).logits.detach().float().cpu()
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                if logits.shape[-1] != 1:
                    raise ValueError(
                        "reward-model artifact must expose one scalar logit per record"
                    )
                values.extend(float(value) for value in logits[:, 0].tolist())
        return values

    def invoke(
        self,
        item: Mapping[str, Any],
        *,
        contract: RewardContract | Mapping[str, Any] | Any,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> VerifierObservation:
        reward_contract = normalize_reward_contract(contract)
        batch_size = max(
            1,
            int(item.get("batch_size") or item.get("production_batch_size") or 1),
        )
        candidates = list(item.get("candidates") or ())
        prompt = str(item.get("prompt") or item.get("input") or "")

        def rendered(value: Any) -> str:
            candidate = self._text(value)
            return f"{prompt}\n{candidate}" if prompt else candidate

        started = time.perf_counter()
        try:
            if self.task_type in {"pairwise", "ranking"}:
                if len(candidates) < 2:
                    raise ValueError(
                        f"{self.task_type} reward-model evidence requires candidates"
                    )
                scores = self._score_many(
                    [rendered(value) for value in candidates], batch_size=batch_size
                )
                raw_scores = scores
                order = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)
                if self.task_type == "pairwise":
                    difference = scores[0] - scores[1]
                    reward = 1.0 / (1.0 + math.exp(-difference))
                    parsed: Any = self._text(candidates[order[0]])
                else:
                    reward = reward_contract.maximum
                    parsed = [self._text(candidates[index]) for index in order]
            else:
                candidate = item.get(
                    "candidate",
                    item.get("output", item.get("response", item.get("text", ""))),
                )
                score = self._score_many([rendered(candidate)], batch_size=batch_size)[0]
                raw_scores = [score]
                reward = 1.0 / (1.0 + math.exp(-score)) if reward_contract.probability_semantics else score
                parsed = reward
            reward = reward_contract.validate_reward(reward)
            passed = reward_contract.classify(reward)
            return VerifierObservation(
                reward=reward,
                passed=passed,
                parsed_value=parsed,
                raw_output={"artifact_scores": raw_scores},
                details={"artifact_hash": self.content_hash, "batch_size": batch_size},
                latency_ms=(time.perf_counter() - started) * 1000.0,
                runtime_identity=runtime_identity(
                    {**dict(runtime or {}), "artifact_hash": self.content_hash}
                ),
            )
        except Exception as exc:
            return VerifierObservation(
                reward=None,
                passed=False if reward_contract.fails_closed else None,
                details={"artifact_hash": self.content_hash, "batch_size": batch_size},
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                runtime_identity=runtime_identity(
                    {**dict(runtime or {}), "artifact_hash": self.content_hash}
                ),
            )

    def invoke_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        contract: RewardContract | Mapping[str, Any] | Any,
        batch_size: int,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> list[VerifierObservation]:
        """Score distinct records in a real production-sized tensor batch."""

        reward_contract = normalize_reward_contract(contract)
        rendered: list[str] = []
        spans: list[tuple[int, int, list[Any]]] = []
        for item in items:
            prompt = str(item.get("prompt") or item.get("input") or "")
            candidates = list(item.get("candidates") or ())
            if self.task_type in {"pairwise", "ranking"}:
                if len(candidates) < 2:
                    raise ValueError(
                        f"{self.task_type} reward-model evidence requires candidates"
                    )
                start = len(rendered)
                for candidate in candidates:
                    candidate_text = self._text(candidate)
                    rendered.append(
                        f"{prompt}\n{candidate_text}" if prompt else candidate_text
                    )
                spans.append((start, len(rendered), candidates))
            else:
                candidate = item.get(
                    "candidate",
                    item.get("output", item.get("response", item.get("text", ""))),
                )
                candidate_text = self._text(candidate)
                start = len(rendered)
                rendered.append(
                    f"{prompt}\n{candidate_text}" if prompt else candidate_text
                )
                spans.append((start, len(rendered), [candidate]))
        started = time.perf_counter()
        try:
            scores = self._score_many(rendered, batch_size=max(1, int(batch_size)))
            if len(scores) != len(rendered):
                raise ValueError("reward-model batch returned the wrong number of scores")
            latency = (time.perf_counter() - started) * 1000.0
            observations: list[VerifierObservation] = []
            for start_index, end_index, candidates in spans:
                item_scores = scores[start_index:end_index]
                if self.task_type == "pairwise":
                    difference = item_scores[0] - item_scores[1]
                    reward = 1.0 / (1.0 + math.exp(-difference))
                    order = sorted(
                        range(len(candidates)),
                        key=lambda index: item_scores[index],
                        reverse=True,
                    )
                    parsed: Any = self._text(candidates[order[0]])
                elif self.task_type == "ranking":
                    order = sorted(
                        range(len(candidates)),
                        key=lambda index: item_scores[index],
                        reverse=True,
                    )
                    reward = reward_contract.maximum
                    parsed = [self._text(candidates[index]) for index in order]
                else:
                    score = item_scores[0]
                    reward = (
                        1.0 / (1.0 + math.exp(-score))
                        if reward_contract.probability_semantics
                        else score
                    )
                    parsed = reward
                reward = reward_contract.validate_reward(reward)
                observations.append(
                    VerifierObservation(
                        reward=reward,
                        passed=reward_contract.classify(reward),
                        parsed_value=parsed,
                        raw_output={"artifact_scores": item_scores},
                        details={
                            "artifact_hash": self.content_hash,
                            "batch_size": max(1, int(batch_size)),
                            "true_batch_record_count": len(items),
                        },
                        latency_ms=latency / max(1, len(items)),
                        runtime_identity=runtime_identity(
                            {**dict(runtime or {}), "artifact_hash": self.content_hash}
                        ),
                    )
                )
            return observations
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000.0
            return [
                VerifierObservation(
                    reward=None,
                    passed=False if reward_contract.fails_closed else None,
                    details={
                        "artifact_hash": self.content_hash,
                        "batch_size": max(1, int(batch_size)),
                        "true_batch_record_count": len(items),
                    },
                    latency_ms=latency / max(1, len(items)),
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=runtime_identity(
                        {**dict(runtime or {}), "artifact_hash": self.content_hash}
                    ),
                )
                for _ in items
            ]
class VerifierReliabilityAdapterRegistry:
    """Version-aware adapter catalog used by calibration and guided pickers."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], VerifierReliabilityAdapter] = {}

    def register(self, adapter: VerifierReliabilityAdapter) -> None:
        capability = adapter.capability()
        key = (capability.key, capability.adapter_version)
        if key in self._adapters and self._adapters[key] is not adapter:
            raise ValueError(f"reliability adapter already registered for {key[0]} v{key[1]}")
        self._adapters[key] = adapter

    def get(self, key: str, *, version: Optional[str] = None) -> VerifierReliabilityAdapter:
        canonical = str(key).strip().lower()
        candidates = [
            (adapter_version, adapter)
            for (name, adapter_version), adapter in self._adapters.items()
            if name == canonical
        ]
        if not candidates:
            raise KeyError(f"unknown verifier reliability adapter {key!r}")
        if version is not None:
            for adapter_version, adapter in candidates:
                if adapter_version == str(version):
                    return adapter
            raise KeyError(f"unknown verifier reliability adapter {key!r} version {version!r}")
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    def capabilities(self) -> list[dict[str, Any]]:
        return [
            adapter.capability().to_dict()
            for _, adapter in sorted(self._adapters.items(), key=lambda item: item[0])
        ]

    @classmethod
    def from_existing_registry(cls) -> "VerifierReliabilityAdapterRegistry":
        result = cls()
        for item in inventory():
            name = str(item["name"])
            result.register(
                RegistryVerifierReliabilityAdapter(
                    name,
                    family="llm_judge" if name == "llm_judge" else "deterministic",
                    modalities=("text", "tool", "vlm", "audio")
                    if name == "llm_judge"
                    else ("text",),
                    tasks=("scalar", "pairwise", "ranking")
                    if name == "llm_judge"
                    else ("binary",),
                )
            )
        return result


def implementation_fingerprint(adapter: VerifierReliabilityAdapter) -> Optional[str]:
    """Convenience accessor for resolved profile construction."""

    return adapter.capability().fingerprint


__all__ = [
    "ArtifactRewardModelReliabilityAdapter",
    "CallableReliabilityAdapter",
    "RegistryVerifierReliabilityAdapter",
    "ReliabilityInvoker",
    "VERIFIER_FAMILIES",
    "VerifierCapability",
    "VerifierReliabilityAdapter",
    "VerifierReliabilityAdapterRegistry",
    "implementation_fingerprint",
]
