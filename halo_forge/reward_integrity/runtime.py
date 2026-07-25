"""Execution bridge from immutable reward systems to existing verifier runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Mapping, Optional

from halo_forge.run_db import RunDatabase

from .models import RewardIntegritySample
from .service import RewardIntegrityService


def _mapped_reward(result: Any, mapping: Mapping[str, Any]) -> Any:
    """Apply the immutable optimizer reward mapping without hiding failures."""

    from halo_forge.rlvr.verifiers.base import VerifyResult

    raw = float(result.reward)
    normalization = dict(mapping.get("normalization") or {})
    minimum = float(normalization.get("minimum", normalization.get("min", 0.0)))
    maximum = float(normalization.get("maximum", normalization.get("max", 1.0)))
    direction = str(normalization.get("direction") or "maximize")
    if not math.isfinite(raw) or not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("reward mapping requires finite values")
    if maximum <= minimum or direction not in {"maximize", "minimize"}:
        raise ValueError("reward normalization contract is invalid")
    if raw < minimum or raw > maximum:
        raise ValueError(
            f"optimizer reward {raw} is outside [{minimum}, {maximum}]"
        )
    normalized = (raw - minimum) / (maximum - minimum)
    if direction == "minimize":
        normalized = 1.0 - normalized
    scaling = mapping.get("scaling", 1.0)
    centering = mapping.get("centering", 0.0)
    if isinstance(scaling, Mapping):
        scaling = scaling.get("factor", scaling.get("scale", 1.0))
    if isinstance(centering, Mapping):
        centering = centering.get("value", centering.get("center", 0.0))
    scale = float(mapping.get("scale", scaling))
    center = float(mapping.get("center", centering))
    reward = normalized * scale + center
    output_min = float(mapping.get("minimum", 0.0))
    output_max = float(mapping.get("maximum", 1.0))
    if not math.isfinite(reward) or reward < output_min or reward > output_max:
        raise ValueError(
            f"mapped optimizer reward {reward} is outside [{output_min}, {output_max}]"
        )
    filtering = dict(mapping.get("filtering") or {})
    threshold_value = mapping.get("threshold", filtering.get("threshold"))
    success = result.success if threshold_value is None else reward >= float(threshold_value)
    if result.error:
        success = False
    metadata = dict(result.metadata or {})
    metadata.update(
        raw_optimizer_reward=raw,
        mapped_optimizer_reward=reward,
        reward_mapping_hash=hashlib.sha256(
            json.dumps(
                dict(mapping),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    )
    return VerifyResult(
        success=bool(success),
        reward=reward,
        details=result.details,
        error=result.error,
        metadata=metadata,
    )


class RewardMappedVerifier:
    """Small verifier-compatible facade applying one pinned reward mapping."""

    def __init__(self, verifier: Any, mapping: Mapping[str, Any]) -> None:
        self.verifier = verifier
        self.mapping = dict(mapping)
        self.max_workers = getattr(verifier, "max_workers", 1)

    def verify(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return _mapped_reward(self.verifier.verify(*args, **kwargs), self.mapping)
        except Exception as exc:
            from halo_forge.rlvr.verifiers.base import VerifyResult

            behavior = str(self.mapping.get("failure_behavior") or "reject")
            if behavior == "raise":
                raise
            return VerifyResult(
                success=False,
                reward=float(self.mapping.get("failure_reward", 0.0)),
                details="Reward-system mapping rejected the optimizer observation",
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "reward_mapping_error": True,
                    "reward_mapping_abstained": behavior == "abstain",
                },
            )

    def _verify_with_prompt(self, candidate: str, prompt: str) -> Any:
        return self.verify(candidate=candidate, prompt=prompt)

    def verify_with_prompt(self, candidate: str, prompt: str) -> Any:
        return self._verify_with_prompt(candidate, prompt)

    def verify_batch(self, codes: list[str], prompts: Optional[list[str]] = None) -> list[Any]:
        if prompts:
            return [
                self.verify(candidate=code, prompt=prompt)
                for code, prompt in zip(codes, prompts)
            ]
        return [self.verify(code) for code in codes]


_REGISTERED_REWARD_MAPPERS: dict[tuple[str, str, str], str] = {}


def register_reward_mapped_verifier(
    revision_id: str,
    mapping: Mapping[str, Any],
    *,
    database: Optional[str] = None,
) -> str:
    """Register a zero-argument mapped verifier for legacy name registries."""

    from halo_forge.rlvr.verifiers import register_verifier
    from halo_forge.verifier_lab.runtime import ProfileRevisionVerifier

    mapping_payload = json.dumps(
        dict(mapping), sort_keys=True, separators=(",", ":"), default=str
    )
    mapping_hash = hashlib.sha256(mapping_payload.encode("utf-8")).hexdigest()
    key = (str(database or ""), str(revision_id), mapping_hash)
    cached = _REGISTERED_REWARD_MAPPERS.get(key)
    if cached is not None:
        return cached
    suffix = hashlib.sha256(":".join(key).encode("utf-8")).hexdigest()[:16]
    name = f"reward_system_{suffix}"

    class BoundRewardMappedVerifier(ProfileRevisionVerifier):
        def __init__(self, max_workers: int = 1) -> None:
            super().__init__(revision_id, database=database, max_workers=max_workers)

        def verify(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return _mapped_reward(super().verify(*args, **kwargs), dict(mapping))
            except Exception as exc:
                from halo_forge.rlvr.verifiers.base import VerifyResult

                if str(mapping.get("failure_behavior") or "reject") == "raise":
                    raise
                behavior = str(mapping.get("failure_behavior") or "reject")
                return VerifyResult(
                    success=False,
                    reward=float(mapping.get("failure_reward", 0.0)),
                    details="Reward-system mapping rejected the optimizer observation",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={
                        "reward_mapping_error": True,
                        "reward_mapping_abstained": behavior == "abstain",
                    },
                )

    BoundRewardMappedVerifier.__name__ = f"RewardMappedVerifier_{suffix}"
    BoundRewardMappedVerifier.__qualname__ = BoundRewardMappedVerifier.__name__
    register_verifier(name)(BoundRewardMappedVerifier)
    _REGISTERED_REWARD_MAPPERS[key] = name
    return name


def verifier_input(sample: RewardIntegritySample) -> Dict[str, Any]:
    """Map one persisted sample without regenerating or substituting its output."""

    captured_input = dict(sample.input or {})
    prompt = captured_input.get("prompt", captured_input.get("input", captured_input))
    context = captured_input.get("context")
    return {
        **captured_input,
        "input": captured_input,
        "prompt": prompt,
        "context": context,
        "completion": sample.output,
        "output": sample.output,
        "response": sample.output,
        "expected": sample.expected,
        "reference": sample.expected,
        "media": list(sample.media),
        "generation": dict(sample.generation),
        "record_id": sample.record_id,
        "record_hash": sample.record_hash,
        "instance_id": sample.instance_id,
        "candidate_ordinal": sample.candidate_ordinal,
    }


def _callback(
    verifiers: Any, revision_id: str
) -> Callable[[RewardIntegritySample], Dict[str, Any]]:
    def invoke(sample: RewardIntegritySample) -> Dict[str, Any]:
        observation = verifiers.invoke_revision(revision_id, verifier_input(sample))
        value = observation.to_dict() if hasattr(observation, "to_dict") else dict(observation)
        value.setdefault("runtime_identity", {})
        value["runtime_identity"].setdefault("verifier_profile_revision_id", revision_id)
        return value

    return invoke


def execute_pinned_audit(
    database: RunDatabase,
    audit_id: str,
    *,
    root: str | None = None,
    bootstrap_resamples: int = 10_000,
) -> Any:
    """Execute an audit with the exact ordered auditors pinned by its system."""

    from halo_forge.verifier_lab import VerifierLabService

    service = RewardIntegrityService(database, root=root)
    audit = service.get_audit(audit_id)
    system = service.get_system_revision(audit.reward_system_revision_id)
    primary = system.primary_sentinel
    if primary is None:
        raise ValueError("reward system has no primary sentinel")
    # A sealed snapshot and every pinned auditor identity are scientific
    # prerequisites. Deterministic invalidity publishes a review gate; it is
    # not a provider failure and must not consume automatic retries.
    evidence_problems = service.audit_signal_evidence_problems(audit_id)
    runtime_problems = service.auditor_runtime_identity_problems(
        audit.reward_system_revision_id,
        audit.runtime_identity,
    )
    if evidence_problems or runtime_problems:
        return service.complete_incomplete_evidence(
            audit_id,
            reasons=[*evidence_problems, *runtime_problems],
            classification="audit_execution_identity_invalid",
        )
    service.development_evaluation_evidence(audit_id, require_complete=True)
    verifiers = VerifierLabService(database)
    diagnostics = sorted(
        (value for value in system.auditors if value.role == "diagnostic"),
        key=lambda value: value.ordinal,
    )
    return service.execute_audit(
        audit_id,
        sentinel=_callback(verifiers, primary.verifier_revision_id),
        diagnostic_auditors=[
            _callback(verifiers, value.verifier_revision_id) for value in diagnostics
        ],
        bootstrap_resamples=bootstrap_resamples,
    )


__all__ = [
    "RewardMappedVerifier",
    "execute_pinned_audit",
    "register_reward_mapped_verifier",
    "verifier_input",
]
