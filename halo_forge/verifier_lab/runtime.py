"""Runtime bridge from immutable verifier revisions to legacy trainers.

The training implementations historically consume the small ``Verifier`` ABC.
Verifier Reliability deliberately does not change that interface.  This module
therefore adapts one exact profile revision to ``VerifyResult`` and, when a
trainer resolves verifiers by registry name, installs a process-local class
whose identity is derived from the immutable revision hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.run_db import get_database


def _candidate_payload(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the call shapes used by RAFT, GRPO, and modality trainers."""

    payload = dict(kwargs)
    candidate = next(
        (
            payload.get(key)
            for key in ("candidate", "completion", "output", "response", "code", "prediction")
            if payload.get(key) is not None
        ),
        args[0] if args else "",
    )
    prompt = next(
        (
            payload.get(key)
            for key in ("prompt", "input", "question", "instruction")
            if payload.get(key) is not None
        ),
        None,
    )
    expected = next(
        (
            payload.get(key)
            for key in (
                "expected",
                "expected_answer",
                "expected_calls",
                "ground_truth",
                "reference",
                "label",
            )
            if payload.get(key) is not None
        ),
        args[1] if len(args) > 1 else None,
    )
    payload.update(
        candidate=candidate,
        completion=candidate,
        output=candidate,
        response=candidate,
    )
    if prompt is not None:
        payload.setdefault("prompt", prompt)
        payload.setdefault("input", prompt)
    if expected is not None:
        payload.setdefault("expected", expected)
        payload.setdefault("reference", expected)
    return payload


def _mapped_payload(payload: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the revision's small target-to-source field mapping."""

    result = dict(payload)
    record = payload.get("record") if isinstance(payload.get("record"), Mapping) else payload
    for target, source in mapping.items():
        current: Any = record
        for part in str(source).split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            result[str(target)] = current
    return result


class ProfileRevisionVerifier(Verifier):
    """Invoke one exact verifier profile revision through the v7 service."""

    def __init__(
        self,
        revision_id: str,
        *,
        database: Optional[str] = None,
        max_workers: int = 1,
    ) -> None:
        super().__init__(max_workers=max_workers)
        from halo_forge.verifier_lab.service import VerifierLabService

        self.revision_id = str(revision_id)
        self.database = database
        self.service = VerifierLabService(get_database(database))
        self.revision = self.service.store.get_profile_revision(self.revision_id)
        # Resolve and verify the exact implementation/artifact once at the
        # trainer boundary. Per-sample re-hashing would make local reward
        # models unusable, while omitting this check would allow silent drift.
        self.service._assert_implementation_identity(self.revision)

    def verify(self, *args: Any, **kwargs: Any) -> VerifyResult:
        payload = _mapped_payload(_candidate_payload(args, kwargs), self.revision.input_mapping)
        observation = self.service.invoke_revision(self.revision_id, payload)
        contract = self.revision.reward_contract
        if observation.reward is None:
            reward = contract.minimum if contract.direction == "maximize" else contract.maximum
        else:
            reward = float(observation.reward)
        passed = observation.passed
        if passed is None and observation.reward is not None and contract.threshold is not None:
            passed = (
                reward >= contract.threshold
                if contract.direction == "maximize"
                else reward <= contract.threshold
            )
        if observation.error:
            passed = False
        details_value = observation.details or observation.parsed_value or observation.raw_output
        details = (
            details_value
            if isinstance(details_value, str)
            else json.dumps(details_value, sort_keys=True, default=str)
        )
        if not details:
            details = "Verifier profile revision completed"
        return VerifyResult(
            success=bool(passed),
            reward=reward,
            details=details,
            error=observation.error,
            metadata={
                "verifier_profile_revision_id": self.revision_id,
                "revision_hash": self.revision.content_hash,
                "parsed_value": observation.parsed_value,
                "component_trace": list(observation.component_trace),
                "latency_ms": observation.latency_ms,
                "runtime_identity": dict(observation.runtime_identity),
            },
        )

    def _verify_with_prompt(self, candidate: str, prompt: str) -> VerifyResult:
        return self.verify(candidate=candidate, prompt=prompt)

    def verify_with_prompt(self, candidate: str, prompt: str) -> VerifyResult:
        return self._verify_with_prompt(candidate, prompt)


_REGISTERED_PROFILE_CLASSES: dict[tuple[str, str], tuple[str, type[Verifier]]] = {}


def register_profile_verifier(
    revision_id: str,
    *,
    database: Optional[str] = None,
) -> str:
    """Register a zero-argument bridge class and return its stable runtime name."""

    from halo_forge.verifier_lab.service import VerifierLabService
    from halo_forge.rlvr.verifiers import register_verifier

    db = get_database(database)
    revision = VerifierLabService(db).store.get_profile_revision(revision_id)
    key = (str(db.path), revision.id)
    cached = _REGISTERED_PROFILE_CLASSES.get(key)
    if cached is not None:
        return cached[0]
    suffix = hashlib.sha256(
        f"{db.path}:{revision.id}:{revision.content_hash}".encode("utf-8")
    ).hexdigest()[:16]
    name = f"profile_revision_{suffix}"

    class BoundProfileRevisionVerifier(ProfileRevisionVerifier):
        def __init__(self, max_workers: int = 1) -> None:
            super().__init__(revision.id, database=str(db.path), max_workers=max_workers)

    BoundProfileRevisionVerifier.__name__ = f"ProfileRevisionVerifier_{suffix}"
    BoundProfileRevisionVerifier.__qualname__ = BoundProfileRevisionVerifier.__name__
    register_verifier(name)(BoundProfileRevisionVerifier)
    _REGISTERED_PROFILE_CLASSES[key] = (name, BoundProfileRevisionVerifier)
    return name


__all__ = ["ProfileRevisionVerifier", "register_profile_verifier"]
