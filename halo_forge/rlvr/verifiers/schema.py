"""Schema-shaped verifiers (Track V3).

Three verifiers for outputs that have a *structural* contract rather
than a behavioral / programmatic one:

  - **json_structure** — candidate parses as JSON. The simplest schema
    check; useful for "make the model emit valid JSON".
  - **json_schema** — candidate parses as JSON *and* validates against
    a supplied JSON Schema. The right verifier for tool-calling +
    structured-output finetunes.
  - **regex_format** — candidate matches a configured regex. Useful
    for "the answer must look like ``Final answer: <number>``" or
    similar format-discipline rules.

All three plug into the V1 plugin registry via `@register_verifier`
so a trainer can pick them by short name from CLI / YAML / API.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Pattern, Union

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.rlvr.verifiers.registry import register_verifier

logger = logging.getLogger(__name__)


@register_verifier("json_structure")
class JSONStructureVerifier(Verifier):
    """Pass if the candidate parses as JSON; fail otherwise.

    Tolerant of leading / trailing whitespace and code-fence wrapping
    (```json ... ```) — common artifacts of chat-model output that we
    don't want to penalize when the actual JSON inside is valid.
    """

    def __init__(self, *, strip_code_fences: bool = True, max_workers: int = 8):
        super().__init__(max_workers=max_workers)
        self.strip_code_fences = strip_code_fences

    def verify(self, code: str) -> VerifyResult:
        text = (code or "").strip()
        if not text:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty response", error="empty_response",
            )
        if self.strip_code_fences:
            text = _strip_code_fence(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return VerifyResult(
                success=False, reward=0.0,
                details=f"Invalid JSON: {exc}",
                error="invalid_json",
            )
        return VerifyResult(
            success=True, reward=1.0,
            details=f"Valid JSON ({type(parsed).__name__})",
        )


@register_verifier("json_schema")
class JSONSchemaVerifier(Verifier):
    """Pass if the candidate is JSON *and* validates against a schema.

    Uses the ``jsonschema`` package (lazy-imported so the module loads
    on installs without it). Tool-calling and structured-output
    workflows are the canonical use cases.

    Args:
        schema: A JSON Schema dict (Draft 7+ supported by ``jsonschema``).
        partial_credit: If True, an output that's valid JSON but fails
            schema validation earns reward=0.5; if False, fails with
            reward=0.0. Useful in RAFT where partial signal helps.
    """

    def __init__(
        self,
        *,
        schema: Optional[dict] = None,
        partial_credit: bool = True,
        max_workers: int = 8,
    ):
        super().__init__(max_workers=max_workers)
        self.schema = schema or {}
        self.partial_credit = partial_credit

    def verify(self, code: str) -> VerifyResult:
        text = _strip_code_fence((code or "").strip())
        if not text:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty response", error="empty_response",
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return VerifyResult(
                success=False, reward=0.0,
                details=f"Invalid JSON: {exc}",
                error="invalid_json",
            )

        if not self.schema:
            # No schema configured → fall back to structure-only check.
            return VerifyResult(
                success=True, reward=1.0,
                details="No schema configured; valid JSON.",
            )

        try:
            import jsonschema
        except ImportError:
            return VerifyResult(
                success=False, reward=0.5 if self.partial_credit else 0.0,
                details=(
                    "jsonschema package not installed; can't validate. "
                    "Install with `pip install jsonschema`."
                ),
                error="jsonschema_unavailable",
            )

        try:
            jsonschema.validate(parsed, self.schema)
        except jsonschema.ValidationError as exc:
            return VerifyResult(
                success=False,
                reward=0.5 if self.partial_credit else 0.0,
                details=f"Schema validation failed: {exc.message}",
                error="schema_invalid",
            )

        return VerifyResult(
            success=True, reward=1.0,
            details="JSON valid + schema-compliant",
        )


@register_verifier("regex_format")
class RegexFormatVerifier(Verifier):
    """Pass if the candidate matches a configured regex.

    The regex is compiled once at constructor time so per-call
    verification is fast. Both ``re.search`` (default) and ``re.fullmatch``
    semantics are exposed via the ``full_match`` flag.

    Args:
        pattern: Regex string or pre-compiled `re.Pattern`.
        flags: int flags to pass to `re.compile` (e.g. `re.MULTILINE`).
            Ignored when `pattern` is already compiled.
        full_match: When True, requires the entire candidate to match
            the pattern; when False (default), any substring match passes.
        partial_credit: When True, a near-miss (pattern present but at
            wrong position under full_match=True) earns 0.5 reward.
    """

    def __init__(
        self,
        *,
        pattern: Union[str, Pattern[str]] = "",
        flags: int = 0,
        full_match: bool = False,
        partial_credit: bool = False,
        max_workers: int = 8,
    ):
        super().__init__(max_workers=max_workers)
        if not pattern:
            raise ValueError("regex_format verifier requires a non-empty pattern")
        if isinstance(pattern, str):
            self.pattern: Pattern[str] = re.compile(pattern, flags)
        else:
            self.pattern = pattern
        self.full_match = full_match
        self.partial_credit = partial_credit

    def verify(self, code: str) -> VerifyResult:
        text = code or ""
        if not text:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty response", error="empty_response",
            )

        if self.full_match:
            match = self.pattern.fullmatch(text)
            if match is not None:
                return VerifyResult(
                    success=True, reward=1.0,
                    details=f"Full match against /{self.pattern.pattern}/",
                )
            partial = self.pattern.search(text)
            if partial is not None and self.partial_credit:
                return VerifyResult(
                    success=False, reward=0.5,
                    details=(
                        f"Pattern found at offset {partial.start()} but "
                        f"not full-match"
                    ),
                    error="partial_match",
                )
            return VerifyResult(
                success=False, reward=0.0,
                details=f"No match for /{self.pattern.pattern}/",
                error="no_match",
            )

        # search semantics
        match = self.pattern.search(text)
        if match is not None:
            return VerifyResult(
                success=True, reward=1.0,
                details=f"Matched at offset {match.start()}",
            )
        return VerifyResult(
            success=False, reward=0.0,
            details=f"No match for /{self.pattern.pattern}/",
            error="no_match",
        )


def _strip_code_fence(text: str) -> str:
    """Trim a leading ```json … ``` (or any language tag) wrapper.

    Common artifact of chat-model output. We don't want a model that
    correctly produced JSON to fail because it wrapped it in markdown.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first line (```lang) and any trailing ```
        lines = stripped.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:])
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[: -len("```")].rstrip()
            return inner.strip()
    return stripped


__all__ = [
    "JSONStructureVerifier",
    "JSONSchemaVerifier",
    "RegexFormatVerifier",
]
