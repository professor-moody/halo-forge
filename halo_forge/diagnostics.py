"""Shared diagnostic taxonomy helpers for readiness and qualification surfaces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ISSUE_SCOPES = ("module", "cross_module")
ISSUE_SEVERITIES = ("info", "warning", "error")

_PATH_RE = re.compile(r"((?:[A-Za-z]:\\|/)[^,\s;]+)")


def derive_issue_metadata(
    *,
    module: str,
    issue_class: str,
    launch_blocked: bool,
    errors: Iterable[str],
    warnings: Iterable[str],
    action_hint: str = "",
    evidence: Mapping[str, Any] | None = None,
    last_output_dir: str = "",
) -> Dict[str, Any]:
    """Build normalized issue metadata used across reports and UI rendering."""
    module_key = _normalized_module_key(module)
    errors_list = [str(v) for v in errors if v is not None]
    warnings_list = [str(v) for v in warnings if v is not None]

    issue_scope = "cross_module" if module_key == "UI_OPS" else "module"
    severity = "info"
    if errors_list or launch_blocked:
        severity = "error"
    elif warnings_list:
        severity = "warning"

    issue_code = _issue_code(
        module_key=module_key,
        issue_class=str(issue_class or "none"),
        launch_blocked=bool(launch_blocked),
        has_errors=bool(errors_list),
        has_warnings=bool(warnings_list),
    )
    what_is_missing = _derive_missing_items(
        errors=errors_list,
        warnings=warnings_list,
        evidence=evidence or {},
        last_output_dir=last_output_dir,
    )

    if launch_blocked:
        fix_now = action_hint or "Resolve launch-blocking preflight issues, then retry."
    elif errors_list:
        fix_now = action_hint or "Repair malformed contract artifacts, then rerun validation."
    elif warnings_list:
        fix_now = action_hint or "Generate fresh evidence artifacts via a probe or bounded run."
    else:
        fix_now = "No action needed."

    fix_options = _default_fix_options(
        module=module,
        launch_blocked=bool(launch_blocked),
        has_warnings=bool(warnings_list),
    )
    return {
        "issue_code": issue_code,
        "issue_scope": issue_scope,
        "severity": severity,
        "what_is_missing": what_is_missing,
        "fix_now": fix_now,
        "fix_options": fix_options,
    }


def validate_issue_metadata_payload(
    payload: Mapping[str, Any],
    *,
    module: str,
) -> List[str]:
    """Validate additive issue metadata fields for schema compatibility."""
    errors: List[str] = []
    issue_code = payload.get("issue_code")
    if issue_code is not None and not isinstance(issue_code, str):
        errors.append(f"issue_code must be a string for {module}")

    issue_scope = payload.get("issue_scope")
    if issue_scope is not None and str(issue_scope) not in ISSUE_SCOPES:
        errors.append(f"invalid issue_scope for {module}: {issue_scope}")

    severity = payload.get("severity")
    if severity is not None and str(severity) not in ISSUE_SEVERITIES:
        errors.append(f"invalid severity for {module}: {severity}")

    missing = payload.get("what_is_missing")
    if missing is not None and not isinstance(missing, list):
        errors.append(f"what_is_missing must be a list for {module}")

    fix_now = payload.get("fix_now")
    if fix_now is not None and not isinstance(fix_now, str):
        errors.append(f"fix_now must be a string for {module}")

    fix_options = payload.get("fix_options")
    if fix_options is not None and not isinstance(fix_options, list):
        errors.append(f"fix_options must be a list for {module}")
    return errors


def _default_fix_options(*, module: str, launch_blocked: bool, has_warnings: bool) -> List[str]:
    key = str(module or "").strip().lower()
    options = [
        f"halo-forge test --level all-modules --module {key}",
        f"halo-forge test --level all-module-qualification --module {key}",
    ]
    if has_warnings:
        options.insert(0, f"Run a contract probe for `{key}` from the UI surface.")
    if launch_blocked:
        options.insert(0, "Correct required launch/preflight inputs before relaunch.")
    return options


def _derive_missing_items(
    *,
    errors: List[str],
    warnings: List[str],
    evidence: Mapping[str, Any],
    last_output_dir: str,
) -> List[str]:
    missing: List[str] = []
    for text in list(errors) + list(warnings):
        lowered = text.lower()
        if any(token in lowered for token in ("missing", "not found", "no ")):
            missing.append(text)
            path_match = _PATH_RE.search(text)
            if path_match:
                missing.append(path_match.group(1))

    for value in evidence.values():
        missing.extend(_collect_missing_paths(value))

    if last_output_dir:
        root = Path(last_output_dir)
        if not root.exists():
            missing.append(f"evidence root not found: {root}")

    deduped: List[str] = []
    for item in missing:
        text = str(item).strip()
        if not text:
            continue
        if text not in deduped:
            deduped.append(text)
    return deduped[:5]


def _collect_missing_paths(value: Any) -> List[str]:
    paths: List[str] = []
    if isinstance(value, Mapping):
        for item in value.values():
            paths.extend(_collect_missing_paths(item))
        return paths
    if isinstance(value, list):
        for item in value:
            paths.extend(_collect_missing_paths(item))
        return paths
    if not isinstance(value, str):
        return paths

    candidate = Path(value)
    if ("/" in value or "\\" in value) and not candidate.exists():
        paths.append(f"missing path: {value}")
    return paths


def _issue_code(
    *,
    module_key: str,
    issue_class: str,
    launch_blocked: bool,
    has_errors: bool,
    has_warnings: bool,
) -> str:
    if launch_blocked and has_errors:
        if issue_class == "preflight_blocker":
            base = "PREFLIGHT_BLOCKED"
        else:
            base = "CONTRACT_BLOCKED"
    elif has_errors:
        base = "CONTRACT_ERROR"
    elif has_warnings:
        base = "EVIDENCE_GAP"
    else:
        base = "READY_OK"
    return f"{module_key}_{base}"


def _normalized_module_key(module: str) -> str:
    key = str(module or "").strip().upper()
    key = key.replace("-", "_")
    return re.sub(r"[^A-Z0-9_]", "_", key)

