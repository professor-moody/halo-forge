"""Atomic, verified publication primitives for local evidence bundles."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ._canonical import canonical_json


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"bundle file path must be a safe relative path: {value!r}")
    return path


def _file_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return (canonical_json(value) + "\n").encode("utf-8")


@dataclass(frozen=True)
class PublishedEvidenceBundle:
    path: Path
    content_hash: str
    manifest: Dict[str, Any]
    size_bytes: int
    reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "content_hash": self.content_hash,
            "manifest": dict(self.manifest),
            "size_bytes": self.size_bytes,
            "reused": self.reused,
        }


def verify_evidence_bundle(
    path: str | Path, *, content_hash: Optional[str] = None
) -> PublishedEvidenceBundle:
    """Verify a published manifest and every declared file checksum."""

    target = Path(path).expanduser()
    try:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"evidence bundle manifest is not readable: {target}") from exc
    actual_hash = str(manifest.get("content_hash") or "")
    if not actual_hash or (content_hash is not None and actual_hash != content_hash):
        raise RuntimeError(f"evidence bundle content identity does not match: {target}")
    inventory = manifest.get("files")
    if not isinstance(inventory, Mapping):
        raise RuntimeError(f"evidence bundle file inventory is missing: {target}")
    size = (target / "manifest.json").stat().st_size
    for name, raw_expected in inventory.items():
        relative = _safe_relative_path(str(name))
        if not isinstance(raw_expected, Mapping):
            raise RuntimeError(f"invalid evidence inventory entry: {name}")
        candidate = target / relative
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"evidence bundle file is missing: {relative}") from exc
        expected_hash = str(raw_expected.get("sha256") or "")
        expected_size = raw_expected.get("size_bytes")
        if _sha256(data) != expected_hash or len(data) != expected_size:
            raise RuntimeError(f"evidence bundle file verification failed: {relative}")
        size += len(data)
    return PublishedEvidenceBundle(target, actual_hash, dict(manifest), size, reused=True)


def markdown_report_html(markdown: str, *, title: str = "Halo Forge Evidence") -> str:
    """Render a dependency-free, faithful HTML view of a Markdown report."""

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{max-width:960px;margin:2rem auto;padding:0 1rem;"
        "font:15px/1.55 system-ui,sans-serif;color:#17202a}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head>"
        f"<body><pre>{html.escape(markdown)}</pre></body></html>\n"
    )


def comparison_interval_svg(comparisons: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a deterministic, dependency-free matched-delta interval plot."""

    rows = [(str(key), dict(value)) for key, value in sorted(comparisons.items())]
    width = 900
    row_height = 46
    height = max(150, 90 + len(rows) * row_height)
    plot_left, plot_right = 250, width - 40
    numeric: list[float] = [0.0]
    for _, value in rows:
        interval = value.get("confidence_interval") or {}
        for candidate in (
            value.get("mean_delta"),
            interval.get("lower"),
            interval.get("upper"),
        ):
            if isinstance(candidate, (int, float)):
                numeric.append(float(candidate))
    extent = max(abs(value) for value in numeric) or 1.0

    def x(value: float) -> float:
        return plot_left + ((float(value) + extent) / (2.0 * extent)) * (plot_right - plot_left)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Matched-seed directional deltas</title>',
        '<desc id="desc">Mean direction-normalized delta and bootstrap interval by candidate.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="30" font-family="system-ui" font-size="18" '
        'fill="#17202a">Matched-seed directional deltas</text>',
        f'<line x1="{x(0):.2f}" y1="48" x2="{x(0):.2f}" y2="{height - 32}" '
        'stroke="#8a94a3" stroke-width="1"/>',
    ]
    if not rows:
        elements.append(
            '<text x="20" y="80" font-family="system-ui" font-size="14" '
            'fill="#5f6b7a">No baseline comparisons are available.</text>'
        )
    for index, (subject_id, value) in enumerate(rows):
        y = 70 + index * row_height
        label = html.escape(subject_id)
        classification = html.escape(str(value.get("classification") or "unknown"))
        elements.append(
            f'<text x="20" y="{y + 5}" font-family="system-ui" font-size="13" '
            f'fill="#17202a">{label} · {classification}</text>'
        )
        interval = value.get("confidence_interval") or {}
        mean = value.get("mean_delta")
        lower, upper = interval.get("lower"), interval.get("upper")
        if all(isinstance(item, (int, float)) for item in (mean, lower, upper)):
            elements.extend(
                [
                    f'<line x1="{x(float(lower)):.2f}" y1="{y}" '
                    f'x2="{x(float(upper)):.2f}" y2="{y}" '
                    'stroke="#2f6feb" stroke-width="3"/>',
                    f'<circle cx="{x(float(mean)):.2f}" cy="{y}" r="5" ' 'fill="#2f6feb"/>',
                ]
            )
        else:
            elements.append(
                f'<text x="{plot_left}" y="{y + 5}" font-family="system-ui" '
                'font-size="12" fill="#7a8492">insufficient evidence</text>'
            )
    elements.append("</svg>\n")
    return "".join(elements)


def publish_evidence_bundle(
    destination: str | Path,
    *,
    content_hash: str,
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
) -> PublishedEvidenceBundle:
    """Write, verify, and atomically expose one evidence directory.

    Staging is a sibling of the destination, which guarantees that the final
    rename remains on one filesystem. An already-published directory is reused
    only when its manifest carries the exact requested content identity.
    """

    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return verify_evidence_bundle(target, content_hash=content_hash)

    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        checksums: Dict[str, Dict[str, Any]] = {}
        for name in sorted(files):
            relative = _safe_relative_path(name)
            if relative.as_posix() == "manifest.json":
                raise ValueError("manifest.json is reserved by the evidence publisher")
            data = _file_bytes(files[name])
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            checksums[relative.as_posix()] = {"sha256": _sha256(data), "size_bytes": len(data)}

        resolved_manifest = dict(manifest)
        resolved_manifest["content_hash"] = content_hash
        resolved_manifest["files"] = checksums
        manifest_data = (canonical_json(resolved_manifest) + "\n").encode("utf-8")
        (staging / "manifest.json").write_bytes(manifest_data)

        for relative_name, expected in checksums.items():
            data = (staging / relative_name).read_bytes()
            if _sha256(data) != expected["sha256"] or len(data) != expected["size_bytes"]:
                raise RuntimeError(f"evidence file verification failed: {relative_name}")
        parsed = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if parsed != resolved_manifest:
            raise RuntimeError("evidence manifest verification failed")

        os.replace(staging, target)
        size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        return PublishedEvidenceBundle(target, content_hash, resolved_manifest, size)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "PublishedEvidenceBundle",
    "comparison_interval_svg",
    "markdown_report_html",
    "publish_evidence_bundle",
    "verify_evidence_bundle",
]
