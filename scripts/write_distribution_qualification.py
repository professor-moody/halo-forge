#!/usr/bin/env python3
"""Write truthful, checksummed desktop distribution qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path


def candidates(root: Path) -> list[tuple[str, Path]]:
    system = platform.system()
    patterns = {
        "Darwin": (("dmg", "dmg/*.dmg"),),
        "Linux": (("deb", "deb/*.deb"), ("appimage", "appimage/*.AppImage")),
        "Windows": (("nsis", "nsis/*.exe"),),
    }.get(system, ())
    return [
        (kind, path)
        for kind, pattern in patterns
        for path in sorted(root.glob(pattern))
        if path.is_file()
    ]


def platform_name() -> str:
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        platform.system(), platform.system().lower() or "unknown"
    )


def architecture() -> str:
    return {
        "arm64": "arm64",
        "aarch64": "arm64",
        "AMD64": "x86_64",
        "x86_64": "x86_64",
    }.get(platform.machine(), platform.machine().lower() or "unknown")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_candidate(
    package_type: str,
    path: Path,
    *,
    signature_state: str,
    smoke_status: str,
) -> dict[str, object]:
    evidence = {
        "package_name": path.name,
        "package_sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    identity = {
        "platform": platform_name(),
        "architecture": architecture(),
        "package_type": package_type,
        "signature_state": signature_state,
        "smoke_status": smoke_status,
        "supported_backends": ["cpu"],
        "evidence": evidence,
    }
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": f"release-qualification-{uuid.uuid4().hex}",
        **identity,
        "content_hash": hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest(),
        "work_item_id": None,
        "created_at": now,
        "status": "completed",
        "progress": {"stage": "verified", "complete": True},
        "error": None,
        "completed_at": now,
    }


def build_payload(root: Path, *, signature_state: str, smoke_status: str) -> dict[str, object]:
    found = candidates(root)
    if not found:
        raise SystemExit(f"No package candidates found under {root}")
    items = [
        qualify_candidate(
            package_type,
            path,
            signature_state=signature_state,
            smoke_status=smoke_status,
        )
        for package_type, path in found
    ]
    return {
        "format_version": 1,
        "platform": items[0]["platform"],
        "architecture": items[0]["architecture"],
        "trusted_normal_installer": all(
            item["signature_state"] in {"signed", "signed_notarized"}
            and item["smoke_status"] == "passed"
            for item in items
        ),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signature-state", default="unsigned")
    parser.add_argument("--smoke-status", choices=("passed", "failed"), required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        args.bundle_root,
        signature_state=args.signature_state,
        smoke_status=args.smoke_status,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
