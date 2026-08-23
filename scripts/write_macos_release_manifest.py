#!/usr/bin/env python3
"""Write checksums and truthful distribution metadata for a macOS DMG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PREVIEW_INSTRUCTIONS = (
    "https://halo-forge.io/docs/getting-started/"
    "install-desktop/#unsigned-macos-developer-preview"
)


def release_channel(version: str) -> str:
    """Return the prerelease channel encoded in a version string."""
    for candidate in ("alpha", "beta", "rc"):
        if candidate in version:
            return candidate
    return "stable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: Path) -> tuple[Path, str]:
    """Write a checksum file compatible with ``shasum --check``."""
    digest = sha256(path)
    checksum_path = Path(f"{path}.sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return checksum_path, digest


def build_manifest(
    *,
    dmg: Path,
    tag: str,
    repository: str,
    signature_state: str,
    digest: str,
) -> dict[str, Any]:
    version = tag.removeprefix("v")
    channel = release_channel(version)
    supported = signature_state == "signed_notarized"
    published_preview = not supported and channel != "stable"
    distribution_status = "supported" if supported else "preview"
    return {
        "format_version": 1,
        "version": version,
        "release_channel": channel,
        "tag": tag,
        "distribution_status": distribution_status,
        "signature_state": signature_state,
        "notarization_status": "validated" if supported else "not_submitted",
        "supported": supported,
        "trusted_normal_installer": supported,
        "preview_install_instructions": None if supported else PREVIEW_INSTRUCTIONS,
        "artifacts": [
            {
                "platform": "macos-aarch64",
                "kind": "dmg",
                "name": dmg.name,
                "size_bytes": dmg.stat().st_size,
                "sha256": digest,
                "signature_state": signature_state,
                "distribution_status": distribution_status,
                "supported": supported,
                "availability": (
                    "github_release"
                    if supported
                    else "github_prerelease"
                    if published_preview
                    else "workflow_artifact"
                ),
                "url": (
                    f"https://github.com/{repository}/releases/download/{tag}/{dmg.name}"
                    if supported or published_preview
                    else None
                ),
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument(
        "--signature-state",
        choices=("unsigned", "signed_notarized"),
        required=True,
    )
    args = parser.parse_args()

    if not args.dmg.is_file():
        raise SystemExit(f"DMG does not exist: {args.dmg}")
    _, digest = write_checksum(args.dmg)
    manifest = build_manifest(
        dmg=args.dmg,
        tag=args.tag,
        repository=args.repository,
        signature_state=args.signature_state,
        digest=digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksum(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
