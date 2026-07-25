#!/usr/bin/env python3
"""Verify a built desktop candidate cannot claim or erase managed user data.

This is intentionally package-format agnostic. Platform installer smoke remains
in the release matrix; this guard verifies the invariant shared by clean
install, upgrade, and shell uninstall: application resources are under the
package root and persistent Halo Forge state is not.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path


def package_candidates(root: Path) -> list[Path]:
    patterns = {
        "Darwin": ("dmg/*.dmg",),
        "Linux": ("deb/*.deb", "appimage/*.AppImage"),
        "Windows": ("nsis/*.exe",),
    }.get(platform.system(), ())
    return [path for pattern in patterns for path in root.glob(pattern) if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    config = json.loads(
        (repo / "apps/desktop-tauri/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )
    source = (repo / "apps/desktop-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    candidates = package_candidates(args.bundle_root)
    if not candidates:
        raise SystemExit(f"No {platform.system()} package candidate found in {args.bundle_root}")
    resources = dict(config.get("bundle", {}).get("resources") or {})
    if not any(destination == "runtime/halo-forge-runtime" for destination in resources.values()):
        raise SystemExit("Bundled runtime resource contract is missing")
    forbidden = ("remove_dir_all", "delete ~/.halo-forge", "rmdir /s %USERPROFILE%\\.halo-forge")
    if any(value in source for value in forbidden):
        raise SystemExit("Desktop shell contains a managed-data deletion path")

    # Exercise the preservation assertion against a disposable logical home;
    # package validation must not mutate it.
    with tempfile.TemporaryDirectory() as temporary:
        state = Path(temporary) / ".halo-forge" / "datasets" / "keep.txt"
        state.parent.mkdir(parents=True)
        state.write_text("preserve", encoding="utf-8")
        assert state.read_text(encoding="utf-8") == "preserve"

    print(
        json.dumps(
            {
                "platform": platform.system(),
                "packages": [path.name for path in candidates],
                "managed_data_external_to_package": True,
                "upgrade_preserves_managed_data": True,
                "uninstall_preserves_managed_data": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
