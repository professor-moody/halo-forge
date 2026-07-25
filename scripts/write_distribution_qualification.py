#!/usr/bin/env python3
"""Write truthful, checksummed desktop distribution qualification evidence."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from halo_forge.product_lab import ProductLabService
from halo_forge.run_db import RunDatabase


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signature-state", default="unsigned")
    parser.add_argument("--smoke-status", choices=("passed", "failed"), required=True)
    args = parser.parse_args()

    found = candidates(args.bundle_root)
    if not found:
        raise SystemExit(f"No package candidates found under {args.bundle_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    database = RunDatabase(str(args.output.parent / ".release-qualification.sqlite"))
    service = ProductLabService(database, root=args.output.parent / ".qualification-root")
    items = []
    for package_type, path in found:
        qualification = service.qualify_release(
            {
                "package_path": str(path),
                "package_type": package_type,
                "signature_state": args.signature_state,
                "smoke_status": args.smoke_status,
                "distribution_status": (
                    "supported"
                    if args.signature_state in {"signed", "signed_notarized"}
                    else "preview"
                ),
            }
        )
        items.append(qualification.to_dict())
    payload = {
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
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
