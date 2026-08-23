"""Clean-run contracts for release distribution evidence generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import write_distribution_qualification as qualification


def test_writer_runs_with_only_the_standard_library(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "scripts/write_distribution_qualification.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_writer_hashes_unsigned_macos_preview(tmp_path: Path, monkeypatch) -> None:
    bundle_root = tmp_path / "bundle"
    dmg = bundle_root / "dmg" / "Halo-Forge_preview.dmg"
    dmg.parent.mkdir(parents=True)
    dmg.write_bytes(b"preview-candidate")
    monkeypatch.setattr(qualification.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "arm64")

    payload = qualification.build_payload(
        bundle_root,
        signature_state="unsigned",
        smoke_status="passed",
    )
    item = payload["items"][0]

    assert payload["platform"] == "macos"
    assert payload["architecture"] == "arm64"
    assert payload["trusted_normal_installer"] is False
    assert item["package_type"] == "dmg"
    assert item["signature_state"] == "unsigned"
    assert item["smoke_status"] == "passed"
    assert item["evidence"]["package_name"] == dmg.name
    assert item["evidence"]["size_bytes"] == len(b"preview-candidate")
    assert len(item["evidence"]["package_sha256"]) == 64
    assert len(item["content_hash"]) == 64
    json.dumps(payload, allow_nan=False)
