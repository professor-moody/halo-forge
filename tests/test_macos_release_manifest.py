"""macOS signed-release and no-cost developer-preview contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.write_macos_release_manifest import build_manifest, main, sha256


@pytest.mark.parametrize("channel", ["alpha", "beta", "rc"])
def test_unsigned_prerelease_is_a_published_but_unsupported_preview(
    tmp_path: Path, channel: str
):
    dmg = tmp_path / "Halo-Forge_2.0.0_aarch64-unsigned-preview.dmg"
    dmg.write_bytes(b"preview")

    manifest = build_manifest(
        dmg=dmg,
        tag=f"v2.0.0-{channel}-1",
        repository="professor-moody/halo-forge",
        signature_state="unsigned",
        digest=sha256(dmg),
    )

    artifact = manifest["artifacts"][0]
    assert manifest["release_channel"] == channel
    assert manifest["signature_state"] == "unsigned"
    assert manifest["notarization_status"] == "not_submitted"
    assert manifest["distribution_status"] == "preview"
    assert manifest["supported"] is False
    assert manifest["trusted_normal_installer"] is False
    assert manifest["preview_install_instructions"].startswith("https://halo-forge.io/")
    assert artifact["availability"] == "github_prerelease"
    assert artifact["supported"] is False
    assert artifact["url"].endswith(dmg.name)


def test_unsigned_stable_build_is_withheld_as_a_workflow_artifact(tmp_path: Path):
    dmg = tmp_path / "Halo-Forge_2.0.0_aarch64-unsigned-preview.dmg"
    dmg.write_bytes(b"preview")

    manifest = build_manifest(
        dmg=dmg,
        tag="v2.0.0",
        repository="professor-moody/halo-forge",
        signature_state="unsigned",
        digest=sha256(dmg),
    )

    artifact = manifest["artifacts"][0]
    assert manifest["release_channel"] == "stable"
    assert artifact["availability"] == "workflow_artifact"
    assert artifact["url"] is None


def test_signed_notarized_build_is_a_supported_release(tmp_path: Path):
    dmg = tmp_path / "Halo-Forge_2.0.0_aarch64.dmg"
    dmg.write_bytes(b"signed")

    manifest = build_manifest(
        dmg=dmg,
        tag="v2.0.0",
        repository="professor-moody/halo-forge",
        signature_state="signed_notarized",
        digest=sha256(dmg),
    )

    artifact = manifest["artifacts"][0]
    assert manifest["notarization_status"] == "validated"
    assert manifest["distribution_status"] == "supported"
    assert manifest["supported"] is True
    assert manifest["trusted_normal_installer"] is True
    assert manifest["preview_install_instructions"] is None
    assert artifact["availability"] == "github_release"
    assert artifact["supported"] is True


def test_cli_writes_verifiable_checksums_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dmg = tmp_path / "Halo-Forge_2.0.0-alpha-2_aarch64-unsigned-preview.dmg"
    dmg.write_bytes(b"preview bytes")
    output = tmp_path / "halo-forge-release-manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_macos_release_manifest.py",
            "--dmg",
            str(dmg),
            "--output",
            str(output),
            "--tag",
            "v2.0.0-alpha-2",
            "--repository",
            "professor-moody/halo-forge",
            "--signature-state",
            "unsigned",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    dmg_checksum = Path(f"{dmg}.sha256").read_text(encoding="utf-8")
    manifest_checksum = Path(f"{output}.sha256").read_text(encoding="utf-8")
    assert dmg_checksum == f"{sha256(dmg)}  {dmg.name}\n"
    assert manifest_checksum == f"{sha256(output)}  {output.name}\n"
    assert payload["artifacts"][0]["sha256"] == sha256(dmg)
