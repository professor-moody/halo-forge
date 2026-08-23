from __future__ import annotations

import runpy
import zipfile
from pathlib import Path


BUILD_SCRIPT = Path("apps/desktop-tauri/scripts/build_runtime.py")


def test_windows_torch_license_tree_is_archived_without_dropping_notices(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = runpy.run_path(str(BUILD_SCRIPT))
    monkeypatch.setattr(namespace["platform"], "system", lambda: "Windows")

    bundle = tmp_path / "halo-forge-runtime"
    licenses = bundle / "_internal" / "torch-2.13.0.dist-info" / "licenses"
    nested = (
        licenses
        / "third_party"
        / "kineto"
        / "libkineto"
        / "third_party"
        / "prometheus-cpp"
        / "LICENSE.txt"
    )
    nested.parent.mkdir(parents=True)
    nested.write_text("third-party notice\n", encoding="utf-8")
    (licenses / "LICENSE").write_text("torch license\n", encoding="utf-8")

    archives = namespace["archive_windows_torch_licenses"](bundle)

    assert len(archives) == 1
    assert not licenses.exists()
    assert archives[0].name == "licenses.zip"
    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.read("LICENSE").decode() == "torch license\n"
        assert archive.read(nested.relative_to(licenses).as_posix()).decode() == (
            "third-party notice\n"
        )


def test_non_windows_runtime_keeps_torch_license_tree(tmp_path: Path, monkeypatch) -> None:
    namespace = runpy.run_path(str(BUILD_SCRIPT))
    monkeypatch.setattr(namespace["platform"], "system", lambda: "Darwin")
    bundle = tmp_path / "halo-forge-runtime"
    licenses = bundle / "_internal" / "torch-2.13.0.dist-info" / "licenses"
    licenses.mkdir(parents=True)
    (licenses / "LICENSE").write_text("torch license\n", encoding="utf-8")

    assert namespace["archive_windows_torch_licenses"](bundle) == ()
    assert (licenses / "LICENSE").is_file()
