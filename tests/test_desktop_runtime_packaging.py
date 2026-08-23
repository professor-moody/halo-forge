from __future__ import annotations

import runpy
import sys
import zipfile
from pathlib import Path


BUILD_SCRIPT = Path("apps/desktop-tauri/scripts/build_runtime.py")
SMOKE_SCRIPT = Path("apps/desktop-tauri/scripts/packaged_sft_smoke.py")


def test_packaged_smoke_discovers_windows_runtime_executable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    namespace = runpy.run_path(str(SMOKE_SCRIPT))

    assert namespace["DIST_RUNTIME"].name == "halo-forge-runtime.exe"


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


def test_dashboard_runtime_seeds_cpu_torch_before_project_install(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = runpy.run_path(str(BUILD_SCRIPT))
    constraints = tmp_path / "constraints"
    constraints.mkdir()
    (constraints / "release.txt").write_text(
        "pyinstaller==6.19.0\nsetuptools==83.0.0\ntorch==2.13.0\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def capture(cmd: list[str], **_kwargs) -> None:
        calls.append(cmd)

    install = namespace["install_runtime_deps"]
    monkeypatch.setitem(install.__globals__, "run", capture)
    install(
        Path("python"),
        tmp_path,
        profile="linux-dashboard",
    )

    torch_call = next(call for call in calls if "torch" in call)
    project_call = next(call for call in calls if call[-1] == ".")
    assert "https://download.pytorch.org/whl/cpu" in torch_call
    assert calls.index(torch_call) < calls.index(project_call)
