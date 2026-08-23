from __future__ import annotations

from pathlib import Path

from halo_forge.artifact_studio import ArtifactStudioService, SubprocessServingStarter
from halo_forge.run_db import LabV4Catalog, RunDatabase


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self):
        return None


def test_subprocess_serving_starter_uses_verified_occurrence_location_and_identity(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    catalog = LabV4Catalog(database)
    service = ArtifactStudioService(
        database,
        catalog=catalog,
        artifact_root=tmp_path / "artifacts",
    )
    source = tmp_path / "model"
    source.mkdir()
    (source / "weights.bin").write_text("weights", encoding="utf-8")
    artifact = service.import_artifact(
        source,
        artifact_kind="final",
        artifact_format="raw",
        managed=True,
    )
    occurrence = catalog.get_occurrence(artifact["occurrence"]["id"])
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return _FakeProcess()

    starter = SubprocessServingStarter(
        catalog,
        base_path=tmp_path,
        log_dir=tmp_path / "logs",
        python_executable="python-test",
        popen=fake_popen,
        process_identity=lambda pid: 123.5,
    )
    result = starter(
        {
            "id": "profile-one",
            "backend": "local",
            "endpoint_settings": {"host": "127.0.0.1", "port": 8123},
        },
        occurrence,
        {"serving_id": "serve-one"},
    )

    command, options = calls[0]
    assert command[:4] == ["python-test", "-m", "halo_forge.cli", "serve"]
    assert "--backend" not in command
    assert str(Path(artifact["locations"][0]["path"]).resolve()) in command
    assert options["start_new_session"] is True
    assert result["process_id"] == 4242
    assert result["process_started_at"] == 123.5
    assert result["url"] == "http://127.0.0.1:8123/v1"
    assert Path(result["log_path"]).parent == tmp_path / "logs"
    database.close()
