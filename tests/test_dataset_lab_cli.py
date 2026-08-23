"""CLI parity checks for the local Dataset Lab facade."""

from __future__ import annotations

import json
import importlib.util
import sys
import time
from pathlib import Path


def _run_cli(monkeypatch, capsys, *args: str):
    import halo_forge.cli as cli

    monkeypatch.setattr(sys, "argv", ["halo-forge", *args])
    cli.main()
    return json.loads(capsys.readouterr().out)


def _assert_scenario_availability_contract(scenarios) -> None:
    """Assert the runtime-availability contract for scenarios listed by the CLI.

    `available` must be a bool, and `unavailable_reason` must be populated if
    and only if `available` is False. Which scenarios are available depends on
    the host runtime and is deliberately not asserted here.
    """
    for scenario in scenarios:
        available = scenario["available"]
        assert isinstance(available, bool), scenario
        reason = scenario.get("unavailable_reason")
        if available:
            assert not reason, scenario
        else:
            assert isinstance(reason, str) and reason.strip(), scenario


def test_dataset_lab_cli_add_build_preview_and_versions(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    source = tmp_path / "examples.jsonl"
    source.write_text(
        "".join(
            json.dumps({"question": f"q{i}", "answer": f"a{i}"}) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "sft",
                "seed": 42,
                "steps": [
                    {
                        "kind": "map",
                        "schema": "sft",
                        "fields": {"prompt": "question", "response": "answer"},
                    },
                    {
                        "kind": "split",
                        "method": "random",
                        "ratios": {"train": 0.5, "test": 0.5},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "lab"

    added = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "add",
        "--root",
        str(root),
        "--name",
        "CLI fixture",
        "--path",
        str(source),
        "--kind",
        "sft",
    )
    dataset_id = added["id"]
    assert added["sources"][0]["row_count"] == 4

    preview = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "preview",
        "--root",
        str(root),
        dataset_id,
        "--limit",
        "2",
    )
    assert preview["total"] == 4
    assert len(preview["items"]) == 2

    version = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "build",
        "--root",
        str(root),
        dataset_id,
        "--recipe",
        str(recipe),
    )
    assert version["split_counts"] == {"test": 2, "train": 2}

    versions = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "versions",
        "--root",
        str(root),
        dataset_id,
    )
    assert versions["items"][0]["id"] == version["id"]

    artifact = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "render",
        "--root",
        str(root),
        version["id"],
        "--trainer",
        "sft",
    )
    assert artifact["adapter_id"] == "sft"
    assert Path(artifact["split_paths"]["train"]).is_file()

    comparison = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "compare",
        "--root",
        str(root),
        version["id"],
        version["id"],
    )
    assert comparison["added"] == []
    assert comparison["removed"] == []

    monkeypatch.setenv("HALOFORGE_EVALUATION_ROOT", str(tmp_path / "evaluations"))
    suite = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "suite",
        "create",
        "--name",
        "CLI feedback",
        "--items",
        json.dumps(
            [
                {
                    "id": "failure",
                    "record_id": "cli-failure",
                    "input": "fix",
                    "expected": "fixed",
                    "score_by_subject": {"base": 1.0, "candidate": 0.0},
                }
            ]
        ),
    )
    revision_id = suite["revision"]["id"]
    detached = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "run",
        "--suite-revision",
        revision_id,
        "--subject",
        "detached-fixture",
        "--subject-revision",
        "pinned",
        "--request",
        json.dumps({"scores": {"failure": 1.0}}),
    )["evaluation"]
    assert detached["status"] == "queued"
    assert detached["worker_pid"] > 0
    from halo_forge.run_db import get_database

    deadline = time.time() + 10
    detached_record = get_database().get_evaluation(detached["id"])
    while detached_record.status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.05)
        detached_record = get_database().get_evaluation(detached["id"])
    assert detached_record.status == "completed", detached_record.error

    base = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "run",
        "--suite-revision",
        revision_id,
        "--subject",
        "base",
        "--wait",
    )["evaluation"]
    candidate = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "run",
        "--suite-revision",
        revision_id,
        "--subject",
        "candidate",
        "--wait",
    )["evaluation"]
    preview = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "mine",
        "--root",
        str(root),
        "--base",
        base["id"],
        "--candidate",
        candidate["id"],
        "--selector",
        "regression",
    )
    assert preview["total"] == 1
    mined = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "mine",
        "--root",
        str(root),
        "--base",
        base["id"],
        "--candidate",
        candidate["id"],
        "--selector",
        "regression",
        "--dataset",
        dataset_id,
        "--parent-version",
        version["id"],
    )
    assert mined["status"] == "completed"
    assert mined["version_id"] != version["id"]


def test_dataset_lab_cli_requires_exactly_one_source_kind(tmp_path: Path, monkeypatch, capsys):
    import pytest

    import halo_forge.cli as cli

    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))

    monkeypatch.setattr(
        sys,
        "argv",
        ["halo-forge", "data", "add", "--root", str(tmp_path), "--name", "missing"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "exactly one" in capsys.readouterr().out.lower()


def test_dataset_lab_cli_scenarios_share_registry_and_write_template(
    tmp_path: Path, monkeypatch, capsys
):
    listed = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "list",
        "--include-unavailable",
        "--json",
    )
    scenario_ids = {item["id"] for item in listed["items"]}
    assert "instruction-sft" in scenario_ids
    assert "audio-asr" in scenario_ids
    assert "audio-classification" in scenario_ids
    assert listed["registry_revision"]

    all_scenarios = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "list",
        "--include-unavailable",
        "--json",
    )
    # `available` is a probe of what this host has installed, not a static
    # contract; a runtime without torch answers False for every trainer-backed
    # scenario. Assert the contract here and cover both branches
    # deterministically in
    # `test_dataset_lab_cli_scenario_availability_contract_holds_in_both_branches`.
    _assert_scenario_availability_contract(all_scenarios["items"])
    classification = next(
        item for item in all_scenarios["items"] if item["id"] == "audio-classification"
    )
    assert classification["declared_trainer_modes"] == ["classify"]
    # audio-tts is unavailable in the registry itself on every host.
    unavailable = next(
        item for item in all_scenarios["items"] if item["id"] == "audio-tts"
    )
    assert unavailable["available"] is False
    assert "trainer contract" in unavailable["unavailable_reason"]

    output = tmp_path / "fixture.jsonl"
    rendered = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "template",
        "instruction-sft",
        "--output",
        str(output),
        "--json",
    )
    assert rendered["scenario_revision_id"] == "instruction-sft@1"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["instruction"]
    assert records[0]["answer"]

    media_root = tmp_path / "vlm-fixture"
    media_rendered = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "template",
        "vlm-captioning",
        "--output",
        str(media_root),
        "--json",
    )
    media_record = json.loads(
        Path(media_rendered["path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert (media_root / media_record["image"]).is_file()
    assert len(media_rendered["files"]) == 2


def test_dataset_lab_cli_scenario_availability_contract_holds_in_both_branches(
    monkeypatch, capsys
):
    """Both availability branches must honour the reason contract.

    The availability probe (`_module_available`) plus the active backend are
    the host-dependent seams; pinning both makes the available and unavailable
    branches deterministic on any runner, including CI jobs without torch.
    """
    import halo_forge.cli as cli
    from halo_forge.own_data import service as own_data_service

    monkeypatch.setattr(cli, "_guided_active_backend_name", lambda: "cpu")

    def _scenarios():
        listed = _run_cli(
            monkeypatch,
            capsys,
            "data",
            "scenarios",
            "list",
            "--include-unavailable",
            "--json",
        )
        return {item["id"]: item for item in listed["items"]}

    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: True)
    ready = _scenarios()
    _assert_scenario_availability_contract(ready.values())
    assert ready["audio-classification"]["available"] is True
    assert not ready["audio-classification"]["unavailable_reason"]
    assert ready["audio-classification"]["trainer_modes"] == ["classify"]

    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: False)
    thin = _scenarios()
    _assert_scenario_availability_contract(thin.values())
    assert thin["audio-classification"]["available"] is False
    assert thin["audio-classification"]["unavailable_reason"]
    assert thin["audio-classification"]["trainer_modes"] == []


def test_dataset_lab_cli_scenarios_and_add_use_active_runtime(
    tmp_path: Path, monkeypatch, capsys
):
    import pytest

    import halo_forge.cli as cli

    original_find_spec = importlib.util.find_spec

    def installed(name: str, *args, **kwargs):
        if name in {"mlx", "mlx_lm"}:
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(cli, "_guided_active_backend_name", lambda: "mlx")
    monkeypatch.setattr("halo_forge.own_data.service.importlib.util.find_spec", installed)

    listed = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "list",
        "--include-unavailable",
        "--json",
    )
    preference = next(
        item for item in listed["items"] if item["id"] == "preference-pairs"
    )
    assert listed["active_backend"] == "mlx"
    assert preference["trainer_modes"] == ["dpo"]
    compatibility = {
        item["trainer_mode"]: item for item in preference["compatible_trainers"]
    }
    assert compatibility["dpo"]["compatible"] is True
    assert compatibility["orpo"]["compatible"] is False

    shown = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "scenarios",
        "show",
        "preference-pairs",
        "--json",
    )
    assert shown["active_backend"] == "mlx"
    assert shown["trainer_modes"] == ["dpo"]
    assert shown["examples"][0]["records"]

    source = tmp_path / "audio.jsonl"
    source.write_text('{"audio":"clip.wav","transcript":"hello"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "data",
            "add",
            "--name",
            "Unsupported MLX audio",
            "--path",
            str(source),
            "--scenario",
            "audio-asr",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "not implemented on MLX" in capsys.readouterr().out


def test_dataset_lab_cli_add_accepts_scenario_and_structured_map_flags(
    tmp_path: Path, monkeypatch, capsys
):
    import halo_forge.cli as cli
    from halo_forge.own_data import service as own_data_service

    # `data add --scenario` exits 2 when the active runtime has no verified
    # trainer, so pin the host-dependent seams instead of depending on what
    # the runner happens to have installed.
    monkeypatch.setattr(cli, "_guided_active_backend_name", lambda: "cpu")
    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: True)

    source = tmp_path / "questions.csv"
    source.write_text("question,answer\nWhat is 2+2?,4\n", encoding="utf-8")
    database = tmp_path / "guided.db"
    root = tmp_path / "datasets"

    added = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "add",
        "--root",
        str(root),
        "--database",
        str(database),
        "--name",
        "Guided questions",
        "--path",
        str(source),
        "--scenario",
        "instruction-sft",
        "--map",
        "prompt=question",
        "--map",
        "response=answer",
        "--accept-recommended",
        "--json",
    )
    assert added["canonical_schema"] == "sft"
    assert added["modality"] == "text"
    guided = added["sources"][0]["metadata"]["guided_own_data"]
    assert guided["scenario_revision_id"] == "instruction-sft@1"
    assert guided["preparation_plan"]["mapping_plan"]["confirmed"] is True

    version = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "build",
        "--root",
        str(root),
        "--database",
        str(database),
        added["id"],
        "--recommended-recipe",
        "--json",
    )
    assert version["row_count"] == 1

    listed = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "list",
        "--root",
        str(root),
        "--database",
        str(database),
        "--json",
    )
    assert [item["id"] for item in listed["items"]] == [added["id"]]


def test_dataset_lab_cli_inspect_and_managed_import_are_persistent(
    tmp_path: Path, monkeypatch, capsys
):
    from halo_forge.run_db import RunDatabase
    from halo_forge.workstation_jobs import WorkstationScheduler

    database = tmp_path / "guided.db"
    source = tmp_path / "questions.csv"
    source.write_text(
        "question,answer\nWhat is 2+2?,4\nWhat is 3+3?,6\n", encoding="utf-8"
    )
    root = tmp_path / "lab"
    catalog = RunDatabase(database)
    try:
        unrelated = WorkstationScheduler(catalog).enqueue(
            kind="unrelated_test_work",
            launch_spec={"handler": "test.must_not_run"},
            resource_class="cpu",
            domain_kind="test",
            domain_id="unrelated",
        )
    finally:
        catalog.close()

    inspected = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "inspect",
        "--root",
        str(root),
        "--database",
        str(database),
        "--path",
        str(source),
        "--json",
    )
    assert inspected["inspection"]["status"] == "completed"
    assert inspected["inspection"]["row_count"] == 2
    assert inspected["inspection"]["schema_candidates"][0]["scenario_id"] == "instruction-sft"
    assert inspected["work_item_id"]
    durable = RunDatabase(database)
    try:
        work = durable.get_work_item(inspected["work_item_id"])
        assert work is not None
        assert work.status == "completed"
        assert work.domain_kind == "dataset_inspection"
        assert work.domain_id == inspected["inspection"]["id"]
        assert durable.get_work_item(unrelated.id).status == "queued"
    finally:
        durable.close()

    managed = _run_cli(
        monkeypatch,
        capsys,
        "data",
        "import",
        "--root",
        str(root),
        "--database",
        str(database),
        "--path",
        str(source),
        "--managed",
        "--capacity-override-reason",
        "Temporary test fixture on a deliberately constrained CI volume",
        "--json",
    )
    assert managed["import"]["source_kind"] == "upload"
    assert managed["import"]["status"] == "completed"
    assert managed["import"]["files"][0]["status"] == "verified"
    assert (
        managed["inspection"]["source_fingerprint"]
        == inspected["inspection"]["source_fingerprint"]
    )
    assert managed["reused"] is True
