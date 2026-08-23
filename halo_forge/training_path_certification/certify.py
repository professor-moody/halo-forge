"""Container-side real certification steps.

This module is intentionally argv-driven and emits one JSON object. It is not a
simulation: the SFT path uses Dataset Lab, the v3 renderer, and the shipped SFT
CLI. Other path profiles fail closed until they gain an equivalent executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .registry import FIXTURES


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path, patterns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    values: list[Path] = []
    for pattern in patterns:
        values.extend(root.glob(pattern))
    for path in sorted({value.resolve() for value in values if value.is_file()}):
        relative = str(path.relative_to(root.resolve())).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def _context(request: Path) -> dict[str, Any]:
    value = json.loads(request.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("certification request must be an object")
    return value


def _prior(context: Mapping[str, Any], step: str) -> dict[str, Any]:
    value = dict(context.get("evidence") or {}).get(step) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid prior evidence for {step}")
    return dict(value)


def fixture_dataset(context: Mapping[str, Any]) -> dict[str, Any]:
    from halo_forge.data_lab import DatasetLab

    revision = dict(context["path_revision"])
    if revision["trainer_mode"] != "sft" or revision["fixture_id"] not in FIXTURES:
        raise RuntimeError(
            "This path has no real certification fixture executor yet; it remains Needs verification"
        )
    attempt = Path(str(context["attempt_dir"]))
    lab_root = attempt / "datasets"
    source_path = attempt / "fixture.jsonl"
    rows = list(FIXTURES[str(revision["fixture_id"])])
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    lab = DatasetLab(lab_root)
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(source_path),
            "canonical_kind": "sft",
            "modality": "text",
            "field_mapping": {"prompt": "prompt", "response": "response"},
        },
        dataset_id="v21-certification-sft",
    )
    version = lab.build(
        source.id,
        {
            "schema": "sft",
            "seed": 42,
            "steps": [
                {"kind": "normalize", "fields": ["prompt", "response"]},
                {"kind": "validate", "schema": "sft", "on_error": "quarantine"},
                {"kind": "dedup", "method": "exact"},
                {"kind": "split", "ratios": {"train": 0.75, "validation": 0.25}, "seed": 42},
            ],
        },
    )
    return {
        "passed": True,
        "dataset_root": str(lab_root),
        "dataset_id": version.dataset_id,
        "dataset_version_id": version.version_id,
        "dataset_content_hash": version.content_hash,
        "source_fingerprint": source.fingerprint,
        "split_counts": dict(version.split_counts),
    }


def trainer_artifact(context: Mapping[str, Any]) -> dict[str, Any]:
    from halo_forge.data_lab import DatasetBinding, DatasetLab

    dataset = _prior(context, "fixture_dataset")
    revision = dict(context["path_revision"])
    lab = DatasetLab(dataset["dataset_root"])
    artifact = lab.render_training_artifact(
        [
            DatasetBinding("train", dataset["dataset_version_id"], "train", dataset["dataset_id"]),
            DatasetBinding("validation", dataset["dataset_version_id"], "validation", dataset["dataset_id"]),
        ],
        trainer_mode=revision["trainer_mode"],
        adapter_id="sft",
        seed=42,
        validation_fraction=0.0,
    )
    return {
        "passed": True,
        "artifact_id": artifact.artifact_id,
        "artifact_hash": artifact.artifact_hash,
        "artifact_path": artifact.path,
        "format_version": artifact.format_version,
        "split_paths": dict(artifact.split_paths),
        "lineage_paths": dict(artifact.lineage_paths),
        "row_counts": dict(artifact.row_counts),
    }


def model_preparation(context: Mapping[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    revision = dict(context["path_revision"])
    attempt = Path(str(context["attempt_dir"]))
    cache = attempt / "model-cache"
    snapshot = Path(
        snapshot_download(
            repo_id=str(revision["model_id"]),
            revision=None,
            cache_dir=str(cache),
            local_files_only=False,
        )
    ).resolve()
    resolved_commit = snapshot.name
    contract_hash = _hash_tree(
        snapshot,
        (
            "config.json",
            "generation_config.json",
            "tokenizer*.json",
            "tokenizer*.model",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "processor_config.json",
        ),
    )
    if not (snapshot / "config.json").is_file() or not contract_hash:
        raise RuntimeError("prepared model is missing configuration or tokenizer/processor identity")
    return {
        "passed": True,
        "model_path": str(snapshot),
        "model_id": revision["model_id"],
        "resolved_model_commit": resolved_commit,
        "tokenizer_processor_hash": contract_hash,
    }


def _run_sft(context: Mapping[str, Any], *, output: Path, max_samples: int) -> dict[str, Any]:
    artifact = _prior(context, "trainer_artifact")
    model = _prior(context, "model_preparation")
    command = [
        sys.executable,
        "-m",
        "halo_forge.cli",
        "sft",
        "train",
        "--model",
        str(model["model_path"]),
        "--data",
        str(artifact["split_paths"]["train"]),
        "--validation-data",
        str(artifact["split_paths"]["validation"]),
        "--output",
        str(output),
        "--max-steps",
        "1",
        "--max-samples",
        str(max_samples),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "1",
        "--max-seq-length",
        "256",
        "--lora-rank",
        "4",
        "--lora-alpha",
        "8",
        "--seed",
        "42",
        "--capture-parameter-hashes",
        "--save-steps",
        "1",
        "--eval-steps",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
    (output.parent / f"{output.name}.trainer.log").write_text(
        (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"real SFT trainer exited with status {completed.returncode}")
    summary_path = output / "training_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("real SFT trainer did not publish training_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"summary": summary, "summary_path": str(summary_path), "command_argv": command}


def capacity_step(context: Mapping[str, Any]) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"]))
    scratch = attempt / "capacity-scratch"
    result = _run_sft(context, output=scratch, max_samples=2)
    summary = result["summary"]
    total_train_steps = int(
        summary.get("total_train_steps_executed")
        or summary.get("total_train_steps")
        or 0
    )
    executed = bool(summary.get("weights_updated")) and total_train_steps >= 1
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "passed": executed and not scratch.exists(),
        "optimizer_step_executed": executed,
        "scratch_cleaned": not scratch.exists(),
        "measurements": {
            "train_steps": total_train_steps,
            "final_train_loss": summary.get("final_train_loss"),
        },
    }


def optimizer_update(context: Mapping[str, Any]) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"]))
    output = attempt / "real-trainer-output"
    result = _run_sft(context, output=output, max_samples=8)
    summary = result["summary"]
    parameter = dict(summary.get("parameter_evidence") or {})
    total_train_steps = int(
        summary.get("total_train_steps_executed")
        or summary.get("total_train_steps")
        or 0
    )
    return {
        "passed": bool(summary.get("weights_updated")) and bool(parameter.get("changed")),
        "real_trainer_entrypoint": True,
        "trainer_mode": "sft",
        "output_dir": str(output),
        "summary_path": result["summary_path"],
        "weights_updated": bool(summary.get("weights_updated")),
        "total_train_steps": total_train_steps,
        "parameter_evidence": parameter,
    }


def parameter_delta(context: Mapping[str, Any]) -> dict[str, Any]:
    result = _prior(context, "optimizer_update")
    parameter = dict(result.get("parameter_evidence") or {})
    return {
        "passed": bool(parameter.get("changed")),
        "algorithm": parameter.get("algorithm"),
        "before": parameter.get("before"),
        "after": parameter.get("after"),
        "changed": bool(parameter.get("changed")),
    }


def artifact_files(context: Mapping[str, Any]) -> dict[str, Any]:
    result = _prior(context, "optimizer_update")
    revision = dict(context["path_revision"])
    output = Path(str(result["output_dir"]))
    expected = [str(value) for value in revision.get("expected_artifacts") or []]
    missing = [relative for relative in expected if not (output / relative).is_file()]
    hashes = {relative: _hash_file(output / relative) for relative in expected if (output / relative).is_file()}
    return {"passed": not missing, "verified": not missing, "missing": missing, "file_hashes": hashes}


def artifact_reload(context: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = _prior(context, "model_preparation")
    trained = _prior(context, "optimizer_update")
    base_path = str(model["model_path"])
    adapter_path = str(Path(str(trained["output_dir"])) / "final_model")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="auto")
    restored = PeftModel.from_pretrained(base, adapter_path)
    encoded = tokenizer("Certification input", return_tensors="pt")
    encoded = {key: value.to(restored.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = restored(**encoded).logits
    finite = bool(torch.isfinite(output).all().item())
    del restored, base
    return {"passed": finite, "reloaded": True, "finite_output": finite, "fixed_input_hash": hashlib.sha256(b"Certification input").hexdigest()}


def replay_lineage(context: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _prior(context, "trainer_artifact")
    dataset = _prior(context, "fixture_dataset")
    model = _prior(context, "model_preparation")
    trained = _prior(context, "optimizer_update")
    replay_path = Path(str(trained["output_dir"])) / "replay.json"
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    else:
        replay = {}
    replay["manifest_version"] = 14
    replay["training_path"] = {
        "training_path_revision_id": context["path_revision"]["id"],
        "fixture_id": context["path_revision"]["fixture_id"],
        "fixture_hash": context["path_revision"]["fixture_hash"],
        "exact_model_commit": model["resolved_model_commit"],
        "trainer_adapter_version": context["path_revision"]["trainer_adapter_version"],
        "capacity_adapter_version": context["path_revision"]["capacity_adapter_version"],
        "runtime_qualification_id": context["runtime_qualification_id"],
        "dataset_version_id": dataset["dataset_version_id"],
        "dataset_content_hash": dataset["dataset_content_hash"],
        "training_artifact_id": artifact["artifact_id"],
        "training_artifact_hash": artifact["artifact_hash"],
    }
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True), encoding="utf-8")
    lineage_complete = all(
        artifact.get(key)
        for key in ("artifact_id", "artifact_hash", "lineage_paths")
    ) and all(dataset.get(key) for key in ("dataset_version_id", "dataset_content_hash"))
    return {"passed": bool(lineage_complete), "replay_version": 14, "replay_path": str(replay_path), "replay_hash": _hash_file(replay_path), "lineage_complete": bool(lineage_complete)}


def scratch_cleanup(context: Mapping[str, Any]) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"]))
    leftovers = [path for path in (attempt / "capacity-scratch", attempt / ".tmp") if path.exists()]
    for path in leftovers:
        shutil.rmtree(path, ignore_errors=True)
    remaining = [str(path) for path in leftovers if path.exists()]
    return {"passed": not remaining, "scratch_cleaned": not remaining, "remaining": remaining}


STEPS = {
    "fixture_dataset": fixture_dataset,
    "trainer_artifact": trainer_artifact,
    "model_preparation": model_preparation,
    "capacity_step": capacity_step,
    "optimizer_update": optimizer_update,
    "parameter_delta": parameter_delta,
    "artifact_files": artifact_files,
    "artifact_reload": artifact_reload,
    "replay_lineage": replay_lineage,
    "scratch_cleanup": scratch_cleanup,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=tuple(STEPS), required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    result = STEPS[args.step](_context(args.request))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
