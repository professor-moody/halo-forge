from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.training_signal import (
    BoundarySignalSession,
    CaptureFidelity,
    TRAINING_SIGNAL_CAPABILITIES,
    TrainingRecordRef,
    TrainingSignalCapabilityDescriptor,
    TrainingSignalSink,
    TrainingSignalSnapshot,
    VerifierObservation,
    complete_signal_boundary,
    default_audit_boundaries,
    load_training_signal_shard,
    record_resolver_from_training_artifact,
    verify_training_signal_shard,
)


def _snapshot(sink: TrainingSignalSink, index: int, *, error: str | None = None):
    return sink.capture(
        record=None,
        source_index=index,
        candidate_ordinal=0,
        prompt=f"prompt {index}",
        output=f"output {index}",
        expected=f"expected {index}",
        training_observation={
            "reward": (index % 101) / 100,
            "success": index % 3 != 0,
            "error": error,
            "details": {
                "component_trace": [
                    {"id": "a", "reward": (index % 5) / 4},
                    {"id": "b", "reward": ((index + 1) % 5) / 4},
                ]
            },
        },
        selected=index % 3 != 0,
        selection_reason="kept" if index % 3 != 0 else "rejected",
    )


def test_capability_registry_is_versioned_and_truthful():
    values = TRAINING_SIGNAL_CAPABILITIES.list()
    assert {value.id for value in values} == {
        "raft:hf",
        "raft:mlx",
        "grpo:hf",
        "grpo:mlx",
        "reasoning:hf",
        "agentic:hf",
        "vlm:hf",
        "audio:hf",
    }
    assert TRAINING_SIGNAL_CAPABILITIES.resolve("grpo", "mlx").boundary_unit == "final"
    assert TRAINING_SIGNAL_CAPABILITIES.resolve("grpo", "mlx").resumable is False
    assert TRAINING_SIGNAL_CAPABILITIES.get("raft:hf").version == "2"
    assert TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf").version == "2"
    assert all(
        value.version == "1"
        for value in values
        if value.id not in {"raft:hf", "grpo:hf"}
    )


def test_training_artifact_v3_exposes_row_aligned_lineage_index(tmp_path):
    from halo_forge.data_lab import DatasetBinding, DatasetLab, TrainingArtifactRenderer
    from halo_forge.data_lab.sources import content_hash, hash_file

    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "".join(
            json.dumps({"prompt": f"p{index}", "response": f"r{index}"}) + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    lab = DatasetLab(tmp_path / "lab")
    try:
        source = lab.add_source(
            {"kind": "local", "path": str(source_path), "canonical_kind": "sft"}
        )
        version = lab.build(
            source.id,
            {"steps": [{"kind": "split", "ratios": {"train": 1}}]},
        )
        artifact = TrainingArtifactRenderer(lab.store).render(
            [DatasetBinding("train", version.version_id, "train")],
            trainer_mode="sft",
            validation_fraction=0.0,
        )
        assert artifact.format_version == 3
        assert set(artifact.lineage_paths) == set(artifact.split_paths) == {"train"}
        lineage = list(artifact.iter_lineage("train"))
        assert len(lineage) == artifact.row_counts["train"] == 6
        assert all(
            {"record_id", "record_hash", "instance_id"}.issubset(value)
            for value in lineage
        )
        resolver = record_resolver_from_training_artifact(artifact)
        assert resolver is not None
        for index in (5, 0, 3, 3):
            resolved_record = resolver(index)
            assert resolved_record.record_id == lineage[index]["record_id"]
            assert resolved_record.record_hash == lineage[index]["record_hash"]
            assert resolved_record.instance_id == lineage[index]["instance_id"]
            assert resolved_record.source_index == index
            assert resolved_record.virtual is lineage[index]["virtual"]
            assert resolved_record.source["record_index"] == lineage[index]["record_index"]
        manifest = json.loads(Path(artifact.path, "manifest.json").read_text())
        assert manifest["format_version"] == 3
        assert manifest["lineage_paths"] == {"train": "lineage/train.jsonl"}

        # A real v2-shaped bundle remains verifiable and falls back to the
        # canonical JSON lineage object without modifying that old artifact.
        legacy_root = tmp_path / "legacy-artifacts"
        legacy_root.mkdir()
        stage = legacy_root / "stage"
        shutil.copytree(artifact.path, stage)
        shutil.rmtree(stage / "lineage")
        legacy_manifest = json.loads((stage / "manifest.json").read_text())
        legacy_manifest["format_version"] = 2
        legacy_manifest.pop("lineage_paths", None)
        legacy_manifest.pop("lineage_index_content", None)
        identity_keys = (
            "format_version",
            "adapter_id",
            "adapter_version",
            "trainer_mode",
            "schema",
            "canonical_schemas",
            "model",
            "tokenizer_revision",
            "chat_template_hash",
            "token_statistics",
            "validation_policy",
            "trainer_content",
            "held_out_content",
            "row_counts",
            "split_paths",
            "asset_roots",
            "lineage_content",
        )
        legacy_identity = {key: legacy_manifest.get(key) for key in identity_keys}
        legacy_identity["bindings"] = legacy_manifest.get("resolved_bindings")
        legacy_hash = content_hash(legacy_identity)
        legacy_id = legacy_hash[:24]
        legacy_manifest["artifact_hash"] = legacy_hash
        legacy_manifest["artifact_id"] = legacy_id
        legacy_manifest["artifact_hashes"] = {
            value.relative_to(stage).as_posix(): hash_file(value)
            for value in stage.rglob("*")
            if value.is_file() and value.name != "manifest.json"
        }
        (stage / "manifest.json").write_text(
            json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n"
        )
        legacy_path = legacy_root / legacy_id
        stage.rename(legacy_path)
        legacy = TrainingArtifactRenderer(lab.store, root=legacy_root).get(legacy_id)
        assert legacy.format_version == 2
        assert legacy.lineage_paths == {}
        assert len(list(legacy.iter_lineage("train"))) == 6
        assert record_resolver_from_training_artifact(legacy) is None
    finally:
        lab.close()


def test_record_identity_and_snapshot_ids_are_stable_and_virtual_is_explicit():
    first = TrainingRecordRef.virtual_identity(
        {"prompt": "p"}, source_index=7, source={"path": "manual.jsonl"}
    )
    second = TrainingRecordRef.virtual_identity(
        {"prompt": "p"}, source_index=7, source={"path": "manual.jsonl"}
    )
    assert first == second
    assert first.virtual is True
    observation = VerifierObservation.from_result(
        {"reward": 0.75, "success": True, "details": {"parsed_value": "ok"}}
    )
    one = TrainingSignalSnapshot.create(
        record=first,
        candidate_ordinal=2,
        prompt="p",
        output="o",
        training_observation=observation,
        run_id="run-1",
        segment_id="segment-1",
        boundary="cycle-1",
    )
    two = TrainingSignalSnapshot.create(
        record=second,
        candidate_ordinal=2,
        prompt="p",
        output="o",
        training_observation=observation,
        run_id="run-1",
        segment_id="segment-1",
        boundary="cycle-1",
    )
    assert one.snapshot_id == two.snapshot_id
    assert observation.parsed_value == "ok"
    with pytest.raises(ValueError, match="finite"):
        VerifierObservation(reward=float("nan"), passed=False)


def test_occurrence_identity_retains_repeated_outputs_and_deduplicates_retries(tmp_path):
    sink = TrainingSignalSink(
        tmp_path,
        run_id="run-occurrence",
        segment_id="segment-1",
        boundary="1",
        capability=TRAINING_SIGNAL_CAPABILITIES.get("raft:hf"),
    )
    common = {
        "record": None,
        "source_index": 4,
        "candidate_ordinal": 0,
        "prompt": "same prompt",
        "output": "same output",
        "training_observation": {"reward": 1.0, "success": True},
    }
    first = sink.capture(**common, occurrence_id="cycle:1:source:4:candidate:0")
    second = sink.capture(**common, occurrence_id="cycle:2:source:4:candidate:0")
    retry = sink.capture(
        **{**common, "output": "retry produced different unsealed tail"},
        occurrence_id="cycle:1:source:4:candidate:0",
    )
    assert first.snapshot_id != second.snapshot_id
    assert retry.snapshot_id == first.snapshot_id
    assert first.identity_mode == second.identity_mode == "trainer_occurrence"
    shard = sink.seal()
    assert shard.observed_count == shard.retained_count == 2
    rows = [
        json.loads(line)
        for line in Path(shard.path, "samples.jsonl").read_text().splitlines()
    ]
    assert {row["occurrence_id"] for row in rows} == {
        "cycle:1:source:4:candidate:0",
        "cycle:2:source:4:candidate:0",
    }


def test_producer_names_are_not_mislabeled_as_content_hashes():
    snapshot = TrainingSignalSnapshot.create(
        record=None,
        source_index=0,
        candidate_ordinal=0,
        prompt="p",
        output="o",
        training_observation={"reward": 1.0, "success": True},
        run_id="run-producer",
        segment_id="segment",
        boundary="final",
        occurrence_id="source:0:candidate:0",
        producer_model_hash="organization/model-name",
    )
    assert len(snapshot.producer_model_hash) == 64
    assert snapshot.producer_model_hash != "organization/model-name"
    assert snapshot.producer_model_identity == {
        "identity_kind": "reference_hash",
        "content_available": False,
        "identity_hash": snapshot.producer_model_hash,
        "reference": "organization/model-name",
    }


def test_signal_sink_exact_sealing_is_idempotent_and_deduplicates(tmp_path):
    capability = TRAINING_SIGNAL_CAPABILITIES.get("raft:hf")
    sink = TrainingSignalSink(
        tmp_path,
        run_id="run-exact",
        segment_id="cycle-0",
        boundary="cycle-0",
        capability=capability,
    )
    snapshots = [_snapshot(sink, index) for index in range(12)]
    assert sink.observe(snapshots[0]) is False
    shard = sink.seal(checkpoint_hash="checkpoint-a")
    assert shard.capture_fidelity == "exact"
    assert shard.observed_count == shard.retained_count == 12
    assert shard.core_count == 12
    assert shard.diagnostic_count == 0
    assert sink.seal(checkpoint_hash="checkpoint-a") == shard
    rows = [
        json.loads(line)
        for line in Path(shard.path, "samples.jsonl").read_text().splitlines()
    ]
    assert len({row["snapshot_id"] for row in rows}) == 12
    assert {row["selection_stratum"] for row in rows} == {"exact"}
    assert {row["checkpoint_hash"] for row in rows} == {"checkpoint-a"}
    assert verify_training_signal_shard(shard.path)["valid"] is True
    assert load_training_signal_shard(shard.path).trace_hash == shard.trace_hash


def test_exhaustive_signal_sealing_and_verification_stream_ids(tmp_path, monkeypatch):
    capability = TRAINING_SIGNAL_CAPABILITIES.get("raft:hf")
    sink = TrainingSignalSink(
        tmp_path,
        run_id="run-exhaustive",
        segment_id="cycle-1",
        boundary="cycle-1",
        capability=capability,
        protocol="exhaustive",
    )
    for index in range(2048):
        _snapshot(sink, index)
    shard = sink.seal(checkpoint_hash="checkpoint-exhaustive")
    manifest = json.loads(Path(shard.path, "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["retained_count"] == 2048
    assert "retained_ids" not in manifest
    assert len(manifest["retained_ids_hash"]) == 64
    with Path(shard.path, "samples.jsonl").open(encoding="utf-8") as handle:
        assert json.loads(next(handle))["checkpoint_hash"] == "checkpoint-exhaustive"

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "samples.jsonl":
            raise AssertionError("verification must stream the JSONL file")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert verify_training_signal_shard(shard.path)["valid"] is True
    assert load_training_signal_shard(shard.path).retained_count == 2048


def test_boundary_completion_hashes_only_for_lifecycle_aware_sinks(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"published checkpoint")
    calls = []

    class LifecycleSink:
        def boundary_complete(self, *, boundary_value, checkpoint_hash):
            calls.append((boundary_value, checkpoint_hash))
            return "sealed"

    assert (
        complete_signal_boundary(
            LifecycleSink(), boundary_value="cycle-3", checkpoint_path=checkpoint
        )
        == "sealed"
    )
    assert calls[0][0] == "cycle-3"
    assert len(calls[0][1]) == 64
    assert (
        complete_signal_boundary(
            None, boundary_value="cycle-3", checkpoint_path=tmp_path / "missing"
        )
        is None
    )
    assert (
        complete_signal_boundary(
            object(), boundary_value="cycle-3", checkpoint_path=tmp_path / "missing"
        )
        is None
    )


def test_boundary_session_uses_managed_lineage_and_normalizes_modality_cycles(tmp_path):
    lineage_path = tmp_path / "train.lineage.jsonl"
    lineage = [
        {
            "record_id": f"record-{index}",
            "record_hash": f"hash-{index}",
            "instance_id": f"instance-{index}",
            "record_index": 100 + index,
            "virtual": False,
        }
        for index in range(3)
    ]
    lineage_path.write_text(
        "".join(json.dumps(value) + "\n" for value in lineage), encoding="utf-8"
    )
    artifact = SimpleNamespace(
        format_version=3,
        artifact_id="artifact-v3",
        artifact_hash="artifact-hash",
        lineage_paths={"train": str(lineage_path)},
        row_counts={"train": 3},
    )
    resolver = record_resolver_from_training_artifact(artifact)
    assert resolver is not None
    capability = TRAINING_SIGNAL_CAPABILITIES.get("audio:hf")
    session = BoundarySignalSession(
        tmp_path / "signals",
        run_id="run-managed",
        trainer="audio",
        capability=capability,
        total_boundaries=2,
        boundaries=["1", "final"],
        protocol="balanced_256",
        reward_threshold=0.5,
        record_resolver=resolver,
    )
    snapshot = session.capture(
        record=None,
        source_index=1,
        source={"trainer": "audio", "cycle": 0},
        candidate_ordinal=0,
        prompt={"task": "asr"},
        output="hello",
        training_observation={"reward": 1.0, "success": True},
    )
    assert snapshot.record.record_id == "record-1"
    assert snapshot.record.instance_id == "instance-1"
    assert snapshot.record.source_index == 1
    assert snapshot.record.virtual is False
    shard = session.boundary_complete(boundary_value=0, checkpoint_hash="checkpoint")
    assert shard is not None and shard.boundary == "1"

    manual = BoundarySignalSession(
        tmp_path / "manual-signals",
        run_id="run-manual",
        trainer="grpo",
        capability=TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf"),
        total_boundaries=1,
        boundaries=["final"],
        protocol="balanced_256",
        reward_threshold=0.5,
    )
    virtual = manual.capture(
        record=None,
        source_index=0,
        source={"path": "manual.jsonl"},
        candidate_ordinal=0,
        prompt="p",
        output="o",
        training_observation={"reward": 1.0, "success": True},
    )
    assert virtual.record.virtual is True
    assert default_audit_boundaries(10) == ["1", "4", "7", "final"]


def test_cli_session_automatically_consumes_managed_record_resolver(tmp_path, monkeypatch):
    from halo_forge.cli import _build_training_signal_session
    from halo_forge.runtime_determinism import RUN_ID_ENV

    lineage_path = tmp_path / "lineage.jsonl"
    lineage_path.write_text(
        json.dumps(
            {
                "record_id": "managed-record",
                "record_hash": "managed-hash",
                "instance_id": "managed-instance",
                "record_index": 9,
                "virtual": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolver = record_resolver_from_training_artifact(
        SimpleNamespace(
            format_version=3,
            artifact_id="artifact",
            artifact_hash="hash",
            lineage_paths={"train": str(lineage_path)},
            row_counts={"train": 1},
        )
    )
    model_path = tmp_path / "producer.bin"
    model_path.write_bytes(b"managed producer bytes")
    args = SimpleNamespace(
        model=str(model_path),
        reward_system_revision="reward-revision",
        reward_audit_boundary=["final"],
        _training_signal_capability=TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf"),
        _training_record_resolver=resolver,
        _managed_dataset_replay={"training_artifact": {"format_version": 3}},
        _reward_integrity_resolved=None,
    )
    monkeypatch.setenv(RUN_ID_ENV, "run-cli-managed")
    monkeypatch.setenv("HALOFORGE_TRAINING_SIGNAL_ROOT", str(tmp_path / "signals"))
    session = _build_training_signal_session(
        args,
        trainer="grpo",
        output_dir=tmp_path / "output",
        total_boundaries=1,
        reward_threshold=0.5,
    )
    snapshot = session.capture(
        record=None,
        source_index=0,
        candidate_ordinal=0,
        prompt="prompt",
        output="output",
        training_observation={"reward": 0.8, "success": True},
    )
    assert snapshot.record.record_id == "managed-record"
    assert snapshot.record.record_hash == "managed-hash"
    assert snapshot.record.instance_id == "managed-instance"
    assert snapshot.record.virtual is False
    assert snapshot.producer_model_hash == hashlib.sha256(
        b"managed producer bytes"
    ).hexdigest()
    assert snapshot.producer_model_identity["content_available"] is True
    assert snapshot.producer_model_identity["identity_source"] == "model"
    assert len(snapshot.runtime_identity["fingerprint"]) == 64
    assert snapshot.runtime_identity["training_signal_capability"] == {
        "id": "grpo:hf",
        "version": "2",
        "backend": "hf",
    }


def test_balanced_protocol_retains_separate_bounded_diagnostics(tmp_path):
    capability = TRAINING_SIGNAL_CAPABILITIES.get("reasoning:hf")
    sink = TrainingSignalSink(
        tmp_path,
        run_id="run-balanced",
        segment_id="cycle-2",
        boundary="cycle-2",
        capability=capability,
        protocol="balanced_256",
    )
    for index in range(1000):
        _snapshot(sink, index, error="parse" if index % 29 == 0 else None)
    shard = sink.seal()
    assert shard.capture_fidelity == "sampled"
    assert shard.observed_count == 1000
    assert shard.retained_count == 256
    assert shard.core_count + shard.diagnostic_count == 256
    assert shard.core_count >= 192
    assert shard.diagnostic_count > 0
    rows = [
        json.loads(line)
        for line in Path(shard.path, "samples.jsonl").read_text().splitlines()
    ]
    assert any(row["selection_stratum"] == "verifier_error" for row in rows)
    assert any(row["selection_stratum"] == "component_disagreement" for row in rows)


def test_aggregate_only_capability_never_claims_sample_evidence(tmp_path):
    capability = TrainingSignalCapabilityDescriptor(
        id="plugin:aggregate",
        version="1",
        trainer="plugin",
        backend="external",
        boundary_unit="final",
        resumable=False,
        available_boundaries=("final",),
        fidelity=CaptureFidelity.AGGREGATE_ONLY,
        reason="plugin exposes only summary rewards",
    )
    sink = TrainingSignalSink(
        tmp_path,
        run_id="run-plugin",
        segment_id="final",
        boundary="final",
        capability=capability,
    )
    for index in range(20):
        _snapshot(sink, index)
    shard = sink.seal()
    assert shard.capture_fidelity == "aggregate_only"
    assert shard.observed_count == 20
    assert shard.retained_count == 0
    assert Path(shard.path, "samples.jsonl").read_text() == ""


def test_mlx_grpo_duplicate_prompts_remain_distinct_groups():
    from halo_forge.grpo.mlx_trainer import _group_prompt_occurrences

    prompts = ["same", "same"]
    samples = [
        ("same", "first-a"),
        ("same", "first-b"),
        ("same", "second-a"),
        ("same", "second-b"),
    ]
    groups = _group_prompt_occurrences(prompts, samples, num_generations=2)
    assert groups == [
        (0, "same", ["first-a", "first-b"]),
        (1, "same", ["second-a", "second-b"]),
    ]
    with pytest.raises(ValueError, match="expected"):
        _group_prompt_occurrences(prompts, samples[:-1], num_generations=2)


def test_hf_grpo_uses_dataset_row_identity_across_shuffled_batches(monkeypatch):
    from halo_forge.grpo.trainer import _build_reward_func

    class Verifier:
        def verify_batch(self, completions, prompts):
            return [SimpleNamespace(reward=1.0, success=True, details={}) for _ in completions]

    class Sink:
        def __init__(self):
            self.source_indexes = []
            self.occurrence_ids = []

        def capture(self, **values):
            self.source_indexes.append(values["source_index"])
            self.occurrence_ids.append(values["occurrence_id"])

    monkeypatch.setattr(
        "halo_forge.rlvr.verifiers.get_verifier", lambda _name: Verifier
    )
    sink = Sink()
    reward = _build_reward_func(
        "test",
        0.0,
        signal_sink=sink,
        num_generations=2,
    )
    values = reward(
        ["p7", "p7", "p3", "p3"],
        ["a", "b", "c", "d"],
        _halo_source_index=[7, 3],
    )
    assert values == [1.0, 1.0, 1.0, 1.0]
    assert sink.source_indexes == [7, 7, 3, 3]
    assert sink.occurrence_ids == [
        "reward-callback:0:source:7:candidate:0",
        "reward-callback:0:source:7:candidate:1",
        "reward-callback:0:source:3:candidate:0",
        "reward-callback:0:source:3:candidate:1",
    ]


def test_audio_candidate_count_and_pairing_are_truthful(tmp_path):
    from halo_forge.audio.data import AudioSample
    from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTTrainer

    trainer = AudioRAFTTrainer(
        AudioRAFTConfig(output_dir=str(tmp_path), samples_per_prompt=3)
    )
    trainer.processor = SimpleNamespace(
        load=lambda _path: SimpleNamespace(waveform=[0.0]),
        load_array=lambda _array, _rate: SimpleNamespace(waveform=[0.0]),
    )
    generated = iter(("one", "two", "three"))
    trainer.adapter = SimpleNamespace(
        transcribe=lambda _waveform: SimpleNamespace(text=next(generated))
    )
    trainer.verifier = SimpleNamespace(
        verify=lambda prediction, expected: SimpleNamespace(
            reward=1.0 if prediction else 0.0,
            success=bool(prediction),
            details={"expected": expected},
        )
    )
    sample = AudioSample("asset.wav", "truth", 1.0)
    predictions = trainer._generate_predictions(
        [sample], show_progress=False, candidates_per_sample=3
    )
    verified = trainer._verify_predictions(predictions, [sample], show_progress=False)
    assert predictions == ["one", "two", "three"]
    assert [value["candidate_ordinal"] for value in verified] == [0, 1, 2]
    with pytest.raises(ValueError, match="exact multiple"):
        trainer._verify_predictions(["one", "two"], [sample, sample, sample], show_progress=False)


def test_metrics_tracker_supports_agentic_sample_logging(tmp_path):
    from halo_forge.utils.metrics import MetricsTracker

    tracker = MetricsTracker(
        str(tmp_path), enable_tensorboard=False, enable_json_logs=True, console_output=False
    )
    tracker.log_samples(3, [0.0, 0.5, 1.0])
    value = json.loads((tmp_path / "sample_metrics.jsonl").read_text())
    assert value == {
        "count": 3,
        "cycle": 3,
        "kind": "sample_rewards",
        "max": 1.0,
        "mean": 0.5,
        "min": 0.0,
    }


def test_reasoning_cycle_save_reads_shared_verifier_details(tmp_path, monkeypatch):
    from halo_forge.reasoning.data import MathSample
    from halo_forge.reasoning.trainer import (
        ReasoningCompletion,
        ReasoningRAFTConfig,
        ReasoningRAFTTrainer,
    )
    from halo_forge.rlvr.verifiers.base import VerifyResult

    monkeypatch.setattr(
        "halo_forge.reasoning.trainer.persist_cycle_artifacts",
        lambda **_kwargs: {},
    )
    trainer = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(output_dir=str(tmp_path), num_cycles=1)
    )
    completion = ReasoningCompletion(
        sample=MathSample(question="2+2", answer="4"),
        completion="The answer is 4",
        reward=1.0,
        verified=True,
        result=VerifyResult(
            success=True,
            reward=1.0,
            details={"extracted_answer": "4"},
        ),
    )
    trainer._save_cycle_results(tmp_path / "cycle_0", [completion], {"cycle": 0})
    saved = json.loads((tmp_path / "cycle_0" / "completions.jsonl").read_text())
    assert saved["extracted_answer"] == "4"


def test_vlm_generation_preserves_structured_verifier_metadata(tmp_path):
    from PIL import Image

    from halo_forge.vlm.data import VLMSample
    from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer

    trainer = VLMRAFTTrainer(VLMRAFTConfig(output_dir=str(tmp_path)))
    trainer.adapter = SimpleNamespace(
        generate=lambda **_kwargs: SimpleNamespace(text="a red square")
    )
    trainer.verifier = SimpleNamespace(
        verify=lambda **_kwargs: SimpleNamespace(
            reward=0.9,
            success=True,
            details="weighted verifier score",
            metadata={
                "details": {"perception": {"score": 1.0}},
                "component_trace": [{"id": "perception", "reward": 1.0}],
            },
        )
    )
    result = trainer.generate_samples(
        [
            VLMSample(
                image=Image.new("RGB", (2, 2), "red"),
                prompt="what is shown?",
                ground_truth="red square",
                metadata={"record_id": "source-row"},
            )
        ],
        1,
    )[0]
    assert result.verifier_metadata["details"]["perception"]["score"] == 1.0
    assert result.verifier_metadata["component_trace"][0]["id"] == "perception"
