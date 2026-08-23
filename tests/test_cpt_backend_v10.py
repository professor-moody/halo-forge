from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from halo_forge.cli import _prepare_cpt_cli_config, _resolve_cpt_cli_config
from halo_forge.cpt import (
    CPTConfig,
    CorpusPackingPlan,
    PACKING_ALGORITHM,
    get_cpt_trainer,
    pack_corpus_records,
    packing_plan_hash,
)
from halo_forge.cpt.mlx_trainer import causal_next_token_loss
from halo_forge.cpt.trainer import (
    resolve_cpt_run_id,
    verify_cpt_packing_plan,
    verify_cpt_training_artifact,
)
from halo_forge.data_lab import (
    TRAINER_DATASET_ADAPTERS,
    DatasetBinding,
    DatasetLab,
    VersionError,
)
from halo_forge.orchestration import resolve_trainer_execution_capability
from halo_forge.public_api.service import PublicApiService
from halo_forge.replay import (
    MANIFEST_VERSION,
    capture_manifest,
    compare_corpus_training_identities,
    load_manifest,
    save_manifest,
)
from halo_forge.runtime_determinism import RUN_ID_ENV
from ui.services.training_service import TrainingService


class WordTokenizer:
    name_or_path = "test/word-tokenizer"
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token = "<pad>"
    pad_token_id = 0
    vocab_size = 16
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}
    init_kwargs = {"normalizer": "identity"}

    _vocab = {
        "<pad>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "j": 10,
        "<eos>": 99,
    }

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[value] for value in str(text).split()]


def _jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _corpus_rows(count: int = 6):
    words = ("a b", "c d e", "f g", "h i", "a c", "d j")
    return [
        {
            "document_id": f"doc-{index}",
            "document_hash": f"hash-{index}",
            "text": f"{words[index % len(words)]}\n\n{words[(index + 1) % len(words)]}",
            "source_ref": f"source-{index}.md",
            "title": f"Document {index}",
        }
        for index in range(count)
    ]


def test_cpt_config_requires_explicit_adaptation_and_budget_semantics():
    with pytest.raises(ValueError, match="explicit"):
        CPTConfig()
    with pytest.raises(ValueError, match="target_tokens"):
        CPTConfig(adaptation="lora", budget_mode="tokens")
    with pytest.raises(ValueError, match="4-bit"):
        CPTConfig(adaptation="full", load_in_4bit=True)

    lora = CPTConfig(
        adaptation="lora",
        budget_mode="tokens",
        target_tokens=4096,
        batch_size=2,
        gradient_accumulation_steps=4,
    )
    full = CPTConfig(adaptation_mode="full", model="local/base", corpus_passes=1.5)
    assert lora.adaptation == "lora"
    assert lora.effective_batch_size == 8
    assert lora.corpus_passes is None
    assert full.adaptation == full.adaptation_mode == "full"
    assert full.model == full.model_name == "local/base"
    assert full.packing == PACKING_ALGORITHM


def test_cpt_yaml_is_the_base_and_omitted_cli_defaults_do_not_overwrite_it(
    tmp_path: Path,
) -> None:
    corpus = _jsonl(tmp_path / "corpus.jsonl", _corpus_rows(2))
    config_path = tmp_path / "cpt.yaml"
    config_path.write_text(
        "\n".join(
            [
                "adaptation: lora",
                "model_name: local/config-model",
                f"train_file: {corpus}",
                "output_dir: configured-output",
                "seed: 17",
                "max_sequence_length: 777",
                f"packing: {PACKING_ALGORITHM}",
                "budget_mode: tokens",
                "target_tokens: 1234",
                "batch_size: 3",
                "gradient_accumulation_steps: 5",
                "learning_rate: 0.00009",
                "warmup_ratio: 0.12",
                "weight_decay: 0.07",
                "max_grad_norm: 0.6",
                "optim: adamw_torch_fused",
                "lora_r: 24",
                "lora_alpha: 48",
                "lora_dropout: 0.15",
                "target_modules: [q_proj, v_proj]",
                "use_dora: true",
                "bf16: false",
                "gradient_checkpointing: false",
                "validation_fraction: 0.2",
                "save_steps: 41",
                "eval_steps: 19",
                "save_total_limit: 2",
                "logging_steps: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=str(config_path),
        adaptation=None,
        train_file=None,
        validation_file=None,
    )

    base, explicit = _prepare_cpt_cli_config(args)
    resolved = _resolve_cpt_cli_config(args, base, explicit)

    assert explicit == set()
    assert resolved.model_name == "local/config-model"
    assert resolved.output_dir == "configured-output"
    assert resolved.seed == 17
    assert resolved.max_sequence_length == 777
    assert resolved.budget_mode == "tokens"
    assert resolved.target_tokens == 1234
    assert resolved.batch_size == 3
    assert resolved.gradient_accumulation_steps == 5
    assert resolved.learning_rate == pytest.approx(0.00009)
    assert resolved.warmup_ratio == pytest.approx(0.12)
    assert resolved.weight_decay == pytest.approx(0.07)
    assert resolved.max_grad_norm == pytest.approx(0.6)
    assert resolved.optim == "adamw_torch_fused"
    assert resolved.lora_r == 24
    assert resolved.lora_alpha == 48
    assert resolved.lora_dropout == pytest.approx(0.15)
    assert resolved.target_modules == ["q_proj", "v_proj"]
    assert resolved.use_dora is True
    assert resolved.bf16 is False
    assert resolved.gradient_checkpointing is False
    assert resolved.validation_fraction == pytest.approx(0.2)
    assert resolved.save_steps == 41
    assert resolved.eval_steps == 19
    assert resolved.save_total_limit == 2
    assert resolved.logging_steps == 7

    override_args = SimpleNamespace(
        config=str(config_path),
        adaptation="full",
        model="local/cli-model",
        train_file=None,
        validation_file=None,
    )
    override_base, override_explicit = _prepare_cpt_cli_config(override_args)
    overridden = _resolve_cpt_cli_config(
        override_args, override_base, override_explicit
    )
    assert override_explicit == {"adaptation", "model"}
    assert overridden.model_name == overridden.model == "local/cli-model"
    assert overridden.adaptation == overridden.adaptation_mode == "full"


def test_public_cpt_effective_batch_is_identical_in_artifact_and_cli() -> None:
    resolved = PublicApiService._normalize_public_training_aliases(
        {
            "mode": "cpt",
            "model": "local/base",
            "dataset": "/tmp/train.jsonl",
            "output_dir": "/tmp/run",
            "adaptation": "lora",
            "effective_batch_size": 1,
        }
    )
    assert resolved["batch_size"] == 1
    assert resolved["gradient_accumulation_steps"] == 1
    assert resolved["effective_batch_size"] == 1
    command = PublicApiService._managed_training_command(resolved)
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--gradient-accumulation") + 1] == "1"

    with pytest.raises(ValueError, match="must equal"):
        PublicApiService._normalize_public_training_aliases(
            {
                "mode": "cpt",
                "effective_batch_size": 4,
                "batch_size": 2,
                "gradient_accumulation_steps": 3,
            }
        )


def test_cpt_preflight_returns_the_published_exact_packing_plan() -> None:
    service = object.__new__(PublicApiService)
    artifact = {
        "artifact_id": "artifact-1",
        "packing_plan": {
            "packing": PACKING_ALGORITHM,
            "train_tokens": 123,
            "estimated_steps": 7,
        },
    }
    service._resolved_cpt_payload = lambda payload: {
        **dict(payload),
        "mode": "cpt",
        "dataset_version_id": "version-1",
    }
    service.preflight_training = lambda _payload: {
        "ok": True,
        "errors": [],
        "warnings": [],
        "training_artifact": artifact,
        "packing_plan": artifact["packing_plan"],
    }
    service.get_dataset_version_readiness = lambda *_args, **_kwargs: {
        "ready": True,
        "blockers": [],
        "warnings": [],
        "actions": [],
    }
    service.corpus_profile = lambda _version_id: {"document_count": 10}

    result = PublicApiService.preflight_cpt(
        service,
        {
            "dataset_version_id": "version-1",
            "model": "local/base",
            "adaptation": "lora",
        },
    )
    assert result["training_artifact"]["artifact_id"] == "artifact-1"
    assert result["packing_plan"]["train_tokens"] == 123


def test_public_artifact_match_accepts_only_renderer_added_sibling_splits() -> None:
    request_with_model_revision = PublicApiService._training_artifact_request(
        {
            "model": "local/base",
            "model_revision": "immutable-model-revision",
        },
        [DatasetBinding("train", "version-1", "train")],
        "cpt",
    )
    assert (
        request_with_model_revision["options"]["tokenizer_revision"]
        == "immutable-model-revision"
    )

    request = {
        "bindings": [
            {
                "role": "train",
                "dataset_version_id": "version-1",
                "split": "train",
            }
        ],
        "options": {
            "trainer_mode": "cpt",
            "model": "local/base",
            "max_sequence_length": 2048,
            "packing": PACKING_ALGORITHM,
            "budget_mode": "passes",
            "corpus_passes": 1.0,
            "effective_batch_size": 1,
        },
    }
    artifact = {
        "trainer_mode": "cpt",
        "model": "local/base",
        "bindings": [
            *request["bindings"],
            {
                "role": "validation",
                "dataset_version_id": "version-1",
                "split": "validation",
            },
            {
                "role": "test",
                "dataset_version_id": "version-1",
                "split": "test",
            },
        ],
        "validation_policy": {"kind": "supplied", "preserved": True},
        "packing_plan": {
            "max_sequence_length": 2048,
            "packing": PACKING_ALGORITHM,
            "budget_mode": "passes",
            "target_tokens": None,
            "corpus_passes": 1.0,
            "effective_batch_size": 1,
        },
    }
    assert PublicApiService._artifact_matches_training_request(
        artifact, request
    )
    artifact["bindings"].append(
        {
            "role": "train",
            "dataset_version_id": "version-2",
            "split": "train",
        }
    )
    assert not PublicApiService._artifact_matches_training_request(
        artifact, request
    )


def test_paragraph_eos_packing_is_deterministic_non_overlapping_and_exact():
    tokenizer = WordTokenizer()
    records = [
        {
            "document_id": "one",
            "document_hash": "h1",
            "source_ref": "one.md",
            "text": "a b\n\nc d e",
        },
        {
            "document_id": "two",
            "document_hash": "h2",
            "source_ref": "two.md",
            "text": "f g",
        },
    ]
    first = pack_corpus_records(records, tokenizer, max_sequence_length=4)
    second = pack_corpus_records(records, tokenizer, max_sequence_length=4)

    assert first == second
    assert [list(value.input_ids) for value in first.sequences] == [
        [1, 2],
        [3, 4, 5, 99],
        [6, 7, 99],
    ]
    assert first.statistics["overlap_tokens"] == 0
    assert first.statistics["dropped_tokens"] == 0
    assert first.statistics["content_tokens"] == 7
    assert first.statistics["eos_tokens"] == 2
    assert first.statistics["packed_tokens"] == 9
    assert first.statistics["paragraph_count"] == 3

    long_paragraph = pack_corpus_records(
        [
            {
                "document_id": "long",
                "document_hash": "long-hash",
                "source_ref": "long.md",
                "text": "a b c d e f",
            }
        ],
        tokenizer,
        max_sequence_length=4,
    )
    assert [list(value.input_ids) for value in long_paragraph.sequences] == [
        [1, 2, 3, 4],
        [5, 6, 99],
    ]


def test_corpus_training_artifact_pins_exact_packing_and_reuses(tmp_path):
    rows = _corpus_rows()
    lab = DatasetLab(tmp_path / "lab")
    try:
        source = lab.add_source(
            {
                "kind": "local",
                "path": str(_jsonl(tmp_path / "corpus.jsonl", rows)),
                "canonical_kind": "corpus",
            },
            dataset_id="corpus",
        )
        version = lab.build(
            source.id,
            {
                "steps": [
                    {
                        "kind": "split",
                        "seed": 7,
                        "ratios": {"train": 0.67, "validation": 0.33},
                    }
                ]
            },
        )
        binding = DatasetBinding("train", version.version_id, "train")
        options = {
            "trainer_mode": "cpt",
            "model": "local/base-model",
            "model_revision": "revision-1",
            "model_hash": "model-content-hash",
            "tokenizer_revision": "tokenizer-revision-1",
            "tokenizer_hash": "tokenizer-content-hash",
            "tokenizer": WordTokenizer(),
            "max_sequence_length": 4,
            "packing": PACKING_ALGORITHM,
            "budget_mode": "passes",
            "corpus_passes": 1.0,
            "effective_batch_size": 2,
        }
        artifact = lab.render_training_artifact([binding], **options)
        reused = lab.render_training_artifact([binding], **options)

        assert artifact.format_version == 4
        assert artifact.schema == "corpus"
        assert artifact.adapter_id == "corpus"
        assert artifact.trainer_mode == "cpt"
        assert artifact.model_revision == "revision-1"
        assert artifact.model_hash == "model-content-hash"
        assert artifact.tokenizer_hash == "tokenizer-content-hash"
        assert artifact.packing_plan
        assert artifact.packing_plan["packing"] == PACKING_ALGORITHM
        assert artifact.packing_plan["artifact_hash"] == artifact.artifact_hash
        assert artifact.packing_plan_hash == packing_plan_hash(artifact.packing_plan)
        assert artifact.token_statistics["exact"] is True
        assert set(artifact.split_fidelity) == {"train", "validation"}
        assert all(
            value["overlap_tokens"] == value["dropped_tokens"] == 0
            for value in artifact.split_fidelity.values()
        )
        assert artifact.validation_policy["kind"] == "supplied"
        assert reused.artifact_hash == artifact.artifact_hash
        assert reused.reused is True
        assert lab.training_artifacts.verify(artifact.artifact_id)["valid"] is True

        applied = PublicApiService._apply_training_artifact_payload(
            {
                "mode": "cpt",
                "model": "local/base-model",
                "adaptation": "lora",
            },
            artifact.to_dict(),
        )
        assert applied["training_artifact_id"] == artifact.artifact_id
        assert applied["training_artifact_hash"] == artifact.artifact_hash
        assert applied["model_revision"] == artifact.model_revision
        assert applied["model_hash"] == artifact.model_hash
        assert applied["tokenizer_revision"] == artifact.tokenizer_revision
        assert applied["tokenizer_hash"] == artifact.tokenizer_hash
        assert applied["expected_packing_plan_hash"] == artifact.packing_plan_hash
        command = PublicApiService._managed_training_command(
            {
                **applied,
                "output_dir": str(tmp_path / "run"),
                "batch_size": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-5,
                "budget_mode": "passes",
                "corpus_passes": 1.0,
            }
        )
        assert command[command.index("--training-artifact-id") + 1] == artifact.artifact_id
        assert (
            command[command.index("--training-artifact-hash") + 1]
            == artifact.artifact_hash
        )
        assert (
            command[command.index("--expected-packing-plan-hash") + 1]
            == artifact.packing_plan_hash
        )

        managed_config = CPTConfig(
            adaptation="lora",
            model_name=str(artifact.model),
            model_revision=artifact.model_revision,
            model_hash=artifact.model_hash,
            tokenizer_revision=artifact.tokenizer_revision,
            tokenizer_hash=artifact.tokenizer_hash,
            train_file=artifact.split_paths["train"],
            validation_file=artifact.split_paths["validation"],
            training_artifact_id=artifact.artifact_id,
            training_artifact_hash=artifact.artifact_hash,
            expected_packing_plan_hash=artifact.packing_plan_hash,
            max_sequence_length=4,
            corpus_passes=1.0,
            batch_size=1,
            gradient_accumulation_steps=2,
        )
        verified_manifest = verify_cpt_training_artifact(
            managed_config,
            train_file=managed_config.train_file,
            validation_file=managed_config.validation_file,
        )
        assert verified_manifest["artifact_hash"] == artifact.artifact_hash
        assert (
            verify_cpt_packing_plan(
                managed_config,
                CorpusPackingPlan(**artifact.packing_plan),
            )
            == artifact.packing_plan_hash
        )
        managed_config.expected_packing_plan_hash = "drifted-packing-plan"
        with pytest.raises(ValueError, match="packing plan drifted"):
            verify_cpt_packing_plan(
                managed_config,
                CorpusPackingPlan(**artifact.packing_plan),
            )

        changed = lab.render_training_artifact(
            [binding],
            **{**options, "corpus_passes": 2.0},
        )
        assert changed.artifact_hash != artifact.artifact_hash
        assert changed.packing_plan["estimated_steps"] > artifact.packing_plan["estimated_steps"]

        with pytest.raises(VersionError, match="share document identity"):
            lab.render_training_artifact(
                [
                    DatasetBinding("train", version.version_id, "train"),
                    DatasetBinding("validation", version.version_id, "train"),
                ],
                **options,
            )

        Path(artifact.split_paths["train"]).write_text(
            json.dumps({"text": "tampered after publication"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(VersionError, match="changed after publication"):
            verify_cpt_training_artifact(
                managed_config,
                train_file=managed_config.train_file,
                validation_file=managed_config.validation_file,
            )
    finally:
        lab.close()


def test_cpt_dispatch_and_orchestration_are_backend_truthful():
    config = CPTConfig(adaptation="lora")
    hf = get_cpt_trainer(config, backend=SimpleNamespace(name="cuda"))
    mlx = get_cpt_trainer(config, backend=SimpleNamespace(name="mlx"))
    assert hf.__class__.__name__ == "CPTTrainer"
    assert mlx.__class__.__name__ == "MLXCPTTrainer"

    hf_capability = resolve_trainer_execution_capability("cpt", "torch_cuda")
    assert hf_capability.segment_unit == "step"
    assert hf_capability.supports_gated_execution is True
    assert hf_capability.resume_parameter == "resume_from_checkpoint"
    assert hf_capability.checkpoint_pattern == "checkpoint-*"

    mlx_capability = resolve_trainer_execution_capability("cpt", "mlx-lm")
    assert mlx_capability.segment_unit == "full_trial"
    assert mlx_capability.supports_gated_execution is False
    assert mlx_capability.checkpoint_pattern is None


def test_cpt_run_id_reuses_managed_identity_and_publishes_direct_identity(monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "managed-cpt-run")
    assert resolve_cpt_run_id("cpt") == "managed-cpt-run"
    assert resolve_cpt_run_id("cpt_mlx") == "managed-cpt-run"

    monkeypatch.delenv(RUN_ID_ENV)
    generated = resolve_cpt_run_id("cpt")
    assert generated.startswith("cpt-")
    assert generated == resolve_cpt_run_id("cpt_mlx")
    assert generated == os.environ[RUN_ID_ENV]


def test_mlx_causal_loss_shifts_inputs_targets_and_masks_padding():
    captured = {}

    class Losses:
        @staticmethod
        def cross_entropy(logits, targets, reduction):
            captured["targets"] = np.asarray(targets)
            captured["reduction"] = reduction
            return np.ones_like(targets, dtype=float)

    class NN:
        losses = Losses()

    class MX:
        @staticmethod
        def mean(value):
            return np.mean(value)

        @staticmethod
        def sum(value):
            return np.sum(value)

        @staticmethod
        def maximum(left, right):
            return np.maximum(left, right)

        @staticmethod
        def array(value):
            return np.asarray(value)

    class Model:
        def __call__(self, value):
            captured["inputs"] = np.asarray(value)
            batch, width = value.shape
            return np.zeros((batch, width, 128), dtype=float)

    input_ids = np.asarray([[1, 2, 3, 0]])
    attention_mask = np.asarray([[1, 1, 1, 0]])
    loss = causal_next_token_loss(
        mx=MX,
        nn=NN,
        model=Model(),
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    assert captured["inputs"].tolist() == [[1, 2, 3]]
    assert captured["targets"].tolist() == [[2, 3, 0]]
    assert captured["reduction"] == "none"
    assert float(loss) == 1.0


def test_replay_v5_captures_corpus_training_identity_and_loads_v4(tmp_path):
    binding = {
        "extraction_identity": {"extractor": "documents@1", "hash": "extract-hash"},
        "corpus_identity": {"dataset_id": "corpus", "content_hash": "corpus-hash"},
        "corpus_version": "version-1",
        "tokenizer_identity": {"id": "test/tokenizer", "revision": "tok-rev"},
        "tokenizer_hash": "tokenizer-hash",
        "packing_plan": {"packing": PACKING_ALGORITHM, "train_tokens": 1024},
        "packing_plan_hash": "packing-hash",
        "budget_mode": "passes",
        "corpus_passes": 1.0,
        "adaptation": "lora",
        "training_artifact": {"artifact_id": "artifact-1", "artifact_hash": "artifact-hash"},
    }
    manifest = capture_manifest(
        run_id="cpt-run",
        modality="cpt",
        model_name="model",
        seed=17,
        config={"adaptation": "lora"},
        corpus_training_binding=binding,
    )
    assert MANIFEST_VERSION == 14
    assert manifest.corpus_training["objective"] == "causal_next_token"
    assert manifest.corpus_training["corpus_version"] == "version-1"
    assert manifest.corpus_training["target_tokens"] is None
    assert manifest.corpus_training["training_artifact"]["artifact_id"] == "artifact-1"
    path = save_manifest(manifest, tmp_path / "run")
    loaded = load_manifest(path)
    assert compare_corpus_training_identities(manifest.corpus_training, loaded.corpus_training) == {
        "matched": True,
        "differences": [],
    }

    legacy_value = manifest.to_dict()
    legacy_value["manifest_version"] = 4
    legacy_value.pop("corpus_training")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_value), encoding="utf-8")
    legacy = load_manifest(legacy_path)
    assert legacy.manifest_version == 4
    assert legacy.corpus_training == {}


def test_managed_worker_replay_preserves_v5_corpus_identity(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    artifact = {
        "artifact_id": "artifact-1",
        "artifact_hash": "artifact-hash",
        "format_version": 4,
        "trainer_mode": "cpt",
        "model": "local/base",
        "tokenizer_revision": "tokenizer-revision",
        "tokenizer_hash": "tokenizer-hash",
        "packing_plan": {
            "packing": PACKING_ALGORITHM,
            "budget_mode": "passes",
            "corpus_passes": 1.0,
            "train_tokens": 1024,
        },
        "packing_plan_hash": "packing-plan-hash",
    }
    TrainingService._persist_replay_manifest(
        output_dir=str(output),
        run_id="managed-cpt-run",
        modality="cpt",
        model="local/base",
        seed=42,
        launch_args={
            "mode": "cpt",
            "model": "local/base",
            "adaptation": "lora",
            "budget_mode": "passes",
            "corpus_passes": 1.0,
            "tokenizer_revision": "tokenizer-revision",
        },
        dataset_value="/managed/artifact/train.jsonl",
        dataset_version_id="version-1",
        dataset_split="train",
        dataset_version_metadata={
            "dataset_id": "corpus",
            "version_id": "version-1",
            "content_hash": "version-content-hash",
            "recipe_hash": "recipe-hash",
            "split": "train",
            "source_fingerprints": {"source": "source-hash"},
            "corpus_extraction": {
                "extraction_id": "extraction-1",
                "content_hash": "extraction-hash",
                "manifest_hash": "manifest-hash",
            },
        },
        dataset_bindings=[
            {
                "role": "train",
                "dataset_version_id": "version-1",
                "split": "train",
            }
        ],
        training_artifact_metadata=artifact,
    )
    manifest = load_manifest(output)
    assert manifest.manifest_version == 14
    assert manifest.corpus_training["corpus_version"] == "version-1"
    assert (
        manifest.corpus_training["extraction_identity"]["extraction_id"]
        == "extraction-1"
    )
    assert (
        manifest.corpus_training["training_artifact"]["artifact_id"]
        == "artifact-1"
    )


def test_default_adapter_registry_exposes_corpus_only_for_cpt():
    adapter = TRAINER_DATASET_ADAPTERS.resolve(schema="corpus", trainer_mode="cpt")
    assert adapter.id == "corpus"
    rendered = adapter.render_record(
        {
            "document_id": "doc",
            "document_hash": "hash",
            "text": "a b",
            "source_ref": "doc.md",
        }
    )
    assert rendered["text"] == "a b"
    with pytest.raises(Exception, match="does not support|No trainer dataset adapter"):
        TRAINER_DATASET_ADAPTERS.resolve(schema="corpus", trainer_mode="sft")
