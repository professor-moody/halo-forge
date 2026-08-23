"""Verified PyTorch trainers for specialized task artifacts.

The implementation intentionally favors a small explicit optimizer loop over a
second orchestration stack. It consumes Dataset Lab canonical JSONL and writes a
normal Hugging Face artifact plus Halo Forge's training summary contract.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class SpecializedTrainConfig:
    task: str
    model: str
    dataset: str
    output_dir: str
    validation_file: Optional[str] = None
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_samples: Optional[int] = None
    seed: int = 42
    multi_label: bool = False
    max_length: int = 512
    proof_run: bool = False
    label_schema_revision_id: Optional[str] = None
    retrieval_corpus_id: Optional[str] = None


def _records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"specialized training dataset does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        values = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else payload.get("records", [])
    elif suffix in {".csv", ".tsv"}:
        with source.open("r", encoding="utf-8", newline="") as stream:
            values = list(
                csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
            )
    else:
        raise ValueError("specialized trainers accept JSONL, JSON, CSV, or TSV")
    rows = [dict(value) for value in values if isinstance(value, Mapping)]
    return rows[:limit] if limit else rows


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(values: List[Any], size: int, *, seed: int, epoch: int) -> Iterable[List[Any]]:
    indices = list(range(len(values)))
    random.Random(seed + epoch).shuffle(indices)
    for start in range(0, len(indices), size):
        yield [values[index] for index in indices[start : start + size]]


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _load_audio(path: str) -> tuple[Any, int]:
    try:
        import soundfile as sf

        values, rate = sf.read(path, dtype="float32")
        if getattr(values, "ndim", 1) > 1:
            values = values.mean(axis=1)
        return values, int(rate)
    except ImportError:
        try:
            import torchaudio

            values, rate = torchaudio.load(path)
            return values.mean(dim=0).numpy(), int(rate)
        except ImportError as exc:
            raise RuntimeError(
                "Audio classification requires soundfile or torchaudio"
            ) from exc


def _classification_kind(rows: List[Mapping[str, Any]]) -> str:
    media = next((str(row.get("media") or "") for row in rows if row.get("media")), "")
    suffix = Path(media).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return "image"
    if suffix in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}:
        return "audio"
    return "text"


def _label_contract(
    rows: List[Mapping[str, Any]], *, multi_label: bool
) -> tuple[List[str], Dict[str, int]]:
    values = set()
    for row in rows:
        labels = row.get("labels") if multi_label else [row.get("label")]
        if isinstance(labels, str) and multi_label:
            try:
                parsed = json.loads(labels)
                labels = parsed if isinstance(parsed, list) else [labels]
            except json.JSONDecodeError:
                labels = [part.strip() for part in labels.split(",") if part.strip()]
        for label in labels or []:
            if label is not None and str(label).strip():
                values.add(str(label))
    labels = sorted(values)
    if len(labels) < 2:
        raise ValueError("classification training requires at least two labels")
    return labels, {label: index for index, label in enumerate(labels)}


def _train_classification(config: SpecializedTrainConfig, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModelForAudioClassification,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    kind = _classification_kind(rows)
    labels, label_to_id = _label_contract(rows, multi_label=config.multi_label)
    common = {
        "num_labels": len(labels),
        "id2label": {index: label for label, index in label_to_id.items()},
        "label2id": label_to_id,
    }
    if kind == "image":
        processor = AutoImageProcessor.from_pretrained(config.model)
        model = AutoModelForImageClassification.from_pretrained(
            config.model, ignore_mismatched_sizes=True, **common
        )
    elif kind == "audio":
        processor = AutoFeatureExtractor.from_pretrained(config.model)
        model = AutoModelForAudioClassification.from_pretrained(
            config.model, ignore_mismatched_sizes=True, **common
        )
    else:
        processor = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model, ignore_mismatched_sizes=True, **common
        )
    device = _device(torch)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []
    steps = 0
    for epoch in range(config.epochs):
        for batch in _batches(rows, config.batch_size, seed=config.seed, epoch=epoch):
            if kind == "image":
                from PIL import Image

                images = [Image.open(str(row["media"])).convert("RGB") for row in batch]
                inputs = processor(images=images, return_tensors="pt")
            elif kind == "audio":
                loaded = [_load_audio(str(row["media"])) for row in batch]
                target_rate = int(getattr(processor, "sampling_rate", 16000) or 16000)
                if any(rate != target_rate for _, rate in loaded):
                    raise ValueError(
                        f"audio sample rate must match processor rate {target_rate}"
                    )
                inputs = processor(
                    [values for values, _ in loaded],
                    sampling_rate=target_rate,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    [str(row.get("input") or "") for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt",
                )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if config.multi_label:
                target = torch.zeros((len(batch), len(labels)), dtype=torch.float32)
                for row_index, row in enumerate(batch):
                    raw_labels = row.get("labels") or []
                    if isinstance(raw_labels, str):
                        try:
                            raw_labels = json.loads(raw_labels)
                        except json.JSONDecodeError:
                            raw_labels = raw_labels.split(",")
                    for label in raw_labels:
                        target[row_index, label_to_id[str(label).strip()]] = 1.0
                target = target.to(device)
                logits = model(**inputs).logits
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target
                )
            else:
                target = torch.tensor(
                    [label_to_id[str(row["label"])] for row in batch],
                    dtype=torch.long,
                    device=device,
                )
                loss = model(**inputs, labels=target).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            print(
                "HALO_METRIC "
                + json.dumps(
                    {
                        "step": steps,
                        "train_loss": losses[-1],
                        "epoch": epoch + 1,
                    }
                ),
                flush=True,
            )
    output = Path(config.output_dir)
    model.save_pretrained(output / "final_model")
    processor.save_pretrained(output / "final_model")
    (output / "final_model" / "label_map.json").write_text(
        json.dumps({"labels": labels, "multi_label": config.multi_label}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "steps": steps,
        "losses": losses,
        "task_modality": kind,
        "labels": labels,
    }


def _train_embedding(config: SpecializedTrainConfig, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModel.from_pretrained(config.model)
    device = _device(torch)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []
    steps = 0
    for epoch in range(config.epochs):
        for batch in _batches(rows, config.batch_size, seed=config.seed, epoch=epoch):
            anchors = tokenizer(
                [str(row["anchor"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            positives = tokenizer(
                [str(row["positive"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            anchors = {key: value.to(device) for key, value in anchors.items()}
            positives = {key: value.to(device) for key, value in positives.items()}
            anchor_vectors = torch.nn.functional.normalize(
                _mean_pool(model(**anchors).last_hidden_state, anchors["attention_mask"]),
                dim=1,
            )
            positive_vectors = torch.nn.functional.normalize(
                _mean_pool(
                    model(**positives).last_hidden_state,
                    positives["attention_mask"],
                ),
                dim=1,
            )
            logits = anchor_vectors @ positive_vectors.T / 0.05
            target = torch.arange(len(batch), device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            print(
                "HALO_METRIC "
                + json.dumps({"step": steps, "train_loss": losses[-1], "epoch": epoch + 1}),
                flush=True,
            )
    output = Path(config.output_dir)
    model.save_pretrained(output / "final_model")
    tokenizer.save_pretrained(output / "final_model")
    return {"steps": steps, "losses": losses, "embedding_dimension": model.config.hidden_size}


def _train_reranker(config: SpecializedTrainConfig, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=1, ignore_mismatched_sizes=True
    )
    device = _device(torch)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []
    steps = 0
    for epoch in range(config.epochs):
        for batch in _batches(rows, config.batch_size, seed=config.seed, epoch=epoch):
            encoded = tokenizer(
                [str(row["query"]) for row in batch],
                [str(row["document"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            targets = torch.tensor(
                [float(row["relevance"]) for row in batch],
                dtype=torch.float32,
                device=device,
            )
            logits = model(**encoded).logits.reshape(-1)
            if all(value in {0.0, 1.0} for value in targets.detach().cpu().tolist()):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets
                )
            else:
                loss = torch.nn.functional.mse_loss(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            print(
                "HALO_METRIC "
                + json.dumps({"step": steps, "train_loss": losses[-1], "epoch": epoch + 1}),
                flush=True,
            )
    output = Path(config.output_dir)
    model.save_pretrained(output / "final_model")
    tokenizer.save_pretrained(output / "final_model")
    return {"steps": steps, "losses": losses}


def run_specialized_training(config: SpecializedTrainConfig) -> Dict[str, Any]:
    if config.task not in {"classify", "embed", "rerank"}:
        raise ValueError(f"unsupported specialized task: {config.task}")
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    rows = _records(config.dataset, config.max_samples)
    if not rows:
        raise ValueError("specialized training dataset is empty")
    random.seed(config.seed)
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        "HALO_STAGE "
        + json.dumps({"key": "building_trainer", "message": "Loading task model and processor."}),
        flush=True,
    )
    if config.task == "classify":
        result = _train_classification(config, rows)
    elif config.task == "embed":
        result = _train_embedding(config, rows)
    else:
        result = _train_reranker(config, rows)
    losses = list(result.pop("losses", []))
    summary = {
        "run_id": output.name,
        "status": "completed",
        "modality": config.task,
        "model": config.model,
        "dataset": config.dataset,
        "proof_run": config.proof_run,
        "weights_updated": bool(result.get("steps")),
        "optimizer_steps": int(result.get("steps") or 0),
        "final_train_loss": losses[-1] if losses else None,
        "initial_train_loss": losses[0] if losses else None,
        "elapsed_seconds": time.time() - started,
        "final_model_path": str(output / "final_model"),
        "label_schema_revision_id": config.label_schema_revision_id,
        "retrieval_corpus_id": config.retrieval_corpus_id,
        "task_contract": {
            "task": config.task,
            "loss_adapter": {
                "classify": "cross_entropy_or_bce_v1",
                "embed": "multiple_negative_ranking_v1",
                "rerank": "binary_scalar_cross_encoder_v1",
            }[config.task],
            **result,
        },
        "config": asdict(config),
    }
    (output / "final_model" / "task_config.json").write_text(
        json.dumps(
            {
                "task": config.task,
                "multi_label": config.multi_label,
                "label_schema_revision_id": config.label_schema_revision_id,
                "retrieval_corpus_id": config.retrieval_corpus_id,
                "loss_adapter": summary["task_contract"]["loss_adapter"],
                "task_modality": summary["task_contract"].get("task_modality", "text"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Qualification starts with a truthful artifact contract: reload the
    # processor and model from the published directory, then exercise one
    # representative input before the run can report success.
    runtime = SpecializedServingRuntime(output / "final_model")
    if config.task == "classify":
        modality = str(summary["task_contract"].get("task_modality") or "text")
        fixed_input = (
            rows[0].get("media")
            if modality in {"image", "audio"}
            else rows[0].get("input")
        )
        verification_output = runtime.classify([fixed_input], top_k=1)
    elif config.task == "embed":
        fixed_input = str(rows[0].get("anchor") or rows[0].get("input") or "")
        verification_output = runtime.embed([fixed_input])
    else:
        query = str(rows[0].get("query") or "")
        documents = rows[0].get("candidates") or [rows[0].get("document") or ""]
        if isinstance(documents, str):
            documents = [documents]
        verification_output = runtime.rerank(
            query, [str(value) for value in documents]
        )
    summary["artifact_verification"] = {
        "valid": True,
        "model_reloaded": True,
        "processor_reloaded": True,
        "task_contract_verified": True,
        "fixed_input_inference": True,
        "output_shape": (
            [len(verification_output), len(verification_output[0])]
            if verification_output and isinstance(verification_output[0], list)
            else [len(verification_output)]
        ),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from halo_forge.replay import capture_manifest, save_manifest

    manifest = capture_manifest(
        run_id=output.name,
        modality=config.task,
        model_name=config.model,
        seed=config.seed,
        config=asdict(config),
        dataset_file=Path(config.dataset),
        training_outcome_binding={
            "proof_run": config.proof_run,
        },
        specialized_task_binding={
            "task": config.task,
            "label_schema_revision_id": config.label_schema_revision_id,
            "retrieval_corpus_id": config.retrieval_corpus_id,
            "loss_adapter": summary["task_contract"]["loss_adapter"],
            "task_artifact_contract": summary["task_contract"],
        },
    )
    save_manifest(manifest, output)
    print(
        "HALO_ARTIFACT "
        + json.dumps(
            {
                "type": "final_model",
                "state": "final_model",
                "path": summary["final_model_path"],
            }
        ),
        flush=True,
    )
    print(
        "HALO_STAGE "
        + json.dumps({"key": "completed", "message": "Specialized task training completed."}),
        flush=True,
    )
    return summary


class SpecializedServingRuntime:
    """Lazy local inference for verified specialized task artifacts."""

    def __init__(self, artifact_path: str | Path):
        self.path = Path(artifact_path).expanduser().resolve()
        config_path = self.path / "task_config.json"
        if not config_path.is_file():
            raise ValueError("specialized artifact is missing task_config.json")
        self.task_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.task = str(self.task_config.get("task") or "")
        if self.task not in {"classify", "embed", "rerank"}:
            raise ValueError("artifact does not declare a supported specialized task")
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoImageProcessor,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForImageClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        modality = str(self.task_config.get("task_modality") or "text")
        if self.task == "embed":
            self._processor = AutoTokenizer.from_pretrained(self.path)
            self._model = AutoModel.from_pretrained(self.path)
        elif self.task == "rerank" or modality == "text":
            self._processor = AutoTokenizer.from_pretrained(self.path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.path)
        elif modality == "image":
            self._processor = AutoImageProcessor.from_pretrained(self.path)
            self._model = AutoModelForImageClassification.from_pretrained(self.path)
        else:
            self._processor = AutoFeatureExtractor.from_pretrained(self.path)
            self._model = AutoModelForAudioClassification.from_pretrained(self.path)
        self._torch = torch
        self._device = _device(torch)
        self._model.to(self._device)
        self._model.eval()

    def embed(self, inputs: List[str]) -> List[List[float]]:
        if self.task != "embed":
            raise ValueError("artifact is not an embedding model")
        self._load()
        torch = self._torch
        encoded = self._processor(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            vectors = torch.nn.functional.normalize(
                _mean_pool(
                    self._model(**encoded).last_hidden_state,
                    encoded["attention_mask"],
                ),
                dim=1,
            )
        return vectors.detach().cpu().tolist()

    def classify(
        self, inputs: List[Any], *, top_k: int = 1
    ) -> List[List[Dict[str, Any]]]:
        if self.task != "classify":
            raise ValueError("artifact is not a classification model")
        self._load()
        torch = self._torch
        modality = str(self.task_config.get("task_modality") or "text")
        if modality == "image":
            from PIL import Image

            encoded = self._processor(
                images=[Image.open(str(value)).convert("RGB") for value in inputs],
                return_tensors="pt",
            )
        elif modality == "audio":
            loaded = [_load_audio(str(value)) for value in inputs]
            rate = int(getattr(self._processor, "sampling_rate", 16000) or 16000)
            if any(actual != rate for _, actual in loaded):
                raise ValueError(f"audio sample rate must match processor rate {rate}")
            encoded = self._processor(
                [values for values, _ in loaded],
                sampling_rate=rate,
                padding=True,
                return_tensors="pt",
            )
        else:
            encoded = self._processor(
                [str(value) for value in inputs],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
        probabilities = (
            torch.sigmoid(logits)
            if bool(self.task_config.get("multi_label"))
            else torch.softmax(logits, dim=-1)
        )
        labels = getattr(self._model.config, "id2label", {}) or {}
        output = []
        for row in probabilities.detach().cpu():
            values, indices = torch.topk(row, min(max(1, int(top_k)), row.numel()))
            output.append(
                [
                    {
                        "label": str(labels.get(int(index), labels.get(str(int(index)), index))),
                        "score": float(score),
                    }
                    for score, index in zip(values, indices)
                ]
            )
        return output

    def rerank(
        self, query: str, documents: List[str], *, top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if self.task != "rerank":
            raise ValueError("artifact is not a reranking model")
        self._load()
        torch = self._torch
        encoded = self._processor(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            scores = self._model(**encoded).logits.reshape(-1).detach().cpu().tolist()
        ranked = sorted(
            (
                {"index": index, "document": document, "score": float(score)}
                for index, (document, score) in enumerate(zip(documents, scores))
            ),
            key=lambda value: value["score"],
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked


__all__ = [
    "SpecializedServingRuntime",
    "SpecializedTrainConfig",
    "run_specialized_training",
]
