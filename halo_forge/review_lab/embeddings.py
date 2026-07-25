"""Pinned, modality-aware embedding generation for diversity acquisition."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .errors import ReviewValidationError


def _model_identity(value: str, modality: str) -> Tuple[str, str, str]:
    requested = str(value or "").strip()
    declared_modality = ""
    if ":" in requested:
        prefix, remainder = requested.split(":", 1)
        if prefix in {"text", "image", "audio"}:
            declared_modality, requested = prefix, remainder
    if declared_modality and declared_modality != modality:
        raise ReviewValidationError(
            f"embedding model is pinned for {declared_modality}, not {modality}"
        )
    if "@" not in requested:
        raise ReviewValidationError(
            "embedding generation requires model@revision (a commit/tag or immutable local revision)"
        )
    model_id, revision = requested.rsplit("@", 1)
    model_id, revision = model_id.strip(), revision.strip()
    if not model_id or not revision or revision.lower() in {"main", "master", "latest"}:
        raise ReviewValidationError(
            "embedding generation requires a pinned model revision, not main/latest"
        )
    return declared_modality or modality, model_id, revision


def _normalize(vector: Sequence[Any]) -> List[float]:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError("embedding backend returned non-numeric values") from exc
    norm = sum(value * value for value in values) ** 0.5
    if not values or not norm:
        raise ReviewValidationError("embedding backend returned an empty or zero vector")
    return [value / norm for value in values]


def _reference(record: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, Mapping):
            value = value.get("path") or value.get("filename")
        if isinstance(value, (str, os.PathLike)) and str(value):
            return os.fspath(value)
    return ""


def _asset_path(record: Mapping[str, Any], source: Mapping[str, Any], *fields: str) -> Path:
    reference = _reference(record, *fields)
    if not reference:
        raise ReviewValidationError(
            f"{fields[0]} diversity acquisition requires an existing media reference"
        )
    mapped = source.get("asset_paths")
    if isinstance(mapped, Mapping) and mapped.get(reference):
        path = Path(str(mapped[reference])).expanduser()
    else:
        path = Path(reference).expanduser()
    if not path.is_absolute() and source.get("asset_root"):
        path = Path(str(source["asset_root"])).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise ReviewValidationError(f"embedding media asset is missing: {path}")
    return path


def _record_modality(record: Mapping[str, Any]) -> str:
    if _reference(record, "image", "image_path"):
        return "image"
    if _reference(record, "audio", "audio_path"):
        return "audio"
    return "text"


class PinnedEmbeddingEngine:
    """Generate real model embeddings; never substitute synthetic evidence."""

    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str, str], Any] = {}

    def _sentence_model(self, model_id: str, revision: str) -> Any:
        key = ("sentence_transformers", model_id, revision)
        if key not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ReviewValidationError(
                    "text/image diversity requires sentence-transformers; install halo-forge[data-lab]"
                ) from exc
            self._models[key] = SentenceTransformer(model_id, revision=revision)
        return self._models[key]

    def _audio_model(self, model_id: str, revision: str) -> Tuple[Any, Any]:
        key = ("clap", model_id, revision)
        if key not in self._models:
            try:
                from transformers import ClapModel, ClapProcessor
            except ImportError as exc:
                raise ReviewValidationError(
                    "audio diversity requires transformers with CLAP support"
                ) from exc
            self._models[key] = (
                ClapModel.from_pretrained(model_id, revision=revision),
                ClapProcessor.from_pretrained(model_id, revision=revision),
            )
        return self._models[key]

    def _embed_text(
        self, records: Sequence[Mapping[str, Any]], model_id: str, revision: str
    ) -> List[List[float]]:
        from halo_forge.data_lab.integrations import record_text

        texts = [record_text(value) for value in records]
        if any(not value.strip() for value in texts):
            raise ReviewValidationError(
                "text diversity requires non-empty record content for every candidate"
            )
        vectors = self._sentence_model(model_id, revision).encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_normalize(vector) for vector in vectors]

    def _embed_images(
        self,
        records: Sequence[Mapping[str, Any]],
        sources: Sequence[Mapping[str, Any]],
        model_id: str,
        revision: str,
    ) -> List[List[float]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ReviewValidationError("image diversity requires Pillow") from exc
        model = self._sentence_model(model_id, revision)
        result: List[List[float]] = []
        for start in range(0, len(records), 16):
            images = []
            try:
                for record, source in zip(records[start : start + 16], sources[start : start + 16]):
                    image = Image.open(_asset_path(record, source, "image", "image_path"))
                    images.append(image.convert("RGB"))
                vectors = model.encode(
                    images,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                result.extend(_normalize(vector) for vector in vectors)
            finally:
                for image in images:
                    image.close()
        return result

    def _embed_audio(
        self,
        records: Sequence[Mapping[str, Any]],
        sources: Sequence[Mapping[str, Any]],
        model_id: str,
        revision: str,
    ) -> List[List[float]]:
        try:
            import soundfile as sf
            import torch
        except ImportError as exc:
            raise ReviewValidationError("audio diversity requires soundfile and torch") from exc
        model, processor = self._audio_model(model_id, revision)
        target_rate = int(getattr(processor.feature_extractor, "sampling_rate", 48_000))
        result: List[List[float]] = []
        for record, source in zip(records, sources):
            samples, sample_rate = sf.read(
                _asset_path(record, source, "audio", "audio_path"),
                dtype="float32",
                always_2d=False,
            )
            if getattr(samples, "ndim", 1) > 1:
                samples = samples.mean(axis=1)
            if int(sample_rate) != target_rate:
                try:
                    import librosa
                except ImportError as exc:
                    raise ReviewValidationError(
                        "audio diversity needs librosa to resample non-48kHz media"
                    ) from exc
                samples = librosa.resample(
                    samples,
                    orig_sr=int(sample_rate),
                    target_sr=target_rate,
                )
            inputs = processor(audios=samples, sampling_rate=target_rate, return_tensors="pt")
            with torch.inference_mode():
                vector = model.get_audio_features(**inputs)[0].detach().cpu().tolist()
            result.append(_normalize(vector))
        return result

    def embed_envelopes(
        self,
        envelopes: Sequence[Mapping[str, Any]],
        *,
        embedding_revision: str,
    ) -> List[Dict[str, Any]]:
        if not envelopes:
            return []
        records = [dict(value.get("record") or {}) for value in envelopes]
        sources = [dict(value.get("source") or {}) for value in envelopes]
        modalities = {_record_modality(record) for record in records}
        if len(modalities) != 1:
            raise ReviewValidationError(
                "one diversity stratum must use a single embedding modality"
            )
        modality = next(iter(modalities))
        _, model_id, revision = _model_identity(embedding_revision, modality)
        if modality == "image":
            vectors = self._embed_images(records, sources, model_id, revision)
        elif modality == "audio":
            vectors = self._embed_audio(records, sources, model_id, revision)
        else:
            vectors = self._embed_text(records, model_id, revision)
        runtime = {
            "engine": "halo-forge-pinned-embedding-v1",
            "model_id": model_id,
            "model_revision": revision,
            "modality": modality,
            "implementation_hash": hashlib.sha256(b"halo-forge-pinned-embedding-v1").hexdigest(),
        }
        output: List[Dict[str, Any]] = []
        for envelope, vector in zip(envelopes, vectors):
            value = dict(envelope)
            evidence = dict(value.get("evidence") or {})
            evidence.update(
                embedding=vector,
                embedding_revision=embedding_revision,
                embedding_model_revision=revision,
                embedding_provenance=runtime,
            )
            value["evidence"] = evidence
            output.append(value)
        return output


__all__ = ["PinnedEmbeddingEngine"]
