"""Artifact-producing operation orchestration and existing-engine adapters."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .hashing import atomic_write_json, fingerprint, hash_path, read_json
from .models import ArtifactBlob, ArtifactOperation, OperationSpec, operation_spec_from_dict
from .store import ArtifactIntegrityError, ArtifactStore, ArtifactStoreError


class ArtifactOperationError(ArtifactStoreError):
    """An artifact engine or its output validation failed."""


class ArtifactEngine(Protocol):
    """Lightweight contract for merge/conversion engines.

    Engines must write a file or directory at ``output_path`` and return only
    JSON-serializable evidence.  They never publish directly into the library.
    """

    def __call__(
        self,
        spec: OperationSpec,
        input_paths: tuple[Path, ...],
        output_path: Path,
    ) -> Optional[Mapping[str, Any]]: ...


OutputVerifier = Callable[[Path, OperationSpec], bool | Mapping[str, Any]]


def _runtime_versions(*packages: str) -> dict[str, str]:
    from halo_forge.version import PACKAGE_VERSION

    values = {"halo-forge": PACKAGE_VERSION}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            continue
    return values


class ArtifactOperationService:
    """Run immutable, reusable operations through same-filesystem staging."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        engines: Optional[Mapping[str, ArtifactEngine]] = None,
        output_verifier: Optional[OutputVerifier] = None,
    ):
        self.store = store
        self.engines: dict[str, ArtifactEngine] = dict(engines or {})
        self.output_verifier = output_verifier

    def register_engine(self, operation_type: str, engine: ArtifactEngine) -> None:
        if operation_type in self.engines:
            raise ValueError(f"An artifact engine is already registered for {operation_type!r}")
        self.engines[operation_type] = engine

    def _manifest_path(self, operation_fingerprint: str) -> Path:
        return self.store.metadata_dir / "operations" / f"{operation_fingerprint}.json"

    @staticmethod
    def _from_dict(value: Mapping[str, Any]) -> ArtifactOperation:
        return ArtifactOperation(
            id=str(value["id"]),
            fingerprint=str(value["fingerprint"]),
            spec=operation_spec_from_dict(value["spec"]),
            status=str(value["status"]),
            created_at=str(value["created_at"]),
            completed_at=value.get("completed_at"),
            output_content_hash=value.get("output_content_hash"),
            output_location_id=value.get("output_location_id"),
            engine_metadata=dict(value.get("engine_metadata") or {}),
            error=value.get("error"),
            reused=bool(value.get("reused", False)),
        )

    def get(self, operation_fingerprint: str) -> ArtifactOperation:
        path = self._manifest_path(operation_fingerprint)
        if not path.is_file():
            raise KeyError(f"Unknown artifact operation: {operation_fingerprint}")
        return self._from_dict(read_json(path))

    def _reusable(self, spec: OperationSpec) -> Optional[ArtifactOperation]:
        path = self._manifest_path(spec.fingerprint)
        if not path.is_file():
            return None
        operation = self._from_dict(read_json(path))
        if operation.spec.fingerprint != spec.fingerprint or operation.status != "completed":
            return None
        if not operation.output_content_hash:
            return None
        try:
            location = self.store.resolve_location(operation.output_content_hash)
            verification = self.store.verify(location.id, structural=True)
        except (KeyError, ArtifactStoreError):
            return None
        if not verification.passed:
            return None
        return replace(operation, reused=True)

    @staticmethod
    def _resolved_output_identity(
        spec: OperationSpec, engine_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = engine_metadata.get("result")
        result = dict(result) if isinstance(result, Mapping) else {}
        actual_quantization = (
            str(
                engine_metadata.get("actual_output_quantization")
                or result.get("actual_quantization")
                or spec.output_quantization
                or ""
            )
            .strip()
            .lower()
            or None
        )
        fallback_used = bool(
            engine_metadata.get("unquantized_fallback_used")
            or result.get("unquantized_fallback_used")
        )
        requested_quantization = (
            str(spec.output_quantization).strip().lower() if spec.output_quantization else None
        )
        if actual_quantization != requested_quantization and not fallback_used:
            raise ArtifactIntegrityError(
                "Artifact engine output quantization differs from its immutable request "
                "without reporting an explicit fallback"
            )
        if fallback_used and not bool(spec.parameters.get("allow_unquantized_fallback", False)):
            raise ArtifactIntegrityError(
                "Artifact engine used an unquantized fallback that the operation did not allow"
            )
        quantized_values = {"q4", "q8", "int4", "int8"}
        output_kind = spec.output_kind
        if spec.operation_type == "quantize" and actual_quantization not in quantized_values:
            output_kind = "converted"
        output_dtype = spec.output_dtype
        if actual_quantization in {"fp16", "bf16", "fp32"}:
            output_dtype = actual_quantization
        return {
            "artifact_kind": output_kind,
            "format": spec.output_format,
            "dtype": output_dtype,
            "requested_quantization": requested_quantization,
            "actual_quantization": actual_quantization,
            "quantization_method": (
                "post_training" if actual_quantization in quantized_values else None
            ),
            "unquantized_fallback_used": fallback_used,
        }

    def _verify_staged_output(
        self,
        output: Path,
        spec: OperationSpec,
        resolved_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        digest = hash_path(output)
        temporary_blob = ArtifactBlob(
            content_hash=digest.content_hash,
            artifact_kind=str(resolved_output["artifact_kind"]),
            format=str(resolved_output["format"]),
            dtype=resolved_output.get("dtype"),
            quantization=resolved_output.get("actual_quantization"),
            quantization_method=resolved_output.get("quantization_method"),
            size_bytes=digest.size_bytes,
            file_count=digest.file_count,
            created_at=self.store._now(),
        )
        checks, errors = self.store._structural_checks(temporary_blob, output)
        evidence: dict[str, Any] = {
            "content_hash": digest.content_hash,
            "structural_checks": checks,
            "errors": list(errors),
            "structural_checked": True,
            "loadability_checked": False,
            "round_trip_checked": False,
            "verification_level": "structural_verified" if not errors else "failed",
        }
        if errors:
            raise ArtifactIntegrityError("; ".join(errors))
        if self.output_verifier is not None:
            result = self.output_verifier(output, spec)
            if isinstance(result, Mapping):
                result_dict = dict(result)
                passed = bool(result_dict.get("passed"))
                evidence["output_verifier"] = result_dict
                if passed and bool(result_dict.get("loadability_checked")):
                    evidence["loadability_checked"] = True
                    evidence["verification_level"] = "load_verified"
                if passed and bool(result_dict.get("round_trip_checked")):
                    evidence["round_trip_checked"] = True
                    evidence["verification_level"] = "round_trip_verified"
            else:
                passed = bool(result)
                evidence["output_verifier"] = {"passed": passed}
            if not passed:
                raise ArtifactIntegrityError("Artifact output verifier rejected the staged payload")
        return evidence

    def run(self, spec: OperationSpec) -> ArtifactOperation:
        """Run or reuse an artifact operation.

        Ordered input hashes, resolved parameters, tool identity, and output
        declarations all participate in the fingerprint.  A completed result is
        reused only while its payload still verifies.
        """

        reused = self._reusable(spec)
        if reused is not None:
            return reused
        engine = self.engines.get(spec.operation_type)
        if engine is None:
            raise ArtifactOperationError(
                f"No artifact engine is registered for {spec.operation_type!r}"
            )
        input_locations = [
            self.store.resolve_location(content_hash) for content_hash in spec.input_content_hashes
        ]
        for location in input_locations:
            report = self.store.verify(location.id, structural=True)
            if not report.satisfies("structural_verified"):
                raise ArtifactIntegrityError(
                    f"Operation input {location.content_hash} failed structural verification: "
                    + ("; ".join(report.errors) or report.verification_level)
                )
        input_paths = tuple(Path(item.path) for item in input_locations)
        operation_id = f"operation-{spec.fingerprint[:24]}"
        created_at = self.store._now()
        stage = Path(
            tempfile.mkdtemp(
                prefix=f"operation-{spec.fingerprint[:12]}-", dir=self.store.staging_dir
            )
        )
        output = stage / "output"
        try:
            engine_metadata = dict(engine(spec, input_paths, output) or {})
            if not output.exists():
                raise ArtifactOperationError(
                    f"Artifact engine {spec.operation_type!r} did not create {output}"
                )
            resolved_output = self._resolved_output_identity(spec, engine_metadata)
            verification_evidence = self._verify_staged_output(output, spec, resolved_output)
            resolved_output = {
                **resolved_output,
                "content_hash": verification_evidence["content_hash"],
            }
            resolved_output["identity_hash"] = fingerprint(resolved_output)
            engine_metadata["resolved_output"] = resolved_output
            engine_metadata["verification"] = verification_evidence
            registration = self.store.import_artifact(
                output,
                artifact_kind=str(resolved_output["artifact_kind"]),
                artifact_format=str(resolved_output["format"]),
                managed=True,
                dtype=resolved_output.get("dtype"),
                quantization=resolved_output.get("actual_quantization"),
                quantization_method=resolved_output.get("quantization_method"),
                occurrence_id=f"artifact-{spec.fingerprint[:24]}",
                metadata={
                    "operation_id": operation_id,
                    "operation_fingerprint": spec.fingerprint,
                    "resolved_output": resolved_output,
                    "verification": verification_evidence,
                    "engine_evidence": {
                        key: value
                        for key, value in engine_metadata.items()
                        if key != "verification"
                    },
                },
            )
            self.store.record_lineage(
                child_content_hash=registration.blob.content_hash,
                # A conversion can be content-preserving (for example, an HF
                # normalization that finds nothing to rewrite). A blob DAG
                # cannot contain a self-edge; the operation manifest still
                # records the complete ordered input list for that no-op.
                parent_content_hashes=tuple(
                    value
                    for value in spec.input_content_hashes
                    if value != registration.blob.content_hash
                ),
                relationship=spec.operation_type,
                operation_fingerprint=spec.fingerprint,
            )
            operation = ArtifactOperation(
                id=operation_id,
                fingerprint=spec.fingerprint,
                spec=spec,
                status="completed",
                created_at=created_at,
                completed_at=self.store._now(),
                output_content_hash=registration.blob.content_hash,
                output_location_id=registration.location.id,
                engine_metadata=engine_metadata,
            )
            atomic_write_json(self._manifest_path(spec.fingerprint), operation.to_dict())
            return operation
        except Exception as exc:
            failed = ArtifactOperation(
                id=f"{operation_id}-attempt-{uuid.uuid4().hex[:12]}",
                fingerprint=spec.fingerprint,
                spec=spec,
                status="failed",
                created_at=created_at,
                completed_at=self.store._now(),
                error=str(exc),
            )
            atomic_write_json(
                self.store.metadata_dir
                / "operations"
                / f"{spec.fingerprint}.failed-{failed.id.rsplit('-', 1)[-1]}.json",
                failed.to_dict(),
            )
            if isinstance(exc, ArtifactOperationError):
                raise
            raise ArtifactOperationError(
                f"Artifact operation {spec.operation_type!r} failed: {exc}"
            ) from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def existing_merge_engine(
    spec: OperationSpec,
    input_paths: tuple[Path, ...],
    output_path: Path,
) -> Mapping[str, Any]:
    """Lazy adapter for :mod:`halo_forge.inference.merge`.

    ``base_model`` remains an explicit resolved parameter because it can be a
    pinned Hugging Face identifier.  Input artifacts are adapters in their
    exact ordered merge sequence.
    """

    from halo_forge.inference.merge import merge

    base_model = str(spec.parameters.get("base_model") or "").strip()
    base_input_index = spec.parameters.get("base_input_index")
    if base_input_index is not None:
        index = int(base_input_index)
        if index < 0 or index >= len(input_paths):
            raise ValueError("Merge operation base_input_index is invalid")
        base_model = str(input_paths[index])
        adapter_paths = tuple(path for ordinal, path in enumerate(input_paths) if ordinal != index)
    else:
        adapter_paths = input_paths
        base_revision = str(spec.parameters.get("base_revision") or "").strip()
        if not base_model or not base_revision:
            raise ValueError(
                "Merge operation requires a resolved base artifact or pinned base revision"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Pinned remote merge bases require huggingface-hub") from exc
        base_model = snapshot_download(repo_id=base_model, revision=base_revision)
    if spec.operation_type == "bake":
        if len(adapter_paths) != 1:
            raise ValueError("Bake expects exactly one adapter artifact input")
        result = merge(
            operation="bake",
            base_model=base_model,
            adapter_path=str(adapter_paths[0]),
            output_path=str(output_path),
            trust_remote_code=bool(spec.parameters.get("trust_remote_code", False)),
        )
    elif spec.operation_type == "combine":
        result = merge(
            operation="combine",
            base_model=base_model,
            adapter_paths=[str(item) for item in adapter_paths],
            weights=spec.parameters.get("weights"),
            method=str(spec.parameters.get("method") or "dare_ties"),
            output_path=str(output_path),
            bake_after_merge=bool(spec.parameters.get("bake_after_merge", False)),
            trust_remote_code=bool(spec.parameters.get("trust_remote_code", False)),
            svd_rank=spec.parameters.get("svd_rank"),
        )
    else:
        raise ValueError(f"existing_merge_engine cannot run {spec.operation_type!r}")
    return {
        "backend": "halo_forge.inference.merge",
        "resolved_base": {
            "kind": "artifact" if base_input_index is not None else "pinned_revision",
            "model": str(spec.parameters.get("base_model") or ""),
            "revision": spec.parameters.get("base_revision"),
            "content_hash": spec.parameters.get("base_content_hash"),
            "occurrence_id": spec.parameters.get("base_occurrence_id"),
            "resolved_path": base_model,
        },
        "runtime_versions": _runtime_versions("transformers", "peft", "torch"),
        "result": result.to_dict(),
    }


def existing_convert_engine(
    spec: OperationSpec,
    input_paths: tuple[Path, ...],
    output_path: Path,
) -> Mapping[str, Any]:
    """Lazy adapter for the currently verified HF/MLX/GGUF converter.

    ONNX is intentionally rejected here until Halo Forge has a verified ONNX
    conversion engine.  q4/q8 conversions are described as post-training
    quantization and never as QAT.
    """

    from halo_forge.inference.convert import convert, list_supported_formats

    if len(input_paths) != 1:
        raise ValueError("Conversion expects exactly one artifact input")
    target_format = spec.output_format.strip().lower()
    if target_format not in list_supported_formats():
        raise ValueError(
            f"No verified {target_format!r} conversion engine is available; "
            f"current formats: {', '.join(list_supported_formats())}"
        )
    if target_format == "gguf":
        engine_output = output_path / "model.gguf"
        output_path.mkdir(parents=True)
    else:
        engine_output = output_path
    resolved_quantization = spec.output_quantization or str(
        spec.parameters.get("quantization")
        or ("q4" if target_format in {"mlx", "gguf"} else spec.output_dtype or "bf16")
    )
    result = convert(
        source=str(input_paths[0]),
        output_path=str(engine_output),
        target_format=target_format,
        quantization=resolved_quantization,
        trust_remote_code=bool(spec.parameters.get("trust_remote_code", False)),
        allow_unquantized_fallback=bool(spec.parameters.get("allow_unquantized_fallback", False)),
    )
    result_value = result.to_dict()
    actual_quantization = str(
        result_value.get("actual_quantization") or resolved_quantization
    ).lower()
    return {
        "backend": "halo_forge.inference.convert",
        "runtime_versions": _runtime_versions(
            "transformers", "torch", "mlx-lm", "llama-cpp-python"
        ),
        "requested_output_quantization": resolved_quantization.lower(),
        "actual_output_quantization": actual_quantization,
        "unquantized_fallback_used": bool(result_value.get("unquantized_fallback_used", False)),
        "quantization_method": (
            "post_training" if actual_quantization in {"q4", "q8", "int4", "int8"} else None
        ),
        "result": result_value,
    }


def default_engines() -> dict[str, ArtifactEngine]:
    return {
        "bake": existing_merge_engine,
        "combine": existing_merge_engine,
        "convert": existing_convert_engine,
        "quantize": existing_convert_engine,
    }


def verify_round_trip(
    *,
    source_generate: Callable[[str], str],
    artifact_generate: Callable[[str], str],
    prompts: Optional[Sequence[str]] = None,
    char_overlap_threshold: float = 0.7,
    first_token_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the existing fixed-prompt verifier with supplied real loaders."""

    from halo_forge.inference.verify_export import compare_generation

    report = compare_generation(
        source_generate=source_generate,
        exported_generate=artifact_generate,
        prompts=prompts,
        char_overlap_threshold=char_overlap_threshold,
        first_token_threshold=first_token_threshold,
    )
    return report.to_dict()


__all__ = [
    "ArtifactEngine",
    "ArtifactOperationError",
    "ArtifactOperationService",
    "OutputVerifier",
    "default_engines",
    "existing_convert_engine",
    "existing_merge_engine",
    "verify_round_trip",
]
