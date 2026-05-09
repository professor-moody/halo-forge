"""Inference Optimization Module.

Exports are resolved lazily so lightweight commands can import
``halo_forge.inference`` without importing torch/transformers-heavy modules.
"""

__all__ = [
    "InferenceOptimizationVerifier",
    "InferenceOptimizer",
    "OptimizationConfig",
    "InferenceError",
    "DependencyError",
    "ValidationError",
    "ModelNotLoadedError",
    "check_dependencies",
    "validate_config",
    "QATTrainer",
    "prepare_qat",
    "convert_to_quantized",
    "CalibrationDataset",
    "CalibrationConfig",
]


def __getattr__(name: str):
    if name == "InferenceOptimizationVerifier":
        from halo_forge.inference.verifier import InferenceOptimizationVerifier

        return InferenceOptimizationVerifier
    if name in {
        "InferenceOptimizer",
        "OptimizationConfig",
        "InferenceError",
        "DependencyError",
        "ValidationError",
        "ModelNotLoadedError",
        "check_dependencies",
        "validate_config",
    }:
        from halo_forge.inference import optimizer

        return getattr(optimizer, name)
    if name in {"QATTrainer", "prepare_qat", "convert_to_quantized"}:
        from halo_forge.inference import quantization

        return getattr(quantization, name)
    if name in {"CalibrationDataset", "CalibrationConfig"}:
        from halo_forge.inference import calibration

        return getattr(calibration, name)
    raise AttributeError(name)
