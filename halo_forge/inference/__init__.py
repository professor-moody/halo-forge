"""
Inference Optimization Module

Tools for optimizing trained models for deployment:
- Quantization-aware training (QAT)
- Model export (GGUF, ONNX, MLX)
- Inference benchmarking
"""

from halo_forge.inference.verifier import InferenceOptimizationVerifier
from halo_forge.inference.optimizer import InferenceOptimizer, OptimizationConfig
from halo_forge.inference.quantization import QATTrainer, prepare_qat, convert_to_quantized
from halo_forge.inference.calibration import CalibrationDataset, CalibrationConfig

__all__ = [
    # Verifier
    "InferenceOptimizationVerifier",
    # Optimizer
    "InferenceOptimizer",
    "OptimizationConfig",
    # Quantization
    "QATTrainer",
    "prepare_qat",
    "convert_to_quantized",
    # Calibration
    "CalibrationDataset",
    "CalibrationConfig",
]
