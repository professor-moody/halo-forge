"""
Inference Optimizer

High-level interface for model optimization and export.
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch


@dataclass
class OptimizationConfig:
    """Configuration for inference optimization."""
    target_precision: str = "int4"  # int4, int8, fp16
    target_latency_ms: float = 50.0
    quality_threshold: float = 0.95
    calibration_samples: int = 512
    export_format: Optional[str] = None  # gguf, onnx, mlx
    output_dir: str = "models/optimized"


class InferenceOptimizer:
    """
    End-to-end inference optimization pipeline.
    
    Handles:
    1. Model quantization (QAT or post-training)
    2. Quality verification against baseline
    3. Export to deployment formats (GGUF, ONNX, MLX)
    
    Usage:
        optimizer = InferenceOptimizer(config)
        result = optimizer.optimize(
            model_path="models/trained",
            calibration_data="data/calibration.jsonl",
            eval_prompts=["Hello, how are you?"]
        )
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.model = None
        self.tokenizer = None
        self.baseline_model = None
    
    def load_model(self, model_path: str):
        """Load model for optimization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Keep a reference to original for baseline comparison
        self.baseline_model = model_path
        
        return self.model
    
    def quantize(
        self,
        calibration_data: Optional[str] = None,
        method: str = "post_training"  # post_training, qat
    ):
        """
        Quantize the model.
        
        Args:
            calibration_data: Path to calibration data JSONL
            method: Quantization method
            
        Returns:
            Quantized model
        """
        from halo_forge.inference.quantization import (
            quantize_model_simple,
            QATTrainer,
            QATConfig
        )
        from halo_forge.inference.calibration import CalibrationDataset
        
        if method == "post_training":
            # Simple post-training quantization
            output_path = Path(self.config.output_dir) / "quantized"
            
            quantize_model_simple(
                model_path=self.baseline_model,
                output_path=str(output_path),
                precision=self.config.target_precision,
                calibration_data=calibration_data
            )
            
            # Reload quantized model
            self.load_model(str(output_path))
            
        elif method == "qat":
            # Quantization-aware training
            if calibration_data is None:
                raise ValueError("QAT requires calibration_data")
            
            from halo_forge.inference.verifier import InferenceOptimizationVerifier
            
            # Load calibration data
            cal_dataset = CalibrationDataset.from_jsonl(
                calibration_data,
                self.tokenizer
            )
            dataloader = cal_dataset.get_dataloader()
            
            # Setup verifier
            verifier = InferenceOptimizationVerifier(
                baseline_model=self.model,
                target_latency_ms=self.config.target_latency_ms,
                quality_threshold=self.config.quality_threshold
            )
            
            # Run QAT
            config = QATConfig(
                target_precision=self.config.target_precision,
                calibration_samples=self.config.calibration_samples,
                output_dir=self.config.output_dir
            )
            
            trainer = QATTrainer(config)
            self.model = trainer.train(
                self.model,
                dataloader,
                verifier=verifier,
                eval_prompts=["Write a function to sort a list."]
            )
        
        return self.model
    
    def verify(self, eval_prompts: List[str]) -> Dict[str, Any]:
        """
        Verify optimized model quality.
        
        Args:
            eval_prompts: Prompts for evaluation
            
        Returns:
            Verification results
        """
        from halo_forge.inference.verifier import InferenceOptimizationVerifier
        
        verifier = InferenceOptimizationVerifier(
            baseline_model_name=self.baseline_model,
            target_latency_ms=self.config.target_latency_ms,
            quality_threshold=self.config.quality_threshold
        )
        
        result = verifier.verify(self.model, eval_prompts, self.tokenizer)
        
        verifier.cleanup()
        
        return {
            "success": result.success,
            "reward": result.reward,
            "metrics": result.metadata
        }
    
    def export(self, format: Optional[str] = None) -> str:
        """
        Export model to deployment format.
        
        Args:
            format: Export format (gguf, onnx, mlx)
            
        Returns:
            Path to exported model
        """
        format = format or self.config.export_format
        
        if format is None:
            raise ValueError("No export format specified")
        
        output_dir = Path(self.config.output_dir)
        
        if format == "gguf":
            from halo_forge.inference.export import GGUFExporter
            
            exporter = GGUFExporter()
            output_path = output_dir / "model.gguf"
            exporter.export(self.model, str(output_path))
            
        elif format == "onnx":
            from halo_forge.inference.export import ONNXExporter
            
            exporter = ONNXExporter()
            output_path = output_dir / "model.onnx"
            exporter.export(self.model, str(output_path))
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(output_path)
    
    def optimize(
        self,
        model_path: str,
        calibration_data: Optional[str] = None,
        eval_prompts: Optional[List[str]] = None,
        export_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full optimization pipeline.
        
        Args:
            model_path: Path to model
            calibration_data: Path to calibration data
            eval_prompts: Prompts for evaluation
            export_format: Optional export format
            
        Returns:
            Optimization results
        """
        results = {
            "model_path": model_path,
            "precision": self.config.target_precision,
            "success": False,
            "verification": None,
            "export_path": None
        }
        
        try:
            # Step 1: Load model
            self.load_model(model_path)
            
            # Step 2: Quantize
            print(f"\nQuantizing to {self.config.target_precision}...")
            self.quantize(calibration_data)
            
            # Step 3: Verify (if prompts provided)
            if eval_prompts:
                print("\nVerifying quality...")
                results["verification"] = self.verify(eval_prompts)
                results["success"] = results["verification"]["success"]
            else:
                results["success"] = True
            
            # Step 4: Export (if format specified)
            export_format = export_format or self.config.export_format
            if export_format:
                print(f"\nExporting to {export_format}...")
                results["export_path"] = self.export(export_format)
            
        except Exception as e:
            results["error"] = str(e)
            raise
        
        finally:
            # Cleanup
            self.cleanup()
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.baseline_model is not None and hasattr(self.baseline_model, 'parameters'):
            del self.baseline_model
            self.baseline_model = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
