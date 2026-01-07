"""
ONNX Exporter

Export models to ONNX format for cross-platform inference.
"""

import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from halo_forge.inference.export.base import ModelExporter, ExportConfig


class ONNXExporter(ModelExporter):
    """
    Export models to ONNX format.
    
    ONNX provides cross-platform inference support through various
    runtimes like ONNX Runtime, TensorRT, OpenVINO, etc.
    
    Usage:
        exporter = ONNXExporter()
        exporter.export(model, "model.onnx")
    """
    
    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        opset_version: int = 17
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            config: Export configuration
            opset_version: ONNX opset version (default: 17)
        """
        super().__init__(config)
        self.opset_version = opset_version
    
    @property
    def format_name(self) -> str:
        return "ONNX"
    
    @property
    def file_extension(self) -> str:
        return ".onnx"
    
    def _check_requirements(self) -> bool:
        """Check if ONNX export requirements are met."""
        try:
            import onnx
            from optimum.onnxruntime import ORTModelForCausalLM
            return True
        except ImportError:
            return False
    
    def export(
        self,
        model: Any,
        output_path: str,
        tokenizer: Any = None,
        optimize: bool = True,
        **kwargs
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: HuggingFace model to export
            output_path: Path for output ONNX file or directory
            tokenizer: Tokenizer (optional)
            optimize: Apply ONNX optimizations
            **kwargs: Additional options
            
        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        
        # ONNX export creates a directory, not a single file
        if output_path.suffix == ".onnx":
            output_dir = output_path.parent / output_path.stem
        else:
            output_dir = output_path
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting to ONNX (opset {self.opset_version})...")
        
        # Try optimum export first
        success = self._export_via_optimum(model, output_dir, tokenizer, optimize)
        
        if not success:
            # Fallback to direct torch export
            success = self._export_via_torch(model, output_dir, tokenizer)
        
        if not success:
            raise RuntimeError(
                "ONNX export failed. Please install optimum: "
                "pip install optimum[onnxruntime]"
            )
        
        print(f"Exported to {output_dir}")
        return str(output_dir)
    
    def _export_via_optimum(
        self,
        model: Any,
        output_dir: Path,
        tokenizer: Any,
        optimize: bool
    ) -> bool:
        """Export using Hugging Face Optimum."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            from optimum.exporters.onnx import main_export
        except ImportError:
            print("Optimum not installed. Install with: pip install optimum[onnxruntime]")
            return False
        
        try:
            # Save model to temp directory first
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / "model"
                
                print("Saving model to temporary directory...")
                model.save_pretrained(str(tmp_path))
                if tokenizer:
                    tokenizer.save_pretrained(str(tmp_path))
                
                print("Running ONNX export via Optimum...")
                
                # Use optimum's main export function
                main_export(
                    model_name_or_path=str(tmp_path),
                    output=str(output_dir),
                    task="text-generation",
                    opset=self.opset_version,
                    device="cpu",  # Export on CPU for compatibility
                    fp16=False,
                    optimize=optimize and "O2" or None,
                    trust_remote_code=True
                )
            
            return True
            
        except Exception as e:
            print(f"Optimum export failed: {e}")
            return False
    
    def _export_via_torch(
        self,
        model: Any,
        output_dir: Path,
        tokenizer: Any
    ) -> bool:
        """Export using PyTorch's native ONNX export."""
        try:
            import torch
            import onnx
        except ImportError:
            print("ONNX not installed. Install with: pip install onnx")
            return False
        
        try:
            model = model.cpu().eval()
            
            # Create dummy input
            if tokenizer:
                dummy_input = tokenizer(
                    "Hello, how are you?",
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Fallback dummy input
                batch_size = 1
                seq_length = 32
                dummy_input = {
                    "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
                    "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
                }
            
            output_file = output_dir / "model.onnx"
            
            print("Running PyTorch ONNX export...")
            
            # Export
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(output_file),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"}
                },
                opset_version=self.opset_version,
                do_constant_folding=True
            )
            
            # Verify
            print("Verifying ONNX model...")
            onnx_model = onnx.load(str(output_file))
            onnx.checker.check_model(onnx_model)
            
            # Save tokenizer
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            
            return True
            
        except Exception as e:
            print(f"PyTorch ONNX export failed: {e}")
            return False
    
    @staticmethod
    def optimize_onnx(model_path: str, output_path: Optional[str] = None) -> str:
        """
        Optimize an existing ONNX model.
        
        Args:
            model_path: Path to ONNX model
            output_path: Path for optimized model (overwrites if not specified)
            
        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
            import onnx
        except ImportError:
            raise ImportError("Install onnxruntime: pip install onnxruntime")
        
        output_path = output_path or model_path
        
        print(f"Optimizing {model_path}...")
        
        # Load and optimize
        model = onnx.load(model_path)
        
        optimized_model = optimizer.optimize_model(
            model_path,
            model_type='gpt2',  # Generic transformer type
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=optimizer.FusionOptions("gpt2")
        )
        
        optimized_model.save_model_to_file(output_path)
        
        print(f"Saved optimized model to {output_path}")
        return output_path
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """
        Get information about an ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Dictionary with model information
        """
        import onnx
        
        model = onnx.load(model_path)
        
        return {
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "graph_name": model.graph.name,
            "inputs": [
                {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
                for inp in model.graph.input
            ],
            "outputs": [
                {"name": out.name}
                for out in model.graph.output
            ]
        }
