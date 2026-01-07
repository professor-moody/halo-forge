"""
GGUF Exporter

Export models to GGUF format for llama.cpp and Ollama.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from halo_forge.inference.export.base import ModelExporter, ExportConfig


# Common GGUF quantization types
GGUF_QUANTIZATIONS = {
    # 4-bit quantizations
    "Q4_0": "Pure 4-bit quantization",
    "Q4_1": "4-bit with scale factor",
    "Q4_K_S": "4-bit K-quant small",
    "Q4_K_M": "4-bit K-quant medium (recommended)",
    # 5-bit quantizations
    "Q5_0": "Pure 5-bit quantization",
    "Q5_1": "5-bit with scale factor",
    "Q5_K_S": "5-bit K-quant small",
    "Q5_K_M": "5-bit K-quant medium",
    # 8-bit quantizations
    "Q8_0": "8-bit quantization (highest quality)",
    # FP16
    "F16": "FP16 (no quantization)",
    "F32": "FP32 (no quantization)",
}


class GGUFExporter(ModelExporter):
    """
    Export models to GGUF format for llama.cpp.
    
    GGUF is the native format for llama.cpp and is widely supported
    by inference engines like Ollama.
    
    Usage:
        exporter = GGUFExporter()
        exporter.export(model, "model.gguf", quantization="Q4_K_M")
    """
    
    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        llama_cpp_path: Optional[str] = None
    ):
        """
        Initialize GGUF exporter.
        
        Args:
            config: Export configuration
            llama_cpp_path: Path to llama.cpp directory (for conversion scripts)
        """
        super().__init__(config)
        
        self.llama_cpp_path = llama_cpp_path
        self._convert_script = None
        self._quantize_script = None
    
    @property
    def format_name(self) -> str:
        return "GGUF"
    
    @property
    def file_extension(self) -> str:
        return ".gguf"
    
    def _find_llama_cpp(self) -> Optional[Path]:
        """Find llama.cpp installation."""
        if self.llama_cpp_path:
            return Path(self.llama_cpp_path)
        
        # Check common locations
        candidates = [
            Path.home() / "llama.cpp",
            Path("/opt/llama.cpp"),
            Path("./llama.cpp"),
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / "convert_hf_to_gguf.py").exists():
                return candidate
        
        return None
    
    def _check_requirements(self) -> bool:
        """Check if GGUF export requirements are met."""
        # Check for llama-cpp-python
        try:
            import llama_cpp
            return True
        except ImportError:
            pass
        
        # Check for llama.cpp scripts
        llama_path = self._find_llama_cpp()
        if llama_path:
            self._convert_script = llama_path / "convert_hf_to_gguf.py"
            self._quantize_script = llama_path / "llama-quantize"
            return self._convert_script.exists()
        
        return False
    
    def export(
        self,
        model: Any,
        output_path: str,
        tokenizer: Any = None,
        quantization: str = "Q4_K_M",
        **kwargs
    ) -> str:
        """
        Export model to GGUF format.
        
        Args:
            model: HuggingFace model to export
            output_path: Path for output GGUF file
            tokenizer: Tokenizer (optional, loaded from model if not provided)
            quantization: Quantization type (Q4_K_M, Q8_0, F16, etc.)
            **kwargs: Additional options
            
        Returns:
            Path to exported GGUF file
        """
        output_path = self.prepare_output_path(output_path)
        
        # Validate quantization type
        if quantization not in GGUF_QUANTIZATIONS:
            raise ValueError(
                f"Unknown quantization: {quantization}. "
                f"Valid options: {list(GGUF_QUANTIZATIONS.keys())}"
            )
        
        print(f"Exporting to GGUF with {quantization} quantization...")
        
        # Save model to temporary HF format first
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_path = Path(tmpdir) / "hf_model"
            
            print("Saving model to temporary HuggingFace format...")
            model.save_pretrained(str(hf_path))
            
            if tokenizer is not None:
                tokenizer.save_pretrained(str(hf_path))
            
            # Try different conversion methods
            success = False
            
            # Method 1: Use llama.cpp convert script
            if self._check_requirements() and self._convert_script:
                success = self._convert_via_script(
                    hf_path, output_path, quantization
                )
            
            # Method 2: Use transformers' GGUF support (if available)
            if not success:
                success = self._convert_via_transformers(
                    hf_path, output_path, quantization
                )
            
            if not success:
                raise RuntimeError(
                    "GGUF export failed. Please install llama.cpp or "
                    "llama-cpp-python: pip install llama-cpp-python"
                )
        
        print(f"Exported to {output_path}")
        return str(output_path)
    
    def _convert_via_script(
        self,
        hf_path: Path,
        output_path: Path,
        quantization: str
    ) -> bool:
        """Convert using llama.cpp scripts."""
        try:
            # Step 1: Convert HF to GGUF (FP16)
            fp16_path = output_path.with_suffix(".fp16.gguf")
            
            print("Converting to GGUF (FP16)...")
            result = subprocess.run(
                [
                    "python3",
                    str(self._convert_script),
                    str(hf_path),
                    "--outfile", str(fp16_path),
                    "--outtype", "f16"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Conversion error: {result.stderr}")
                return False
            
            # Step 2: Quantize if not FP16/FP32
            if quantization not in ("F16", "F32"):
                print(f"Quantizing to {quantization}...")
                
                if self._quantize_script and self._quantize_script.exists():
                    result = subprocess.run(
                        [
                            str(self._quantize_script),
                            str(fp16_path),
                            str(output_path),
                            quantization
                        ],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        print(f"Quantization error: {result.stderr}")
                        return False
                    
                    # Remove intermediate FP16 file
                    fp16_path.unlink(missing_ok=True)
                else:
                    # Just rename FP16 to final output
                    shutil.move(str(fp16_path), str(output_path))
                    print("Note: Quantization skipped (llama-quantize not found)")
            else:
                shutil.move(str(fp16_path), str(output_path))
            
            return True
            
        except Exception as e:
            print(f"Script conversion failed: {e}")
            return False
    
    def _convert_via_transformers(
        self,
        hf_path: Path,
        output_path: Path,
        quantization: str
    ) -> bool:
        """Convert using transformers GGUF support (experimental)."""
        try:
            # This is a placeholder for when transformers adds native GGUF support
            # Currently, this would require manual implementation
            
            print("Direct transformers GGUF export not yet implemented.")
            print("Please install llama.cpp for GGUF export.")
            return False
            
        except Exception as e:
            print(f"Transformers conversion failed: {e}")
            return False
    
    @staticmethod
    def list_quantizations() -> Dict[str, str]:
        """List available quantization types."""
        return GGUF_QUANTIZATIONS.copy()
    
    @staticmethod
    def recommended_quantization(model_size_b: float) -> str:
        """
        Get recommended quantization for model size.
        
        Args:
            model_size_b: Model size in billions of parameters
            
        Returns:
            Recommended quantization type
        """
        if model_size_b <= 3:
            return "Q8_0"  # Can afford higher quality
        elif model_size_b <= 7:
            return "Q4_K_M"  # Good balance
        else:
            return "Q4_K_S"  # Save memory for large models
