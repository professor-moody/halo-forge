#!/usr/bin/env python3
"""
GGUF Export Script

Convert HuggingFace models to GGUF format for llama.cpp and Ollama.

Usage:
    python scripts/export_gguf.py --model models/trained --output model.gguf
    python scripts/export_gguf.py --model models/trained --output model.gguf --quantization Q4_K_M

Requirements:
    pip install llama-cpp-python
    
    For ROCm/AMD GPU acceleration:
    CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
    
    For CUDA GPU acceleration:
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Available quantization types
QUANTIZATIONS = {
    # 4-bit (smallest, some quality loss)
    "Q4_0": "Pure 4-bit quantization",
    "Q4_1": "4-bit with scale factor",
    "Q4_K_S": "4-bit K-quant small",
    "Q4_K_M": "4-bit K-quant medium (RECOMMENDED)",
    
    # 5-bit (balanced)
    "Q5_0": "Pure 5-bit quantization",
    "Q5_1": "5-bit with scale factor",
    "Q5_K_S": "5-bit K-quant small",
    "Q5_K_M": "5-bit K-quant medium",
    
    # 8-bit (highest quality quantized)
    "Q8_0": "8-bit quantization (high quality)",
    
    # No quantization
    "F16": "FP16 (no quantization, larger file)",
    "F32": "FP32 (no quantization, largest file)",
}


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check for transformers
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    # Check for llama-cpp-python (optional but recommended)
    has_llama_cpp = False
    try:
        import llama_cpp
        has_llama_cpp = True
    except ImportError:
        pass
    
    # Check for llama.cpp convert script
    llama_cpp_path = find_llama_cpp()
    
    if not has_llama_cpp and not llama_cpp_path:
        print("\n⚠️  Neither llama-cpp-python nor llama.cpp found!")
        print("\nInstall one of the following:")
        print("\n  Option 1: pip install (easiest)")
        print("    pip install llama-cpp-python")
        print("\n  Option 2: With AMD GPU support")
        print("    CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" pip install llama-cpp-python --force-reinstall")
        print("\n  Option 3: Clone llama.cpp (most flexible)")
        print("    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("    cd ~/llama.cpp && make")
        sys.exit(1)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    return has_llama_cpp, llama_cpp_path


def find_llama_cpp():
    """Find llama.cpp installation."""
    candidates = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
        Path(os.environ.get("LLAMA_CPP_PATH", "")),
    ]
    
    for path in candidates:
        if path.exists() and (path / "convert_hf_to_gguf.py").exists():
            return path
    
    return None


def convert_via_llama_cpp(model_path: Path, output_path: Path, quantization: str, llama_cpp_path: Path):
    """Convert using llama.cpp scripts."""
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    quantize_binary = llama_cpp_path / "llama-quantize"
    
    # Create temp path for FP16 intermediate
    fp16_path = output_path.with_suffix(".fp16.gguf")
    
    print(f"\n📦 Converting {model_path} to GGUF...")
    print(f"   Quantization: {quantization}")
    print(f"   Output: {output_path}")
    
    # Step 1: Convert to FP16 GGUF
    print("\n🔄 Step 1/2: Converting to GGUF (FP16)...")
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(fp16_path),
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        sys.exit(1)
    
    # Step 2: Quantize (if not F16/F32)
    if quantization in ("F16", "F32"):
        shutil.move(str(fp16_path), str(output_path))
        print(f"\n✅ Saved to {output_path}")
    else:
        print(f"\n🔄 Step 2/2: Quantizing to {quantization}...")
        
        if not quantize_binary.exists():
            print(f"⚠️  llama-quantize not found. Build with: cd {llama_cpp_path} && make")
            print(f"   Keeping FP16 version at {fp16_path}")
            shutil.move(str(fp16_path), str(output_path))
        else:
            cmd = [
                str(quantize_binary),
                str(fp16_path),
                str(output_path),
                quantization
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Quantization failed: {result.stderr}")
                sys.exit(1)
            
            # Clean up FP16 intermediate
            fp16_path.unlink(missing_ok=True)
            print(f"\n✅ Saved to {output_path}")
    
    # Show file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")


def convert_via_transformers(model_path: Path, output_path: Path, quantization: str):
    """Convert using transformers + llama-cpp-python."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n📦 Converting {model_path} to GGUF...")
    print(f"   Quantization: {quantization}")
    print(f"   Output: {output_path}")
    
    # This method requires saving to HF format first, then using llama.cpp
    # For now, we'll save and provide instructions
    
    print("\n🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    
    # Save to temp directory in HF format
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_path = Path(tmpdir) / "hf_model"
        
        print("🔄 Saving in HuggingFace format...")
        model.save_pretrained(str(hf_path))
        tokenizer.save_pretrained(str(hf_path))
        
        # Try to find and use llama.cpp
        llama_cpp_path = find_llama_cpp()
        if llama_cpp_path:
            convert_via_llama_cpp(hf_path, output_path, quantization, llama_cpp_path)
        else:
            # Provide manual instructions
            print("\n⚠️  llama.cpp not found for quantization.")
            print("\nTo complete the conversion:")
            print(f"  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
            print(f"  2. Convert: python llama.cpp/convert_hf_to_gguf.py {hf_path} --outfile {output_path}")
            print(f"  3. Quantize: ./llama.cpp/llama-quantize {output_path} {output_path.with_suffix(f'.{quantization}.gguf')} {quantization}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with recommended quantization
  python scripts/export_gguf.py --model models/trained --output model.gguf

  # Specific quantization
  python scripts/export_gguf.py --model models/trained --output model.gguf -q Q8_0

  # List available quantization types
  python scripts/export_gguf.py --list-quantizations

  # Use with Ollama after conversion
  ollama create mymodel -f Modelfile
        """
    )
    
    parser.add_argument("--model", "-m", help="Path to HuggingFace model or model ID")
    parser.add_argument("--output", "-o", help="Output GGUF file path")
    parser.add_argument("--quantization", "-q", default="Q4_K_M",
                       choices=list(QUANTIZATIONS.keys()),
                       help="Quantization type (default: Q4_K_M)")
    parser.add_argument("--list-quantizations", action="store_true",
                       help="List available quantization types")
    parser.add_argument("--llama-cpp-path", help="Path to llama.cpp directory")
    
    args = parser.parse_args()
    
    # List quantizations
    if args.list_quantizations:
        print("\nAvailable Quantization Types:")
        print("=" * 60)
        for quant, desc in QUANTIZATIONS.items():
            rec = " ⭐" if "RECOMMENDED" in desc else ""
            print(f"  {quant:10} - {desc}{rec}")
        print("\nSmaller = faster inference, less memory, lower quality")
        print("Larger = slower inference, more memory, higher quality")
        return
    
    # Validate args
    if not args.model:
        parser.error("--model is required")
    
    if not args.output:
        # Default output name
        model_name = Path(args.model).name
        args.output = f"{model_name}.{args.quantization}.gguf"
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    # Check model exists
    if not model_path.exists():
        # Might be a HuggingFace model ID
        print(f"ℹ️  Model path not found locally, treating as HuggingFace model ID")
    
    # Check dependencies
    has_llama_cpp, llama_cpp_path = check_dependencies()
    
    if args.llama_cpp_path:
        llama_cpp_path = Path(args.llama_cpp_path)
    
    # Convert
    if llama_cpp_path:
        convert_via_llama_cpp(model_path, output_path, args.quantization, llama_cpp_path)
    else:
        convert_via_transformers(model_path, output_path, args.quantization)
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"\n1. Test with llama.cpp:")
    print(f"   ./llama.cpp/llama-cli -m {output_path} -p \"Hello, world!\"")
    print(f"\n2. Use with Ollama:")
    print(f"   Create a Modelfile:")
    print(f"   ---")
    print(f"   FROM {output_path}")
    print(f"   SYSTEM \"You are a helpful assistant.\"")
    print(f"   ---")
    print(f"   ollama create mymodel -f Modelfile")
    print(f"   ollama run mymodel")


if __name__ == "__main__":
    main()
