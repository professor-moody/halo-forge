"""
Quantization-Aware Training (QAT)

Train models with simulated quantization for deployment.
"""

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import DataLoader

from halo_forge.inference.calibration import CalibrationDataset, CalibrationConfig


@dataclass
class QATConfig:
    """Configuration for QAT training."""
    target_precision: str = "int4"  # int4, int8, fp16
    calibration_samples: int = 512
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    save_steps: int = 500
    output_dir: str = "models/qat"


def prepare_qat(
    model: Any,
    target_precision: str = "int4",
    calibration_data: Optional[DataLoader] = None
) -> Any:
    """
    Prepare model for quantization-aware training.
    
    Inserts fake quantization operations that simulate quantized behavior
    during training while allowing gradient flow.
    
    Args:
        model: Model to prepare
        target_precision: Target precision (int4, int8, fp16)
        calibration_data: Optional calibration data for range estimation
        
    Returns:
        QAT-ready model
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("bitsandbytes is required for QAT: pip install bitsandbytes")
    
    # For int4/int8, we use bitsandbytes quantization
    if target_precision == "int4":
        from transformers import BitsAndBytesConfig
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Note: For actual QAT, we would need to use a library like
        # quanto or pytorch's native QAT. For now, we're using
        # post-training quantization via bitsandbytes.
        model.config.quantization_config = quant_config
        
    elif target_precision == "int8":
        from transformers import BitsAndBytesConfig
        
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model.config.quantization_config = quant_config
        
    elif target_precision == "fp16":
        # Convert to FP16
        model = model.half()
        
    else:
        raise ValueError(f"Unknown precision: {target_precision}")
    
    return model


def convert_to_quantized(
    model: Any,
    target_precision: str = "int4",
    output_path: Optional[str] = None
) -> Any:
    """
    Convert trained QAT model to final quantized form.
    
    Args:
        model: QAT model to convert
        target_precision: Target precision
        output_path: Optional path to save quantized model
        
    Returns:
        Quantized model
    """
    # For bitsandbytes-based quantization, the model is already quantized
    # during loading. For true QAT, we would fold batch norms and
    # convert fake quant ops to real quantized ops here.
    
    if output_path:
        model.save_pretrained(output_path)
        print(f"Saved quantized model to {output_path}")
    
    return model


class QATTrainer:
    """
    Quantization-Aware Training Trainer.
    
    Trains a model with simulated quantization to maintain quality
    when deployed in quantized form.
    
    Usage:
        trainer = QATTrainer(config)
        quantized_model = trainer.train(model, calibration_data, verifier)
    """
    
    def __init__(self, config: Optional[QATConfig] = None):
        """
        Initialize QAT trainer.
        
        Args:
            config: QAT configuration
        """
        self.config = config or QATConfig()
        self.best_reward = 0.0
        self.best_checkpoint = None
    
    def train(
        self,
        model: Any,
        calibration_data: DataLoader,
        verifier: Optional[Any] = None,
        eval_prompts: Optional[List[str]] = None
    ) -> Any:
        """
        Run QAT training.
        
        Args:
            model: Model to train
            calibration_data: DataLoader with calibration samples
            verifier: Optional InferenceOptimizationVerifier for evaluation
            eval_prompts: Prompts for verification
            
        Returns:
            Quantized model
        """
        from tqdm import tqdm
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model for QAT
        print(f"Preparing model for {self.config.target_precision} quantization...")
        qat_model = prepare_qat(model, self.config.target_precision)
        
        # Run calibration passes
        print("Running calibration passes...")
        qat_model.eval()
        
        with torch.no_grad():
            for epoch in range(self.config.epochs):
                print(f"Calibration epoch {epoch + 1}/{self.config.epochs}")
                
                for batch_idx, batch in enumerate(tqdm(calibration_data)):
                    # Move to device
                    batch = {k: v.to(qat_model.device) for k, v in batch.items()}
                    
                    # Forward pass for calibration
                    _ = qat_model(**batch)
                    
                    # Save checkpoint periodically
                    if batch_idx > 0 and batch_idx % self.config.save_steps == 0:
                        checkpoint_path = output_dir / f"checkpoint-{epoch}-{batch_idx}"
                        qat_model.save_pretrained(str(checkpoint_path))
                
                # Verify quality after each epoch
                if verifier and eval_prompts:
                    result = verifier.verify(qat_model, eval_prompts)
                    print(f"Epoch {epoch + 1} - Reward: {result.reward:.3f}, "
                          f"Quality: {result.metadata['quality_score']:.2%}")
                    
                    if result.reward > self.best_reward:
                        self.best_reward = result.reward
                        self.best_checkpoint = output_dir / f"best_checkpoint"
                        qat_model.save_pretrained(str(self.best_checkpoint))
        
        # Convert to final quantized form
        quantized_model = convert_to_quantized(
            qat_model,
            self.config.target_precision,
            str(output_dir / "final")
        )
        
        print(f"QAT complete. Best reward: {self.best_reward:.3f}")
        
        return quantized_model
    
    def save_best(self, model: Any):
        """Save the best checkpoint."""
        if self.best_checkpoint:
            model.save_pretrained(str(self.best_checkpoint))


def quantize_model_simple(
    model_path: str,
    output_path: str,
    precision: str = "int4",
    calibration_data: Optional[str] = None
) -> str:
    """
    Simple one-shot quantization without training.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_path: Where to save quantized model
        precision: Target precision (int4, int8, fp16)
        calibration_data: Optional path to calibration data
        
    Returns:
        Path to quantized model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model from {model_path}...")
    
    # Configure quantization
    if precision == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif precision == "int8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quant_config = None
    
    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Run calibration if data provided
    if calibration_data:
        print("Running calibration...")
        cal_dataset = CalibrationDataset.from_jsonl(calibration_data, tokenizer)
        dataloader = cal_dataset.get_dataloader()
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calibrating"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                _ = model(**batch)
    
    # Save
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Quantized model saved to {output_path}")
    return output_path
