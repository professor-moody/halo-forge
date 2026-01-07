"""
Calibration Data Handling

Prepare and manage calibration data for quantization-aware training.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class CalibrationConfig:
    """Configuration for calibration data."""
    num_samples: int = 512
    max_seq_length: int = 2048
    batch_size: int = 4
    seed: int = 42


class CalibrationDataset(Dataset):
    """
    Dataset for model calibration during quantization.
    
    Provides representative samples for determining quantization ranges.
    
    Usage:
        dataset = CalibrationDataset.from_jsonl("data/calibration.jsonl", tokenizer)
        dataloader = dataset.get_dataloader(batch_size=4)
        
        for batch in dataloader:
            model(**batch)  # Calibration pass
    """
    
    def __init__(
        self,
        samples: List[str],
        tokenizer: Any,
        max_length: int = 2048,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize calibration dataset.
        
        Args:
            samples: List of text samples
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            config: Calibration configuration
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config or CalibrationConfig()
        
        # Limit samples
        if len(self.samples) > self.config.num_samples:
            import random
            random.seed(self.config.seed)
            self.samples = random.sample(self.samples, self.config.num_samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }
    
    def get_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """Get DataLoader for calibration."""
        return DataLoader(
            self,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False
        )
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: Any,
        text_field: str = "text",
        config: Optional[CalibrationConfig] = None
    ) -> "CalibrationDataset":
        """
        Load calibration data from JSONL file.
        
        Args:
            path: Path to JSONL file
            tokenizer: Tokenizer for encoding
            text_field: Field containing text
            config: Calibration configuration
            
        Returns:
            CalibrationDataset instance
        """
        samples = []
        
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                text = data.get(text_field, data.get("prompt", ""))
                if text:
                    samples.append(text)
        
        return cls(samples, tokenizer, config=config)
    
    @classmethod
    def from_prompts(
        cls,
        prompts: List[str],
        tokenizer: Any,
        config: Optional[CalibrationConfig] = None
    ) -> "CalibrationDataset":
        """
        Create calibration dataset from prompt list.
        
        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer for encoding
            config: Calibration configuration
            
        Returns:
            CalibrationDataset instance
        """
        return cls(prompts, tokenizer, config=config)
    
    @classmethod
    def generate_synthetic(
        cls,
        tokenizer: Any,
        num_samples: int = 512,
        max_length: int = 2048,
        config: Optional[CalibrationConfig] = None
    ) -> "CalibrationDataset":
        """
        Generate synthetic calibration data.
        
        Uses random sequences from tokenizer's vocabulary.
        
        Args:
            tokenizer: Tokenizer for generation
            num_samples: Number of samples to generate
            max_length: Maximum sequence length
            config: Calibration configuration
            
        Returns:
            CalibrationDataset instance
        """
        import random
        
        config = config or CalibrationConfig(num_samples=num_samples)
        vocab_size = tokenizer.vocab_size
        
        samples = []
        for _ in range(num_samples):
            # Generate random token sequence
            length = random.randint(100, max_length)
            tokens = [random.randint(0, vocab_size - 1) for _ in range(length)]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append(text)
        
        return cls(samples, tokenizer, max_length=max_length, config=config)


def load_calibration_data(
    source: str,
    tokenizer: Any,
    config: Optional[CalibrationConfig] = None
) -> CalibrationDataset:
    """
    Load calibration data from various sources.
    
    Args:
        source: Path to JSONL file, "synthetic", or dataset name
        tokenizer: Tokenizer for encoding
        config: Calibration configuration
        
    Returns:
        CalibrationDataset instance
    """
    if source == "synthetic":
        return CalibrationDataset.generate_synthetic(tokenizer, config=config)
    
    path = Path(source)
    if path.exists() and path.suffix == ".jsonl":
        return CalibrationDataset.from_jsonl(str(path), tokenizer, config=config)
    
    # Try loading from HuggingFace datasets
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(source, split="train")
        text_field = "text" if "text" in dataset.column_names else dataset.column_names[0]
        samples = [item[text_field] for item in dataset]
        
        return CalibrationDataset(samples, tokenizer, config=config)
    except Exception as e:
        raise ValueError(f"Could not load calibration data from '{source}': {e}")
