"""
VLM Dataset Loaders

Load and format vision-language datasets for RLVR training.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
from io import BytesIO
import requests

from PIL import Image


@dataclass
class VLMSample:
    """A single VLM training/evaluation sample."""
    image: Union[Image.Image, str]  # Image or path/URL
    prompt: str
    ground_truth: Optional[str] = None
    alternatives: Optional[List[str]] = None  # Alternative correct answers
    metadata: Optional[Dict[str, Any]] = None
    
    def load_image(self) -> Image.Image:
        """Load image if it's a path or URL."""
        if isinstance(self.image, Image.Image):
            return self.image
        
        if isinstance(self.image, str):
            if self.image.startswith(('http://', 'https://')):
                response = requests.get(self.image)
                return Image.open(BytesIO(response.content))
            else:
                return Image.open(self.image)
        
        raise ValueError(f"Unknown image type: {type(self.image)}")


class VLMDataset(ABC):
    """Abstract base class for VLM datasets."""
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split (train, validation, test)
            cache_dir: Directory for caching
            limit: Limit number of samples
        """
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else Path("~/.cache/halo_forge/vlm").expanduser()
        self.limit = limit
        self.samples: List[VLMSample] = []
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @abstractmethod
    def load(self) -> List[VLMSample]:
        """Load dataset samples."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[VLMSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> VLMSample:
        return self.samples[idx]
    
    def to_rlvr_format(self, output_path: str) -> None:
        """
        Export to RLVR JSONL format.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            for sample in self.samples:
                record = {
                    'prompt': sample.prompt,
                    'image': str(sample.image) if isinstance(sample.image, str) else None,
                    'ground_truth': sample.ground_truth,
                    'metadata': sample.metadata or {}
                }
                f.write(json.dumps(record) + '\n')
    
    def to_sft_format(self, output_path: str, template: str = "qwen") -> None:
        """
        Export to SFT JSONL format.
        
        Args:
            output_path: Output file path
            template: Chat template to use
        """
        with open(output_path, 'w') as f:
            for sample in self.samples:
                if sample.ground_truth:
                    if template == "qwen":
                        text = (
                            f"<|im_start|>user\n{sample.prompt}<|im_end|>\n"
                            f"<|im_start|>assistant\n{sample.ground_truth}<|im_end|>"
                        )
                    else:
                        text = f"### Question\n{sample.prompt}\n\n### Answer\n{sample.ground_truth}"
                    
                    record = {
                        'text': text,
                        'image': str(sample.image) if isinstance(sample.image, str) else None,
                    }
                    f.write(json.dumps(record) + '\n')


class TextVQALoader(VLMDataset):
    """
    TextVQA Dataset Loader
    
    Text reading and reasoning in natural images.
    
    Source: https://huggingface.co/datasets/textvqa
    """
    
    @property
    def name(self) -> str:
        return "textvqa"
    
    def load(self) -> List[VLMSample]:
        """Load TextVQA from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading TextVQA ({self.split})...")
        
        dataset = load_dataset("textvqa", split=self.split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            # TextVQA has multiple answers per question
            answers = item.get('answers', [])
            
            sample = VLMSample(
                image=item['image'],  # PIL Image from HF
                prompt=item['question'],
                ground_truth=answers[0] if answers else None,
                alternatives=answers[1:] if len(answers) > 1 else None,
                metadata={
                    'question_id': item.get('question_id'),
                    'image_id': item.get('image_id'),
                }
            )
            samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} TextVQA samples")
        return samples


class DocVQALoader(VLMDataset):
    """
    DocVQA Dataset Loader
    
    Document Visual Question Answering.
    
    Source: https://huggingface.co/datasets/lmms-lab/DocVQA
    """
    
    @property
    def name(self) -> str:
        return "docvqa"
    
    def load(self) -> List[VLMSample]:
        """Load DocVQA from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading DocVQA ({self.split})...")
        
        # Map split names
        split_map = {
            'train': 'train',
            'validation': 'val',
            'test': 'test'
        }
        hf_split = split_map.get(self.split, self.split)
        
        dataset = load_dataset("lmms-lab/DocVQA", split=hf_split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            sample = VLMSample(
                image=item['image'],
                prompt=item['question'],
                ground_truth=item.get('answers', [None])[0],
                alternatives=item.get('answers', [])[1:],
                metadata={
                    'question_id': item.get('questionId'),
                    'document_id': item.get('ucsf_document_id'),
                }
            )
            samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} DocVQA samples")
        return samples


class ChartQALoader(VLMDataset):
    """
    ChartQA Dataset Loader
    
    Chart understanding and reasoning.
    
    Source: https://huggingface.co/datasets/ahmed-masry/chartqa
    """
    
    @property
    def name(self) -> str:
        return "chartqa"
    
    def load(self) -> List[VLMSample]:
        """Load ChartQA from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading ChartQA ({self.split})...")
        
        dataset = load_dataset("ahmed-masry/chartqa", split=self.split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            sample = VLMSample(
                image=item['image'],
                prompt=item['query'],
                ground_truth=item.get('label'),
                metadata={
                    'type': item.get('type'),  # human or augmented
                }
            )
            samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} ChartQA samples")
        return samples


class RealWorldQALoader(VLMDataset):
    """
    RealWorldQA Dataset Loader
    
    Real-world visual reasoning.
    
    Source: https://huggingface.co/datasets/xai-org/RealworldQA
    """
    
    @property
    def name(self) -> str:
        return "realworldqa"
    
    def load(self) -> List[VLMSample]:
        """Load RealWorldQA from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading RealWorldQA...")
        
        # RealWorldQA only has test split
        dataset = load_dataset("xai-org/RealworldQA", split="test")
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            # Multiple choice format
            question = item['question']
            choices = [item.get(f'choice{j}', '') for j in range(1, 5)]
            choices_text = '\n'.join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices) if c)
            
            full_prompt = f"{question}\n\n{choices_text}"
            
            sample = VLMSample(
                image=item['image'],
                prompt=full_prompt,
                ground_truth=item.get('answer'),
                metadata={
                    'choices': choices,
                    'category': item.get('category'),
                }
            )
            samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} RealWorldQA samples")
        return samples


class MathVistaLoader(VLMDataset):
    """
    MathVista Dataset Loader
    
    Mathematical reasoning with visual context.
    
    Source: https://huggingface.co/datasets/AI4Math/MathVista
    """
    
    @property
    def name(self) -> str:
        return "mathvista"
    
    def load(self) -> List[VLMSample]:
        """Load MathVista from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading MathVista ({self.split})...")
        
        dataset = load_dataset("AI4Math/MathVista", split=self.split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            # Build prompt with choices if multiple choice
            prompt = item['query']
            if item.get('choices'):
                choices_text = '\n'.join(
                    f"{chr(65+j)}. {c}" 
                    for j, c in enumerate(item['choices'])
                )
                prompt = f"{prompt}\n\n{choices_text}"
            
            sample = VLMSample(
                image=item['decoded_image'] if 'decoded_image' in item else item.get('image'),
                prompt=prompt,
                ground_truth=item.get('answer'),
                metadata={
                    'question_type': item.get('question_type'),
                    'grade': item.get('grade'),
                    'source': item.get('source'),
                }
            )
            samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} MathVista samples")
        return samples


# Dataset registry
VLM_DATASETS = {
    'textvqa': TextVQALoader,
    'docvqa': DocVQALoader,
    'chartqa': ChartQALoader,
    'realworldqa': RealWorldQALoader,
    'mathvista': MathVistaLoader,
}


def load_vlm_dataset(
    name: str,
    split: str = "train",
    limit: Optional[int] = None,
    **kwargs
) -> VLMDataset:
    """
    Load a VLM dataset by name.
    
    Args:
        name: Dataset name
        split: Dataset split
        limit: Limit number of samples
        **kwargs: Additional arguments for dataset
        
    Returns:
        VLMDataset instance
    """
    if name not in VLM_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(VLM_DATASETS.keys())}")
    
    loader_cls = VLM_DATASETS[name]
    loader = loader_cls(split=split, limit=limit, **kwargs)
    loader.load()
    
    return loader


def list_vlm_datasets() -> List[str]:
    """List available VLM datasets."""
    return list(VLM_DATASETS.keys())
