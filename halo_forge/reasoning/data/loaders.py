"""
Math Dataset Loaders

Load math and reasoning datasets from HuggingFace for RLVR training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class MathSample:
    """A single math problem sample."""
    
    question: str  # Problem statement
    answer: str  # Final answer
    solution: Optional[str] = None  # Step-by-step solution
    difficulty: Optional[str] = None  # Difficulty level
    subject: Optional[str] = None  # Math subject area
    metadata: Dict[str, Any] = field(default_factory=dict)


class MathDataset(ABC):
    """Abstract base class for math datasets."""
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split (train, test)
            cache_dir: Directory for caching
            limit: Limit number of samples
        """
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else Path("~/.cache/halo_forge/math").expanduser()
        self.limit = limit
        self.samples: List[MathSample] = []
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @abstractmethod
    def load(self) -> List[MathSample]:
        """Load dataset samples."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[MathSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> MathSample:
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
                    'prompt': sample.question,
                    'expected_answer': sample.answer,
                    'solution': sample.solution,
                    'difficulty': sample.difficulty,
                    'subject': sample.subject,
                    'metadata': sample.metadata,
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported {len(self.samples)} samples to {output_path}")


class GSM8KLoader(MathDataset):
    """
    Load GSM8K dataset.
    
    GSM8K: Grade School Math 8K
    - 8,500 high-quality linguistically diverse grade school math problems
    - Problems require 2-8 steps to solve
    - Answers are always integers
    """
    
    @property
    def name(self) -> str:
        return "gsm8k"
    
    def load(self) -> List[MathSample]:
        """Load GSM8K samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading GSM8K {self.split}...")
        dataset = load_dataset("gsm8k", "main", split=self.split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            # Extract final answer from solution
            # GSM8K format: "... #### <answer>"
            answer = self._extract_gsm8k_answer(item["answer"])
            
            samples.append(MathSample(
                question=item["question"],
                answer=answer,
                solution=item["answer"],
                difficulty="grade_school",
                subject="arithmetic",
                metadata={
                    "dataset": "gsm8k",
                    "index": i,
                }
            ))
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} GSM8K samples")
        return samples
    
    def _extract_gsm8k_answer(self, solution: str) -> str:
        """Extract final answer from GSM8K solution format."""
        # GSM8K uses "#### <answer>" format
        match = re.search(r"####\s*([^\n]+)", solution)
        if match:
            answer = match.group(1).strip()
            # Remove commas from numbers
            answer = answer.replace(",", "")
            return answer
        
        # Fallback: try to find last number
        numbers = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", solution)
        if numbers:
            return numbers[-1].replace(",", "")
        
        return solution.strip().split()[-1] if solution else ""


class MATHLoader(MathDataset):
    """
    Load MATH dataset.
    
    MATH: Competition Mathematics
    - 12,500 challenging competition math problems
    - 7 subjects: Algebra, Counting & Probability, Geometry, 
      Intermediate Algebra, Number Theory, Prealgebra, Precalculus
    - 5 difficulty levels
    """
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        min_difficulty: int = 1,
        max_difficulty: int = 5,
    ):
        super().__init__(split, cache_dir, limit)
        self.subjects = subjects
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
    
    @property
    def name(self) -> str:
        return "math"
    
    def load(self) -> List[MathSample]:
        """Load MATH samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading MATH {self.split}...")
        
        # MATH dataset from HuggingFace
        dataset = load_dataset("lighteval/MATH", split=self.split, trust_remote_code=True)
        
        samples = []
        count = 0
        for item in dataset:
            # Filter by difficulty
            level = item.get("level", "Level 1")
            difficulty = self._parse_difficulty(level)
            
            if difficulty < self.min_difficulty or difficulty > self.max_difficulty:
                continue
            
            # Filter by subject
            subject = item.get("type", "").lower()
            if self.subjects and not any(s.lower() in subject for s in self.subjects):
                continue
            
            if self.limit and count >= self.limit:
                break
            
            # Extract answer from solution
            answer = self._extract_math_answer(item.get("solution", ""))
            
            samples.append(MathSample(
                question=item["problem"],
                answer=answer,
                solution=item.get("solution", ""),
                difficulty=level,
                subject=item.get("type", ""),
                metadata={
                    "dataset": "math",
                    "level": difficulty,
                }
            ))
            count += 1
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} MATH samples")
        return samples
    
    def _parse_difficulty(self, level_str: str) -> int:
        """Parse difficulty level from string."""
        match = re.search(r"(\d+)", level_str)
        if match:
            return int(match.group(1))
        return 1
    
    def _extract_math_answer(self, solution: str) -> str:
        """Extract final answer from MATH solution format."""
        # MATH uses \\boxed{answer} format
        match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if match:
            return match.group(1).strip()
        
        # Try nested braces
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        match = re.search(pattern, solution)
        if match:
            return match.group(1).strip()
        
        # Fallback: return empty
        return ""


class AIMELoader(MathDataset):
    """
    Load AIME (American Invitational Mathematics Examination) problems.
    
    AIME answers are always integers from 000 to 999.
    """
    
    @property
    def name(self) -> str:
        return "aime"
    
    def load(self) -> List[MathSample]:
        """Load AIME samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading AIME {self.split}...")
        
        # Try to load AIME dataset
        try:
            dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        except Exception as e:
            logger.warning(f"AIME dataset not available: {e}")
            self.samples = []
            return []
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            samples.append(MathSample(
                question=item.get("problem", item.get("question", "")),
                answer=str(item.get("answer", "")),
                solution=item.get("solution", ""),
                difficulty="competition",
                subject="mixed",
                metadata={
                    "dataset": "aime",
                    "year": item.get("year"),
                    "problem_number": item.get("problem_number"),
                }
            ))
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} AIME samples")
        return samples


# Dataset registry
MATH_DATASETS = {
    "gsm8k": GSM8KLoader,
    "math": MATHLoader,
    "aime": AIMELoader,
}


def load_math_dataset(
    name: str,
    split: str = "train",
    limit: Optional[int] = None,
    **kwargs
) -> MathDataset:
    """
    Load a math dataset by name.
    
    Args:
        name: Dataset name (gsm8k, math, aime)
        split: Dataset split
        limit: Limit number of samples
        **kwargs: Additional arguments for dataset
        
    Returns:
        MathDataset instance
    """
    if name not in MATH_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(MATH_DATASETS.keys())}")
    
    loader_cls = MATH_DATASETS[name]
    loader = loader_cls(split=split, limit=limit, **kwargs)
    loader.load()
    
    return loader


def list_math_datasets() -> List[str]:
    """List available math datasets."""
    return list(MATH_DATASETS.keys())
