"""
Public Dataset Preparation

Download and format public code datasets from HuggingFace.
Supports CodeForces, MBPP, HumanEval, and custom datasets.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

from halo_forge.data.formatters import format_for_training, get_system_prompt


@dataclass
class DatasetSpec:
    """Specification for a public dataset."""
    
    name: str                          # Identifier
    hf_path: str                       # HuggingFace dataset path
    hf_split: str = "train"            # Dataset split
    prompt_field: str = "problem"      # Field containing the prompt
    response_field: str = "solution"   # Field containing the response
    reasoning_field: Optional[str] = None  # Optional reasoning/thinking field
    language_field: Optional[str] = None   # Field for language filtering
    language_filter: Optional[str] = None  # Language to filter for
    difficulty_field: Optional[str] = None # Difficulty metadata field
    tags_field: Optional[str] = None       # Tags/topics field
    max_examples: int = 5000           # Maximum examples to extract
    system_prompt: str = "You are an expert programmer."
    include_reasoning: bool = True     # Include reasoning in <think> tags
    
    # Optional preprocessing function
    preprocess: Optional[Callable[[Dict], Dict]] = None


# Built-in dataset specifications
DATASET_REGISTRY: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec):
    """Register a dataset specification."""
    DATASET_REGISTRY[spec.name] = spec
    return spec


# CodeForces C++
register_dataset(DatasetSpec(
    name="codeforces_cpp",
    hf_path="open-r1/codeforces-cots",
    hf_split="train",
    prompt_field="problem",
    response_field="solution",
    reasoning_field="reasoning",
    language_field="language",
    language_filter="cpp",
    difficulty_field="difficulty",
    tags_field="topics",
    max_examples=4000,
    system_prompt=get_system_prompt("competitive"),
    include_reasoning=True
))

# CodeForces Python
register_dataset(DatasetSpec(
    name="codeforces_python",
    hf_path="open-r1/codeforces-cots",
    hf_split="train",
    prompt_field="problem",
    response_field="solution",
    reasoning_field="reasoning",
    language_field="language",
    language_filter="python",
    max_examples=1000,
    system_prompt=get_system_prompt("competitive"),
    include_reasoning=True
))

# CodeForces Rust
register_dataset(DatasetSpec(
    name="codeforces_rust",
    hf_path="open-r1/codeforces-cots",
    hf_split="train",
    prompt_field="problem",
    response_field="solution",
    reasoning_field="reasoning",
    language_field="language",
    language_filter="rust",
    max_examples=500,
    system_prompt=get_system_prompt("code_rust"),
    include_reasoning=True
))

# MBPP (Google's Python benchmark)
register_dataset(DatasetSpec(
    name="mbpp",
    hf_path="google-research-datasets/mbpp",
    hf_split="train",
    prompt_field="text",
    response_field="code",
    max_examples=500,
    system_prompt=get_system_prompt("code_python"),
    include_reasoning=False
))

# HumanEval (OpenAI's benchmark)
register_dataset(DatasetSpec(
    name="humaneval",
    hf_path="openai/openai_humaneval",
    hf_split="test",
    prompt_field="prompt",
    response_field="canonical_solution",
    max_examples=164,
    system_prompt=get_system_prompt("code_python"),
    include_reasoning=False
))

# HumanEval+ (Extended test cases)
register_dataset(DatasetSpec(
    name="humaneval_plus",
    hf_path="evalplus/humanevalplus",
    hf_split="test",
    prompt_field="prompt",
    response_field="canonical_solution",
    max_examples=164,
    system_prompt=get_system_prompt("code_python"),
    include_reasoning=False
))

# LiveCodeBench (Contamination-free benchmark)
register_dataset(DatasetSpec(
    name="livecodebench",
    hf_path="livecodebench/code_generation_lite",
    hf_split="test",
    prompt_field="question_content",
    response_field="canonical_solution",
    difficulty_field="difficulty",
    max_examples=500,
    system_prompt=get_system_prompt("competitive"),
    include_reasoning=False
))


class DatasetPreparer:
    """
    Prepare training data from public datasets.
    
    Example:
        spec = DATASET_REGISTRY["codeforces_cpp"]
        preparer = DatasetPreparer(spec)
        preparer.prepare("data/codeforces_cpp.jsonl", template="qwen")
    """
    
    def __init__(self, spec: DatasetSpec, system_prompt: Optional[str] = None):
        """
        Initialize dataset preparer.
        
        Args:
            spec: Dataset specification
            system_prompt: Override system prompt
        """
        self.spec = spec
        self.system_prompt = system_prompt or spec.system_prompt
    
    def load_and_filter(self) -> List[Dict]:
        """Load dataset and apply filters."""
        print(f"Loading {self.spec.hf_path}...")
        
        try:
            dataset = load_dataset(self.spec.hf_path, split=self.spec.hf_split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        print(f"Total examples in dataset: {len(dataset)}")
        
        # Convert to list for filtering
        examples = list(dataset)
        
        # Apply language filter if specified
        if self.spec.language_field and self.spec.language_filter:
            examples = [
                ex for ex in examples
                if ex.get(self.spec.language_field, '').lower() == self.spec.language_filter.lower()
            ]
            print(f"After language filter ({self.spec.language_filter}): {len(examples)}")
        
        # Apply preprocessing if specified
        if self.spec.preprocess:
            examples = [self.spec.preprocess(ex) for ex in examples]
            examples = [ex for ex in examples if ex is not None]
        
        # Sample if we have more than max
        if len(examples) > self.spec.max_examples:
            random.seed(42)
            examples = random.sample(examples, self.spec.max_examples)
            print(f"Sampled to: {len(examples)}")
        
        return examples
    
    def format_example(self, example: Dict, template: str = "qwen") -> Dict:
        """Format a single example for training."""
        prompt = example.get(self.spec.prompt_field, '')
        response = example.get(self.spec.response_field, '')
        reasoning = None
        
        if self.spec.include_reasoning and self.spec.reasoning_field:
            reasoning = example.get(self.spec.reasoning_field)
        
        text = format_for_training(
            prompt=prompt,
            response=response,
            system_prompt=self.system_prompt,
            template=template,
            include_thinking=self.spec.include_reasoning,
            thinking=reasoning
        )
        
        # Build metadata
        metadata = {"source": self.spec.name}
        
        if self.spec.difficulty_field:
            metadata["difficulty"] = example.get(self.spec.difficulty_field)
        
        if self.spec.tags_field:
            metadata["tags"] = example.get(self.spec.tags_field)
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def analyze(self, examples: List[Dict]):
        """Print dataset statistics."""
        print(f"\nDataset Analysis: {self.spec.name}")
        print("=" * 50)
        print(f"Total examples: {len(examples)}")
        
        if self.spec.language_field:
            languages = Counter(ex.get(self.spec.language_field, 'unknown') for ex in examples)
            print(f"\nLanguages:")
            for lang, count in languages.most_common(10):
                print(f"  {lang}: {count}")
        
        if self.spec.difficulty_field:
            difficulties = Counter(ex.get(self.spec.difficulty_field, 'unknown') for ex in examples)
            print(f"\nDifficulties:")
            for diff, count in difficulties.most_common():
                print(f"  {diff}: {count}")
    
    def prepare(
        self,
        output_path: str,
        template: str = "qwen",
        analyze: bool = True
    ) -> List[Dict]:
        """
        Full pipeline: load, filter, format, save.
        
        Args:
            output_path: Output JSONL file path
            template: Chat template format
            analyze: Print dataset statistics
            
        Returns:
            List of formatted examples
        """
        # Load and filter
        examples = self.load_and_filter()
        
        if analyze:
            self.analyze(examples)
        
        # Format all examples
        print(f"\nFormatting with {template} template...")
        formatted = []
        
        for ex in tqdm(examples, desc="Formatting"):
            try:
                formatted.append(self.format_example(ex, template))
            except Exception as e:
                continue
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for ex in formatted:
                f.write(json.dumps(ex) + '\n')
        
        print(f"\nSaved {len(formatted)} examples to {output_path}")
        return formatted


def combine_datasets(output_path: str, *input_paths: str, shuffle: bool = True):
    """
    Combine multiple JSONL files into one.
    
    Args:
        output_path: Output file path
        input_paths: Input file paths
        shuffle: Shuffle combined examples
    """
    all_examples = []
    
    for path in input_paths:
        with open(path) as f:
            examples = [json.loads(line) for line in f]
            all_examples.extend(examples)
            print(f"Added {len(examples)} from {path}")
    
    if shuffle:
        random.shuffle(all_examples)
    
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\nCombined {len(all_examples)} examples to {output_path}")


def list_datasets() -> List[str]:
    """List available dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Get a dataset specification by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return DATASET_REGISTRY[name]

