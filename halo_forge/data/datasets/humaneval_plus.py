"""
HumanEval+ Dataset Loader

HumanEval+ is an enhanced version of OpenAI's HumanEval benchmark with:
- 80x more test cases per problem
- Better edge case coverage
- Additional challenging inputs

Source: https://huggingface.co/datasets/evalplus/humanevalplus

Usage:
    loader = HumanEvalPlusLoader()
    prompts = loader.load()
    
    # Convert to RLVR format
    loader.export_rlvr("data/humaneval_plus_rlvr.jsonl")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class HumanEvalPlusProblem:
    """A single HumanEval+ problem."""
    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test_cases: List[Dict]
    base_tests: List[Dict]
    plus_tests: List[Dict]  # Extended tests from HumanEval+


class HumanEvalPlusLoader:
    """
    Loader for the HumanEval+ dataset.
    
    HumanEval+ provides significantly more test cases than the original
    HumanEval, making it better for execution-based verification.
    
    Features:
    - 164 Python function completion problems
    - ~80x more test cases per problem
    - Better edge case coverage
    """
    
    HF_DATASET = "evalplus/humanevalplus"
    
    def __init__(self, include_plus_tests: bool = True):
        """
        Initialize the loader.
        
        Args:
            include_plus_tests: Include extended HumanEval+ tests (recommended)
        """
        self.include_plus_tests = include_plus_tests
        self._problems: Optional[List[HumanEvalPlusProblem]] = None
    
    def load(self, cache: bool = True) -> List[HumanEvalPlusProblem]:
        """
        Load HumanEval+ problems.
        
        Args:
            cache: Cache loaded problems
            
        Returns:
            List of HumanEvalPlusProblem objects
        """
        if self._problems is not None and cache:
            return self._problems
        
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"Loading HumanEval+ from {self.HF_DATASET}...")
        dataset = load_dataset(self.HF_DATASET, split="test")
        
        problems = []
        for item in dataset:
            # Extract test cases
            base_tests = self._parse_tests(item.get('base_input', []), item.get('entry_point', ''))
            plus_tests = self._parse_tests(item.get('plus_input', []), item.get('entry_point', ''))
            
            test_cases = base_tests
            if self.include_plus_tests:
                test_cases = base_tests + plus_tests
            
            problem = HumanEvalPlusProblem(
                task_id=item['task_id'],
                prompt=item['prompt'],
                canonical_solution=item['canonical_solution'],
                entry_point=item['entry_point'],
                test_cases=test_cases,
                base_tests=base_tests,
                plus_tests=plus_tests
            )
            problems.append(problem)
        
        print(f"Loaded {len(problems)} problems with {sum(len(p.test_cases) for p in problems)} total test cases")
        
        if cache:
            self._problems = problems
        
        return problems
    
    def _parse_tests(self, inputs: List, entry_point: str) -> List[Dict]:
        """Parse test inputs into test case format."""
        tests = []
        for i, inp in enumerate(inputs):
            # HumanEval+ provides inputs as argument tuples
            tests.append({
                'name': f'test_{i+1}',
                'input': inp,  # Tuple of args
                'entry_point': entry_point
            })
        return tests
    
    def to_rlvr_format(self) -> List[Dict]:
        """
        Convert to RLVR prompt format.
        
        Returns:
            List of dicts with prompt, test_cases, metadata
        """
        problems = self.load()
        
        rlvr_data = []
        for p in problems:
            rlvr_data.append({
                'prompt': p.prompt,
                'category': 'python_function',
                'metadata': {
                    'task_id': p.task_id,
                    'entry_point': p.entry_point,
                    'source': 'humaneval_plus',
                    'num_tests': len(p.test_cases)
                },
                'test_cases': [
                    {
                        'input': tc['input'],
                        'entry_point': tc['entry_point']
                    }
                    for tc in p.test_cases[:20]  # Limit for efficiency
                ],
                'canonical_solution': p.canonical_solution
            })
        
        return rlvr_data
    
    def export_rlvr(self, output_path: str):
        """
        Export to RLVR JSONL format.
        
        Args:
            output_path: Output file path
        """
        data = self.to_rlvr_format()
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Exported {len(data)} problems to {output_path}")
    
    def export_sft(self, output_path: str, template: str = "qwen"):
        """
        Export to SFT training format.
        
        Args:
            output_path: Output file path
            template: Chat template format
        """
        from halo_forge.data.formatters import format_for_training, get_system_prompt
        
        problems = self.load()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            for p in problems:
                text = format_for_training(
                    prompt=p.prompt,
                    response=p.canonical_solution,
                    system_prompt=get_system_prompt('code_python'),
                    template=template,
                    include_thinking=False
                )
                
                f.write(json.dumps({
                    'text': text,
                    'metadata': {
                        'task_id': p.task_id,
                        'source': 'humaneval_plus'
                    }
                }) + '\n')
        
        print(f"Exported {len(problems)} SFT examples to {output_path}")


def humaneval_plus_test_harness(code: str, problem: HumanEvalPlusProblem) -> Dict[str, Any]:
    """
    Create a test harness for HumanEval+ problem.
    
    Args:
        code: Generated solution code
        problem: HumanEval+ problem
        
    Returns:
        Dict with full_code and test assertions
    """
    # Combine prompt + solution
    full_code = problem.prompt + code
    
    # Add test assertions
    test_code = "\n\n# Test cases\n"
    for i, tc in enumerate(problem.test_cases[:10]):  # Limit tests
        args = tc['input']
        # Create assertion (actual expected output not in dataset, 
        # so we call the canonical solution)
        test_code += f"# Test {i+1}\n"
        test_code += f"assert {problem.entry_point}(*{args!r}) == canonical_{problem.entry_point}(*{args!r})\n"
    
    return {
        'full_code': full_code,
        'test_code': test_code,
        'entry_point': problem.entry_point
    }
