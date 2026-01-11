"""
LiveCodeBench Dataset Loader

LiveCodeBench is a contamination-free code generation benchmark with
problems from competitive programming contests (post-2024).

Source: https://huggingface.co/datasets/livecodebench/code_generation

Features:
- New problems from recent contests (no data contamination)
- Multiple difficulty levels
- Test cases for execution verification
- Multiple programming languages

Usage:
    loader = LiveCodeBenchLoader()
    prompts = loader.load()
    
    # Convert to RLVR format
    loader.export_rlvr("data/livecodebench_rlvr.jsonl")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class LiveCodeBenchProblem:
    """A single LiveCodeBench problem."""
    question_id: str
    contest_id: str
    question_title: str
    question_content: str
    difficulty: str
    platform: str
    contest_date: str
    starter_code: Optional[str] = None
    public_tests: List[Dict] = field(default_factory=list)
    private_tests: List[Dict] = field(default_factory=list)
    canonical_solution: Optional[str] = None


class LiveCodeBenchLoader:
    """
    Loader for the LiveCodeBench dataset.
    
    LiveCodeBench focuses on contamination-free evaluation by using
    only problems from contests after the training cutoff date.
    
    Features:
    - Problems from LeetCode, Codeforces, AtCoder
    - Various difficulty levels
    - Public and private test cases
    - Multiple languages (primarily Python/C++)
    """
    
    HF_DATASET = "livecodebench/code_generation_lite"
    
    def __init__(
        self,
        difficulty: Optional[str] = None,
        platform: Optional[str] = None,
        min_date: Optional[str] = None,
        max_problems: int = 500
    ):
        """
        Initialize the loader.
        
        Args:
            difficulty: Filter by difficulty (easy, medium, hard)
            platform: Filter by platform (leetcode, codeforces, atcoder)
            min_date: Minimum contest date (YYYY-MM-DD)
            max_problems: Maximum number of problems to load
        """
        self.difficulty = difficulty
        self.platform = platform
        self.min_date = min_date
        self.max_problems = max_problems
        self._problems: Optional[List[LiveCodeBenchProblem]] = None
    
    def load(self, cache: bool = True) -> List[LiveCodeBenchProblem]:
        """
        Load LiveCodeBench problems.
        
        Args:
            cache: Cache loaded problems
            
        Returns:
            List of LiveCodeBenchProblem objects
        """
        if self._problems is not None and cache:
            return self._problems
        
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"Loading LiveCodeBench from {self.HF_DATASET}...")
        
        try:
            dataset = load_dataset(self.HF_DATASET, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"Note: LiveCodeBench may require authentication or different config")
            # Try alternative config
            try:
                dataset = load_dataset("livecodebench/code_generation", "release_v1", split="test", trust_remote_code=True)
            except:
                raise e
        
        problems = []
        for item in dataset:
            # Apply filters
            if self.difficulty and item.get('difficulty', '').lower() != self.difficulty.lower():
                continue
            
            if self.platform and item.get('platform', '').lower() != self.platform.lower():
                continue
            
            if self.min_date:
                contest_date = item.get('contest_date', '')
                if contest_date and contest_date < self.min_date:
                    continue
            
            # Parse test cases
            public_tests = self._parse_tests(item.get('public_test_cases', []))
            private_tests = self._parse_tests(item.get('private_test_cases', []))
            
            problem = LiveCodeBenchProblem(
                question_id=item.get('question_id', ''),
                contest_id=item.get('contest_id', ''),
                question_title=item.get('question_title', ''),
                question_content=item.get('question_content', ''),
                difficulty=item.get('difficulty', 'unknown'),
                platform=item.get('platform', 'unknown'),
                contest_date=item.get('contest_date', ''),
                starter_code=item.get('starter_code'),
                public_tests=public_tests,
                private_tests=private_tests,
                canonical_solution=item.get('canonical_solution')
            )
            problems.append(problem)
            
            if len(problems) >= self.max_problems:
                break
        
        print(f"Loaded {len(problems)} problems")
        
        if cache:
            self._problems = problems
        
        return problems
    
    def _parse_tests(self, tests: Any) -> List[Dict]:
        """Parse test cases from various formats."""
        if not tests:
            return []
        
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except:
                return []
        
        if isinstance(tests, list):
            parsed = []
            for i, tc in enumerate(tests):
                if isinstance(tc, dict):
                    parsed.append({
                        'name': f'test_{i+1}',
                        'input': tc.get('input', ''),
                        'expected': tc.get('output', tc.get('expected', ''))
                    })
                elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
                    parsed.append({
                        'name': f'test_{i+1}',
                        'input': tc[0],
                        'expected': tc[1]
                    })
            return parsed
        
        return []
    
    def to_rlvr_format(self, language: str = "python") -> List[Dict]:
        """
        Convert to RLVR prompt format.
        
        Args:
            language: Target language for solutions
            
        Returns:
            List of dicts with prompt, test_cases, metadata
        """
        problems = self.load()
        
        rlvr_data = []
        for p in problems:
            # Build prompt
            prompt = p.question_content
            if p.starter_code:
                prompt += f"\n\n```{language}\n{p.starter_code}\n```"
            
            # Combine public and private tests
            all_tests = p.public_tests + p.private_tests
            
            rlvr_data.append({
                'prompt': prompt,
                'category': p.difficulty,
                'metadata': {
                    'question_id': p.question_id,
                    'contest_id': p.contest_id,
                    'title': p.question_title,
                    'platform': p.platform,
                    'contest_date': p.contest_date,
                    'source': 'livecodebench',
                    'num_tests': len(all_tests)
                },
                'test_cases': all_tests[:20],  # Limit for efficiency
                'canonical_solution': p.canonical_solution
            })
        
        return rlvr_data
    
    def export_rlvr(self, output_path: str, language: str = "python"):
        """
        Export to RLVR JSONL format.
        
        Args:
            output_path: Output file path
            language: Target language
        """
        data = self.to_rlvr_format(language)
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Exported {len(data)} problems to {output_path}")
    
    def export_sft(self, output_path: str, language: str = "python", template: str = "qwen"):
        """
        Export to SFT training format.
        
        Args:
            output_path: Output file path
            language: Target language
            template: Chat template format
        """
        from halo_forge.data.formatters import format_for_training, get_system_prompt
        
        problems = self.load()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output, 'w') as f:
            for p in problems:
                if not p.canonical_solution:
                    continue
                
                prompt = p.question_content
                if p.starter_code:
                    prompt += f"\n\n```{language}\n{p.starter_code}\n```"
                
                text = format_for_training(
                    prompt=prompt,
                    response=p.canonical_solution,
                    system_prompt=get_system_prompt('competitive'),
                    template=template,
                    include_thinking=False
                )
                
                f.write(json.dumps({
                    'text': text,
                    'metadata': {
                        'question_id': p.question_id,
                        'difficulty': p.difficulty,
                        'source': 'livecodebench'
                    }
                }) + '\n')
                count += 1
        
        print(f"Exported {count} SFT examples to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        problems = self.load()
        
        from collections import Counter
        
        difficulties = Counter(p.difficulty for p in problems)
        platforms = Counter(p.platform for p in problems)
        
        return {
            'total_problems': len(problems),
            'by_difficulty': dict(difficulties),
            'by_platform': dict(platforms),
            'total_public_tests': sum(len(p.public_tests) for p in problems),
            'total_private_tests': sum(len(p.private_tests) for p in problems),
            'with_solution': sum(1 for p in problems if p.canonical_solution)
        }
