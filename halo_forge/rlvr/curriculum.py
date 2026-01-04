"""
Curriculum Learning for RAFT Training

Implements curriculum strategies that order prompts from easy to hard,
improving training efficiency and sample quality.

Strategies:
- complexity: Sort by prompt complexity (length, keyword density)
- historical: Use historical success rates from previous runs
- adaptive: Adjust difficulty based on current model performance
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class CurriculumStrategy(Enum):
    """Available curriculum strategies."""
    NONE = "none"                    # No curriculum, use prompts as-is
    COMPLEXITY = "complexity"        # Sort by estimated complexity
    HISTORICAL = "historical"        # Sort by historical success rate
    ADAPTIVE = "adaptive"            # Adjust based on current performance
    PROGRESSIVE = "progressive"      # Gradually increase difficulty each cycle


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    strategy: CurriculumStrategy = CurriculumStrategy.NONE
    
    # For progressive curriculum: what fraction of prompts to use per cycle
    # Cycle 1: easiest 20%, Cycle 2: easiest 40%, etc.
    progressive_start: float = 0.2      # Start with easiest 20%
    progressive_increment: float = 0.2  # Add 20% more each cycle
    
    # For adaptive curriculum: performance thresholds
    adaptive_easy_threshold: float = 0.7   # If success > 70%, move to harder
    adaptive_hard_threshold: float = 0.2   # If success < 20%, move to easier
    
    # Historical data path
    historical_stats_path: Optional[str] = None
    
    # Complexity weights
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "length": 0.3,          # Longer prompts = harder
        "keywords": 0.4,        # More technical keywords = harder
        "specificity": 0.3,     # More specific requirements = harder
    })


# Technical keywords that indicate complexity
COMPLEXITY_KEYWORDS = {
    # Low complexity (weight 1)
    "low": ["print", "hello", "sum", "add", "count", "loop", "array", "string"],
    # Medium complexity (weight 2)
    "medium": ["sort", "search", "recursion", "class", "struct", "pointer", 
               "memory", "file", "read", "write", "parse"],
    # High complexity (weight 3)
    "high": ["thread", "async", "mutex", "socket", "network", "encrypt", 
             "syscall", "injection", "hook", "shellcode", "exploit",
             "evasion", "obfuscate", "callback", "vtable"],
}


def estimate_complexity(prompt: str) -> float:
    """
    Estimate prompt complexity on a scale of 0.0 (easy) to 1.0 (hard).
    
    Factors:
    - Prompt length
    - Presence of technical keywords
    - Specificity indicators
    
    Args:
        prompt: The prompt text
        
    Returns:
        Complexity score between 0.0 and 1.0
    """
    prompt_lower = prompt.lower()
    
    # Length score (normalized)
    length_score = min(1.0, len(prompt) / 500)  # 500+ chars = max length score
    
    # Keyword score
    keyword_score = 0.0
    keyword_count = 0
    
    for level, keywords in COMPLEXITY_KEYWORDS.items():
        weight = {"low": 1, "medium": 2, "high": 3}[level]
        for keyword in keywords:
            if keyword in prompt_lower:
                keyword_score += weight
                keyword_count += 1
    
    # Normalize keyword score (max ~10 keywords expected)
    keyword_score = min(1.0, keyword_score / 15)
    
    # Specificity indicators
    specificity_score = 0.0
    specificity_indicators = [
        (r'\d+', 0.1),           # Numbers (specific values)
        (r'"[^"]*"', 0.1),       # Quoted strings
        (r'\b(must|should|exactly|specific)\b', 0.15),
        (r'\b(optimize|efficient|fast)\b', 0.1),
        (r'\b(error|exception|handle)\b', 0.1),
        (r'\b(test|verify|validate)\b', 0.1),
    ]
    
    for pattern, weight in specificity_indicators:
        if re.search(pattern, prompt_lower):
            specificity_score += weight
    
    specificity_score = min(1.0, specificity_score)
    
    # Weighted combination
    complexity = (
        0.3 * length_score +
        0.4 * keyword_score +
        0.3 * specificity_score
    )
    
    return min(1.0, max(0.0, complexity))


def sort_by_complexity(prompts: List[str], reverse: bool = False) -> List[str]:
    """
    Sort prompts by estimated complexity.
    
    Args:
        prompts: List of prompt strings
        reverse: If True, sort hardest first
        
    Returns:
        Sorted list of prompts
    """
    scored = [(p, estimate_complexity(p)) for p in prompts]
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return [p for p, _ in scored]


def sort_by_historical(
    prompts: List[str],
    stats_path: str,
    reverse: bool = False
) -> List[str]:
    """
    Sort prompts by historical success rate.
    
    Args:
        prompts: List of prompt strings
        stats_path: Path to historical statistics JSON
        reverse: If True, sort hardest (lowest success) first
        
    Returns:
        Sorted list of prompts
    """
    stats_file = Path(stats_path)
    
    if not stats_file.exists():
        # Fall back to complexity-based sorting
        return sort_by_complexity(prompts, reverse=reverse)
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Get success rates per prompt (keyed by truncated prompt)
    success_rates = {}
    for entry in stats.get("prompt_stats", []):
        key = entry.get("prompt", "")[:100]  # Truncated key
        success_rates[key] = entry.get("success_rate", 0.5)
    
    def get_rate(prompt: str) -> float:
        key = prompt[:100]
        return success_rates.get(key, 0.5)  # Default to 0.5 if unknown
    
    # Sort by success rate (higher = easier)
    sorted_prompts = sorted(prompts, key=get_rate, reverse=not reverse)
    return sorted_prompts


class CurriculumScheduler:
    """
    Manages curriculum learning across RAFT cycles.
    
    Example:
        scheduler = CurriculumScheduler(
            prompts=all_prompts,
            config=CurriculumConfig(strategy=CurriculumStrategy.PROGRESSIVE)
        )
        
        for cycle in range(1, num_cycles + 1):
            cycle_prompts = scheduler.get_prompts_for_cycle(cycle, num_cycles)
            # Train on cycle_prompts...
            scheduler.update_performance(cycle, success_rate=0.35)
    """
    
    def __init__(
        self,
        prompts: List[str],
        config: Optional[CurriculumConfig] = None
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            prompts: All available training prompts
            config: Curriculum configuration
        """
        self.original_prompts = prompts.copy()
        self.config = config or CurriculumConfig()
        
        # Pre-sort prompts by complexity
        self.sorted_prompts = sort_by_complexity(prompts)
        self.complexity_scores = {p: estimate_complexity(p) for p in prompts}
        
        # Performance tracking for adaptive curriculum
        self.cycle_performance: Dict[int, float] = {}
        self.current_difficulty = 0.5  # Start in the middle
    
    def get_prompts_for_cycle(
        self,
        cycle: int,
        total_cycles: int
    ) -> List[str]:
        """
        Get prompts for a specific cycle based on curriculum strategy.
        
        Args:
            cycle: Current cycle number (1-indexed)
            total_cycles: Total number of cycles
            
        Returns:
            List of prompts for this cycle
        """
        strategy = self.config.strategy
        
        if strategy == CurriculumStrategy.NONE:
            return self.original_prompts
        
        elif strategy == CurriculumStrategy.COMPLEXITY:
            # Always use complexity-sorted order
            return self.sorted_prompts
        
        elif strategy == CurriculumStrategy.PROGRESSIVE:
            # Gradually increase the subset
            start = self.config.progressive_start
            increment = self.config.progressive_increment
            
            # Fraction of prompts to use this cycle
            fraction = min(1.0, start + (cycle - 1) * increment)
            n_prompts = max(1, int(len(self.sorted_prompts) * fraction))
            
            return self.sorted_prompts[:n_prompts]
        
        elif strategy == CurriculumStrategy.ADAPTIVE:
            return self._get_adaptive_prompts(cycle)
        
        elif strategy == CurriculumStrategy.HISTORICAL:
            if self.config.historical_stats_path:
                return sort_by_historical(
                    self.original_prompts,
                    self.config.historical_stats_path
                )
            return self.sorted_prompts
        
        return self.original_prompts
    
    def _get_adaptive_prompts(self, cycle: int) -> List[str]:
        """Get prompts based on adaptive difficulty."""
        if cycle == 1:
            # Start with easier half
            n = len(self.sorted_prompts) // 2
            return self.sorted_prompts[:n]
        
        # Check last cycle's performance
        last_perf = self.cycle_performance.get(cycle - 1, 0.5)
        
        if last_perf > self.config.adaptive_easy_threshold:
            # Doing well, increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.15)
        elif last_perf < self.config.adaptive_hard_threshold:
            # Struggling, decrease difficulty
            self.current_difficulty = max(0.0, self.current_difficulty - 0.15)
        
        # Select prompts around current difficulty level
        target_complexity = self.current_difficulty
        margin = 0.3  # Include prompts within +/- 0.3 of target
        
        selected = [
            p for p in self.sorted_prompts
            if abs(self.complexity_scores[p] - target_complexity) <= margin
        ]
        
        # If too few, expand margin
        if len(selected) < len(self.sorted_prompts) // 4:
            return self.sorted_prompts
        
        return selected
    
    def update_performance(self, cycle: int, success_rate: float):
        """
        Update performance tracking for adaptive curriculum.
        
        Args:
            cycle: Cycle number
            success_rate: Success rate for this cycle (0.0 to 1.0)
        """
        self.cycle_performance[cycle] = success_rate
    
    def get_curriculum_info(self, cycle: int, total_cycles: int) -> Dict:
        """Get information about curriculum state for logging."""
        prompts = self.get_prompts_for_cycle(cycle, total_cycles)
        
        avg_complexity = sum(
            self.complexity_scores.get(p, 0.5) for p in prompts
        ) / len(prompts) if prompts else 0.5
        
        return {
            "strategy": self.config.strategy.value,
            "cycle": cycle,
            "total_prompts": len(self.original_prompts),
            "cycle_prompts": len(prompts),
            "avg_complexity": round(avg_complexity, 3),
            "current_difficulty": round(self.current_difficulty, 3),
        }


def save_prompt_stats(
    output_path: str,
    prompt_results: List[Dict]
):
    """
    Save prompt-level statistics for future curriculum learning.
    
    Args:
        output_path: Path to save stats
        prompt_results: List of {prompt, success_rate, samples, ...}
    """
    stats = {
        "prompt_stats": prompt_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

