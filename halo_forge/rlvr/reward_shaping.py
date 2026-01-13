"""
Iterative Reward Shaping for RAFT Training

Dynamically adjusts reward thresholds and filtering criteria over cycles
to improve training stability and sample quality.

Strategies:
- annealing: Start lenient, gradually increase thresholds
- adaptive: Adjust based on pass rates each cycle
- fixed: Use fixed thresholds (no shaping)
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class RewardShapingStrategy(Enum):
    """Available reward shaping strategies."""
    FIXED = "fixed"          # No shaping, use fixed thresholds
    ANNEALING = "annealing"  # Start lenient, increase thresholds
    ADAPTIVE = "adaptive"    # Adjust based on performance
    WARMUP = "warmup"        # Very lenient for first cycle(s), then normal


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping."""
    
    strategy: RewardShapingStrategy = RewardShapingStrategy.FIXED
    
    # Base thresholds (used as targets for shaping)
    base_reward_threshold: float = 0.5
    base_keep_percent: float = 0.5
    
    # Annealing settings
    annealing_start_threshold: float = 0.1    # Start lenient but require partial credit
    annealing_start_keep_percent: float = 0.8  # Keep more samples initially
    
    # Warmup settings (number of cycles with lenient thresholds)
    warmup_cycles: int = 1
    warmup_threshold: float = 0.1             # Require partial credit even in warmup
    warmup_keep_percent: float = 0.8
    
    # Adaptive settings
    adaptive_min_samples: int = 50       # Minimum samples to keep
    adaptive_target_keep_rate: float = 0.3  # Target for what fraction passes
    adaptive_adjustment_rate: float = 0.1   # How much to adjust per cycle


class RewardShaper:
    """
    Manages reward shaping across RAFT cycles.
    
    Example:
        shaper = RewardShaper(RewardShapingConfig(
            strategy=RewardShapingStrategy.ANNEALING,
            base_reward_threshold=0.5
        ))
        
        for cycle in range(1, num_cycles + 1):
            threshold, keep_pct = shaper.get_thresholds(cycle, num_cycles)
            # Use threshold and keep_pct for filtering...
            shaper.update_stats(cycle, pass_rate=0.35, samples_kept=120)
    """
    
    def __init__(self, config: Optional[RewardShapingConfig] = None):
        self.config = config or RewardShapingConfig()
        self.cycle_stats: Dict[int, Dict] = {}
        
        # For adaptive shaping
        self._current_threshold = self.config.base_reward_threshold
        self._current_keep_percent = self.config.base_keep_percent
    
    def get_thresholds(
        self,
        cycle: int,
        total_cycles: int
    ) -> tuple[float, float]:
        """
        Get reward threshold and keep percent for a cycle.
        
        Args:
            cycle: Current cycle number (1-indexed)
            total_cycles: Total number of cycles
            
        Returns:
            (reward_threshold, keep_percent) tuple
        """
        strategy = self.config.strategy
        
        if strategy == RewardShapingStrategy.FIXED:
            return (
                self.config.base_reward_threshold,
                self.config.base_keep_percent
            )
        
        elif strategy == RewardShapingStrategy.WARMUP:
            if cycle <= self.config.warmup_cycles:
                return (
                    self.config.warmup_threshold,
                    self.config.warmup_keep_percent
                )
            return (
                self.config.base_reward_threshold,
                self.config.base_keep_percent
            )
        
        elif strategy == RewardShapingStrategy.ANNEALING:
            # Linear interpolation from lenient to strict
            if total_cycles <= 1:
                progress = 1.0
            else:
                progress = (cycle - 1) / (total_cycles - 1)
            
            threshold = (
                self.config.annealing_start_threshold +
                progress * (self.config.base_reward_threshold - self.config.annealing_start_threshold)
            )
            
            keep_percent = (
                self.config.annealing_start_keep_percent +
                progress * (self.config.base_keep_percent - self.config.annealing_start_keep_percent)
            )
            
            return (threshold, keep_percent)
        
        elif strategy == RewardShapingStrategy.ADAPTIVE:
            return self._get_adaptive_thresholds(cycle)
        
        return (
            self.config.base_reward_threshold,
            self.config.base_keep_percent
        )
    
    def _get_adaptive_thresholds(self, cycle: int) -> tuple[float, float]:
        """Get thresholds using adaptive strategy."""
        if cycle == 1:
            # Start lenient but require partial credit - also update internal state
            self._current_threshold = 0.1
            self._current_keep_percent = 0.8
            return (self._current_threshold, self._current_keep_percent)
        
        # Check last cycle's stats
        last_stats = self.cycle_stats.get(cycle - 1, {})
        pass_rate = last_stats.get('pass_rate', 0.5)
        samples_kept = last_stats.get('samples_kept', 0)
        
        target_rate = self.config.adaptive_target_keep_rate
        adjustment = self.config.adaptive_adjustment_rate
        
        # If keeping too few samples, lower threshold (floor at 0.1)
        if samples_kept < self.config.adaptive_min_samples or pass_rate < target_rate * 0.5:
            self._current_threshold = max(0.1, self._current_threshold - adjustment)
            self._current_keep_percent = min(1.0, self._current_keep_percent + adjustment)
        
        # If keeping too many (high pass rate), raise threshold
        elif pass_rate > target_rate * 1.5:
            self._current_threshold = min(1.0, self._current_threshold + adjustment)
            self._current_keep_percent = max(0.2, self._current_keep_percent - adjustment * 0.5)
        
        return (self._current_threshold, self._current_keep_percent)
    
    def update_stats(
        self,
        cycle: int,
        pass_rate: float,
        samples_kept: int,
        total_samples: int = 0
    ):
        """
        Update statistics for a cycle.
        
        Args:
            cycle: Cycle number
            pass_rate: Fraction of samples that passed threshold
            samples_kept: Number of samples kept after filtering
            total_samples: Total samples before filtering
        """
        self.cycle_stats[cycle] = {
            'pass_rate': pass_rate,
            'samples_kept': samples_kept,
            'total_samples': total_samples,
        }
        
        # Warn if adaptive strategy can't reach target sample count
        if self.config.strategy == RewardShapingStrategy.ADAPTIVE:
            if samples_kept < self.config.adaptive_min_samples:
                import warnings
                warnings.warn(
                    f"Cycle {cycle}: Only {samples_kept} samples kept "
                    f"(target: {self.config.adaptive_min_samples}). "
                    f"Threshold will be lowered next cycle."
                )
    
    def get_shaping_info(self, cycle: int, total_cycles: int) -> Dict:
        """Get information about current shaping state for logging."""
        threshold, keep_pct = self.get_thresholds(cycle, total_cycles)
        
        return {
            "strategy": self.config.strategy.value,
            "cycle": cycle,
            "reward_threshold": round(threshold, 3),
            "keep_percent": round(keep_pct, 3),
            "is_warmup": (
                self.config.strategy == RewardShapingStrategy.WARMUP and
                cycle <= self.config.warmup_cycles
            ),
        }


def apply_graduated_filtering(
    samples: List[Dict],
    threshold: float,
    keep_percent: float,
    min_samples: int = 1
) -> List[Dict]:
    """
    Apply graduated filtering with reward shaping.
    
    Args:
        samples: List of sample dicts with 'reward' key
        threshold: Minimum reward threshold
        keep_percent: Keep top X% of samples above threshold
        min_samples: Minimum samples to keep (overrides other settings)
        
    Returns:
        Filtered list of samples
    """
    # Filter by threshold
    above_threshold = [s for s in samples if s.get('reward', 0) >= threshold]
    
    # Sort by reward (highest first)
    above_threshold.sort(key=lambda x: x.get('reward', 0), reverse=True)
    
    # Keep top percent
    keep_count = max(min_samples, int(len(above_threshold) * keep_percent))
    
    # If still below minimum and we have samples, take what we have
    if len(above_threshold) < keep_count and above_threshold:
        return above_threshold
    
    return above_threshold[:keep_count]

