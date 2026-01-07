"""
Metrics Tracking Utilities

Provides structured JSON logging and TensorBoard integration for
tracking RAFT training metrics across cycles.

Features:
- Structured JSON logging with timestamps
- TensorBoard scalar/histogram logging
- Cycle-level metrics aggregation
- Training history persistence

Usage:
    tracker = MetricsTracker(output_dir="models/raft_training")
    
    # Log cycle metrics
    tracker.log_cycle(1, {
        'success_rate': 0.15,
        'avg_reward': 0.08,
        'kept_samples': 100,
        'learning_rate': 5e-5
    })
    
    # Log sample-level data
    tracker.log_samples(1, rewards=[0.5, 0.7, 1.0, 0.0, ...])
    
    # Save summary
    tracker.save_summary()
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@dataclass
class CycleMetrics:
    """Metrics for a single RAFT cycle."""
    cycle: int
    timestamp: str
    duration_seconds: float
    
    # Generation metrics
    total_samples: int = 0
    generation_time_seconds: float = 0.0
    
    # Verification metrics
    verification_time_seconds: float = 0.0
    passed_samples: int = 0
    kept_samples: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    
    # Reward distribution
    reward_0: int = 0      # Failed
    reward_05: int = 0     # Compiled
    reward_07: int = 0     # Ran
    reward_10: int = 0     # Correct
    
    # Training metrics
    training_loss: float = 0.0
    learning_rate: float = 0.0
    train_samples: int = 0
    
    # Model state
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingHistory:
    """Complete training history across all cycles."""
    model_name: str
    start_time: str
    config: Dict = field(default_factory=dict)
    cycles: List[CycleMetrics] = field(default_factory=list)
    final_metrics: Dict = field(default_factory=dict)


class MetricsTracker:
    """
    Tracks and logs metrics for RAFT training.
    
    Provides:
    - Real-time console output (optional Rich tables)
    - JSON log files for each cycle
    - TensorBoard integration
    - Training history persistence
    """
    
    def __init__(
        self,
        output_dir: str,
        model_name: str = "raft_model",
        config: Optional[Dict] = None,
        enable_tensorboard: bool = True,
        enable_json_logs: bool = True,
        console_output: bool = True
    ):
        """
        Initialize metrics tracker.
        
        Args:
            output_dir: Directory for logs and tensorboard
            model_name: Name of the model being trained
            config: Training configuration dict
            enable_tensorboard: Enable TensorBoard logging
            enable_json_logs: Enable JSON file logging
            console_output: Print metrics to console
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.console_output = console_output
        self.enable_json = enable_json_logs
        
        # Initialize history
        self.history = TrainingHistory(
            model_name=model_name,
            start_time=datetime.now().isoformat(),
            config=config or {}
        )
        
        # TensorBoard writer
        self.tb_writer = None
        if enable_tensorboard and HAS_TENSORBOARD:
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
        
        # JSON log file
        self.json_log_path = self.output_dir / "metrics.jsonl"
        
        # Cycle timing
        self._cycle_start_time: Optional[float] = None
        self._generation_start_time: Optional[float] = None
        self._verification_start_time: Optional[float] = None
        
        # Current cycle data
        self._current_cycle: Optional[CycleMetrics] = None
    
    def start_cycle(self, cycle: int):
        """Mark the start of a new cycle."""
        self._cycle_start_time = time.time()
        self._current_cycle = CycleMetrics(
            cycle=cycle,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0.0
        )
    
    def start_generation(self):
        """Mark start of generation phase."""
        self._generation_start_time = time.time()
    
    def end_generation(self, total_samples: int):
        """Mark end of generation phase."""
        if self._generation_start_time and self._current_cycle:
            self._current_cycle.generation_time_seconds = time.time() - self._generation_start_time
            self._current_cycle.total_samples = total_samples
    
    def start_verification(self):
        """Mark start of verification phase."""
        self._verification_start_time = time.time()
    
    def end_verification(
        self,
        rewards: List[float],
        kept_samples: int
    ):
        """
        Mark end of verification phase with results.
        
        Args:
            rewards: List of reward values for all samples
            kept_samples: Number of samples kept for training
        """
        if not self._current_cycle:
            return
        
        if self._verification_start_time:
            self._current_cycle.verification_time_seconds = time.time() - self._verification_start_time
        
        # Calculate reward distribution
        self._current_cycle.reward_0 = sum(1 for r in rewards if r == 0.0)
        self._current_cycle.reward_05 = sum(1 for r in rewards if 0.4 <= r < 0.6)
        self._current_cycle.reward_07 = sum(1 for r in rewards if 0.6 <= r < 0.9)
        self._current_cycle.reward_10 = sum(1 for r in rewards if r >= 0.9)
        
        # Aggregate metrics
        passed = [r for r in rewards if r >= 0.5]
        self._current_cycle.passed_samples = len(passed)
        self._current_cycle.kept_samples = kept_samples
        self._current_cycle.success_rate = len(passed) / len(rewards) if rewards else 0.0
        self._current_cycle.avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    def log_training(
        self,
        loss: float,
        learning_rate: float,
        train_samples: int
    ):
        """Log training metrics."""
        if self._current_cycle:
            self._current_cycle.training_loss = loss
            self._current_cycle.learning_rate = learning_rate
            self._current_cycle.train_samples = train_samples
    
    def end_cycle(self, checkpoint_path: Optional[str] = None):
        """
        End current cycle and persist metrics.
        
        Args:
            checkpoint_path: Path to saved checkpoint
        """
        if not self._current_cycle:
            return
        
        # Calculate total duration
        if self._cycle_start_time:
            self._current_cycle.duration_seconds = time.time() - self._cycle_start_time
        
        self._current_cycle.checkpoint_path = checkpoint_path
        
        # Add to history
        self.history.cycles.append(self._current_cycle)
        
        # Log to outputs
        self._log_cycle_metrics(self._current_cycle)
        
        # Reset current cycle
        self._current_cycle = None
        self._cycle_start_time = None
    
    def log_cycle(self, cycle: int, metrics: Dict[str, Any]):
        """
        Log metrics for a cycle (convenience method).
        
        Args:
            cycle: Cycle number
            metrics: Dictionary of metrics
        """
        cycle_metrics = CycleMetrics(
            cycle=cycle,
            timestamp=datetime.now().isoformat(),
            duration_seconds=metrics.get('duration_seconds', 0.0),
            total_samples=metrics.get('total_samples', 0),
            generation_time_seconds=metrics.get('generation_time', 0.0),
            verification_time_seconds=metrics.get('verification_time', 0.0),
            passed_samples=metrics.get('passed_samples', 0),
            kept_samples=metrics.get('kept_samples', 0),
            success_rate=metrics.get('success_rate', 0.0),
            avg_reward=metrics.get('avg_reward', 0.0),
            training_loss=metrics.get('training_loss', metrics.get('loss', 0.0)),
            learning_rate=metrics.get('learning_rate', 0.0),
            train_samples=metrics.get('train_samples', 0),
            checkpoint_path=metrics.get('checkpoint_path')
        )
        
        self.history.cycles.append(cycle_metrics)
        self._log_cycle_metrics(cycle_metrics)
    
    def _log_cycle_metrics(self, cycle_metrics: CycleMetrics):
        """Log cycle metrics to all outputs."""
        cycle = cycle_metrics.cycle
        
        # Console output
        if self.console_output:
            self._print_cycle_summary(cycle_metrics)
        
        # JSON log
        if self.enable_json:
            with open(self.json_log_path, 'a') as f:
                f.write(json.dumps(asdict(cycle_metrics)) + '\n')
        
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('cycle/success_rate', cycle_metrics.success_rate, cycle)
            self.tb_writer.add_scalar('cycle/avg_reward', cycle_metrics.avg_reward, cycle)
            self.tb_writer.add_scalar('cycle/kept_samples', cycle_metrics.kept_samples, cycle)
            self.tb_writer.add_scalar('cycle/training_loss', cycle_metrics.training_loss, cycle)
            self.tb_writer.add_scalar('cycle/learning_rate', cycle_metrics.learning_rate, cycle)
            self.tb_writer.add_scalar('cycle/duration_minutes', cycle_metrics.duration_seconds / 60, cycle)
            
            # Reward distribution
            self.tb_writer.add_scalars('cycle/reward_distribution', {
                'failed': cycle_metrics.reward_0,
                'compiled': cycle_metrics.reward_05,
                'ran': cycle_metrics.reward_07,
                'correct': cycle_metrics.reward_10,
            }, cycle)
            
            self.tb_writer.flush()
    
    def _print_cycle_summary(self, metrics: CycleMetrics):
        """Print cycle summary to console."""
        print(f"\n{'='*60}")
        print(f"Cycle {metrics.cycle} Summary")
        print(f"{'='*60}")
        print(f"Duration: {metrics.duration_seconds/60:.1f} minutes")
        print(f"Samples: {metrics.total_samples} generated, {metrics.kept_samples} kept")
        print(f"Success rate: {metrics.success_rate*100:.1f}%")
        print(f"Avg reward: {metrics.avg_reward:.4f}")
        print(f"Training loss: {metrics.training_loss:.4f}")
        print(f"Learning rate: {metrics.learning_rate:.2e}")
        print(f"Reward distribution:")
        print(f"  0.0 (failed):  {metrics.reward_0}")
        print(f"  0.5 (compiled): {metrics.reward_05}")
        print(f"  0.7 (runs):     {metrics.reward_07}")
        print(f"  1.0 (correct):  {metrics.reward_10}")
        print(f"{'='*60}\n")
    
    def save_summary(self):
        """Save training summary to JSON."""
        summary_path = self.output_dir / "training_summary.json"
        
        # Calculate final metrics
        if self.history.cycles:
            final_cycle = self.history.cycles[-1]
            self.history.final_metrics = {
                'total_cycles': len(self.history.cycles),
                'final_success_rate': final_cycle.success_rate,
                'final_avg_reward': final_cycle.avg_reward,
                'final_loss': final_cycle.training_loss,
                'total_training_time_minutes': sum(
                    c.duration_seconds for c in self.history.cycles
                ) / 60,
                'checkpoint_path': final_cycle.checkpoint_path
            }
        
        with open(summary_path, 'w') as f:
            json.dump(asdict(self.history), f, indent=2)
        
        if self.console_output:
            print(f"Training summary saved to: {summary_path}")
    
    def get_metrics_dataframe(self):
        """
        Get metrics as a pandas DataFrame.
        
        Returns:
            pandas.DataFrame with cycle metrics
        """
        try:
            import pandas as pd
            return pd.DataFrame([asdict(c) for c in self.history.cycles])
        except ImportError:
            raise ImportError("pandas required for get_metrics_dataframe()")
    
    def close(self):
        """Close tracker and flush all logs."""
        if self.tb_writer:
            self.tb_writer.close()
        self.save_summary()


class TrainingMonitor:
    """
    Real-time training monitor with early stopping detection.
    
    Monitors:
    - Success rate trends
    - Reward improvements
    - Training stability
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_improvement: float = 0.01,
        degradation_threshold: float = 0.1
    ):
        """
        Initialize training monitor.
        
        Args:
            patience: Cycles without improvement before warning
            min_improvement: Minimum improvement to count as progress
            degradation_threshold: Threshold for performance degradation warning
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.degradation_threshold = degradation_threshold
        
        self.best_success_rate = 0.0
        self.best_cycle = 0
        self.cycles_without_improvement = 0
        self.history: List[Dict] = []
    
    def update(self, cycle: int, success_rate: float, avg_reward: float) -> Dict[str, Any]:
        """
        Update monitor with cycle results.
        
        Returns:
            Dict with status and any warnings
        """
        self.history.append({
            'cycle': cycle,
            'success_rate': success_rate,
            'avg_reward': avg_reward
        })
        
        result = {
            'status': 'ok',
            'warnings': [],
            'should_stop': False
        }
        
        # Check for improvement
        if success_rate > self.best_success_rate + self.min_improvement:
            self.best_success_rate = success_rate
            self.best_cycle = cycle
            self.cycles_without_improvement = 0
        else:
            self.cycles_without_improvement += 1
            
            if self.cycles_without_improvement >= self.patience:
                result['warnings'].append(
                    f"No improvement for {self.cycles_without_improvement} cycles. "
                    f"Best was cycle {self.best_cycle} ({self.best_success_rate*100:.1f}%)"
                )
        
        # Check for degradation
        if len(self.history) >= 2:
            prev = self.history[-2]['success_rate']
            if success_rate < prev - self.degradation_threshold:
                result['warnings'].append(
                    f"Performance degradation: {prev*100:.1f}% → {success_rate*100:.1f}%"
                )
                result['status'] = 'degraded'
        
        return result
