"""
Metrics Parser

Parse training metrics from HuggingFace Trainer and RAFT output.
Extracts loss, learning rate, epoch, step, and other metrics from log lines.
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ParsedMetrics:
    """Parsed metrics from a log line."""
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    cycle: Optional[int] = None
    total_cycles: Optional[int] = None
    compile_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    is_checkpoint: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None and v is not False}
    
    def has_metrics(self) -> bool:
        """Check if any metrics were parsed."""
        return bool(self.to_dict())


class MetricsParser:
    """
    Parse training metrics from HuggingFace Trainer output.
    
    Supports multiple log formats:
    - JSON format: {'loss': 2.3456, 'learning_rate': 2e-05, 'epoch': 0.5}
    - Tabular format: Step 100/500 | Loss: 2.3456 | LR: 2e-05
    - RAFT format: Cycle 3/6 | Compile Rate: 85.3%
    """
    
    # HuggingFace Trainer patterns
    PATTERNS = {
        # JSON log format: {'loss': 2.3456, 'learning_rate': 2e-05, 'epoch': 0.5}
        'json_log': re.compile(
            r"\{[^}]*'loss':\s*([\d.]+)[^}]*'learning_rate':\s*([\d.e\-+]+)[^}]*'epoch':\s*([\d.]+)"
        ),
        
        # Alternative JSON: {"loss": 2.3456, "learning_rate": 2e-05}
        'json_log_alt': re.compile(
            r'\{[^}]*"loss":\s*([\d.]+)[^}]*"learning_rate":\s*([\d.e\-+]+)'
        ),
        
        # Step progress: Step 100/500 | Loss: 2.3456
        'step_progress': re.compile(
            r'[Ss]tep\s+(\d+)\s*/\s*(\d+).*?[Ll]oss[:\s]+([\d.]+)'
        ),
        
        # Step with loss only: Step 100 | Loss: 2.3456
        'step_loss': re.compile(
            r'[Ss]tep\s+(\d+).*?[Ll]oss[:\s]+([\d.]+)'
        ),
        
        # Epoch progress: Epoch 1/3
        'epoch_progress': re.compile(
            r'[Ee]poch\s+(\d+)\s*/\s*(\d+)'
        ),
        
        # Epoch float: epoch=0.5 or Epoch: 0.5
        'epoch_float': re.compile(
            r'[Ee]poch[=:\s]+([\d.]+)'
        ),
        
        # RAFT Cycle: Cycle 3/6
        'raft_cycle': re.compile(
            r'[Cc]ycle\s+(\d+)\s*/\s*(\d+)'
        ),
        
        # Compile rate: compile_rate=0.85 or Compile Rate: 85.3%
        'compile_rate_percent': re.compile(
            r'[Cc]ompile\s*[Rr]ate[:\s]+([\d.]+)\s*%'
        ),
        'compile_rate_float': re.compile(
            r'compile_rate[=:\s]+([\d.]+)'
        ),
        
        # Learning rate: lr=2e-05 or LR: 2e-05
        'learning_rate': re.compile(
            r'(?:lr|learning_rate|LR)[=:\s]+([\d.e\-+]+)'
        ),
        
        # Loss standalone: loss=2.3456 or Loss: 2.3456
        'loss_standalone': re.compile(
            r'[Ll]oss[=:\s]+([\d.]+)'
        ),
        
        # Gradient norm: grad_norm=1.234
        'grad_norm': re.compile(
            r'grad_norm[=:\s]+([\d.]+)'
        ),
        
        # Checkpoint detection
        'checkpoint': re.compile(
            r'[Cc]heckpoint.*?(?:saved|saving)|[Ss]aving.*?checkpoint',
            re.IGNORECASE
        ),
        
        # Training loss from HF output
        'training_loss': re.compile(
            r"Training Loss:\s*([\d.]+)"
        ),
        
        # Validation loss
        'eval_loss': re.compile(
            r"(?:Eval|Validation)\s+Loss:\s*([\d.]+)"
        ),
    }
    
    def __init__(self):
        """Initialize the metrics parser."""
        self._last_step = 0
        self._last_epoch = 0.0
    
    def parse_line(self, line: str) -> Optional[ParsedMetrics]:
        """
        Parse a single log line for metrics.
        
        Args:
            line: A single line from training output
            
        Returns:
            ParsedMetrics if any metrics found, None otherwise
        """
        if not line or not line.strip():
            return None
        
        metrics = ParsedMetrics()
        
        # Try JSON log format first (most complete)
        match = self.PATTERNS['json_log'].search(line)
        if match:
            metrics.loss = float(match.group(1))
            metrics.learning_rate = float(match.group(2))
            metrics.epoch = float(match.group(3))
            return metrics
        
        # Try alternative JSON format
        match = self.PATTERNS['json_log_alt'].search(line)
        if match:
            metrics.loss = float(match.group(1))
            metrics.learning_rate = float(match.group(2))
            return metrics
        
        # Try step progress format
        match = self.PATTERNS['step_progress'].search(line)
        if match:
            metrics.step = int(match.group(1))
            metrics.total_steps = int(match.group(2))
            metrics.loss = float(match.group(3))
            self._last_step = metrics.step
            return metrics
        
        # Check for checkpoint
        if self.PATTERNS['checkpoint'].search(line):
            metrics.is_checkpoint = True
            # Continue parsing - there might be other metrics too
        
        # Try RAFT cycle format
        match = self.PATTERNS['raft_cycle'].search(line)
        if match:
            metrics.cycle = int(match.group(1))
            metrics.total_cycles = int(match.group(2))
        
        # Try compile rate
        match = self.PATTERNS['compile_rate_percent'].search(line)
        if match:
            metrics.compile_rate = float(match.group(1)) / 100.0
        else:
            match = self.PATTERNS['compile_rate_float'].search(line)
            if match:
                metrics.compile_rate = float(match.group(1))
        
        # Try epoch
        match = self.PATTERNS['epoch_progress'].search(line)
        if match:
            metrics.epoch = float(match.group(1))
        else:
            match = self.PATTERNS['epoch_float'].search(line)
            if match:
                metrics.epoch = float(match.group(1))
        
        # Try learning rate
        match = self.PATTERNS['learning_rate'].search(line)
        if match:
            try:
                metrics.learning_rate = float(match.group(1))
            except ValueError:
                pass
        
        # Try loss
        match = self.PATTERNS['loss_standalone'].search(line)
        if match and metrics.loss is None:
            metrics.loss = float(match.group(1))
        
        # Try training loss
        match = self.PATTERNS['training_loss'].search(line)
        if match and metrics.loss is None:
            metrics.loss = float(match.group(1))
        
        # Try step (without loss)
        match = self.PATTERNS['step_loss'].search(line)
        if match and metrics.step is None:
            metrics.step = int(match.group(1))
            if metrics.loss is None:
                metrics.loss = float(match.group(2))
            self._last_step = metrics.step
        
        # Try grad norm
        match = self.PATTERNS['grad_norm'].search(line)
        if match:
            metrics.grad_norm = float(match.group(1))
        
        # Return if we found anything
        if metrics.has_metrics():
            return metrics
        
        return None
    
    def parse_lines(self, lines: list[str]) -> list[ParsedMetrics]:
        """
        Parse multiple log lines.
        
        Args:
            lines: List of log lines
            
        Returns:
            List of ParsedMetrics for lines with metrics
        """
        results = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed:
                results.append(parsed)
        return results
    
    def reset(self):
        """Reset parser state."""
        self._last_step = 0
        self._last_epoch = 0.0
