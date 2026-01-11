"""
halo-forge UI State Management

Global application state for job tracking and metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Any
from collections import deque
import uuid


JobType = Literal["sft", "raft", "benchmark"]
JobStatus = Literal["pending", "running", "completed", "failed", "stopped"]


@dataclass
class JobState:
    """State for a single training/benchmark job."""
    id: str
    type: JobType
    status: JobStatus
    name: str
    config_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_cycle: int = 0  # RAFT only
    total_cycles: int = 0   # RAFT only
    
    # Metrics
    latest_loss: Optional[float] = None
    latest_lr: Optional[float] = None
    latest_grad_norm: Optional[float] = None
    verification_rate: Optional[float] = None  # RAFT only
    
    # Error handling
    error_message: Optional[str] = None
    
    # Process handle (not serializable)
    process: Any = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.type == "raft" and self.total_cycles > 0:
            return (self.current_cycle / self.total_cycles) * 100
        elif self.total_epochs > 0:
            return (self.current_epoch / self.total_epochs) * 100
        elif self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100
        return 0.0
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def duration_str(self) -> str:
        """Get formatted duration string."""
        dur = self.duration
        if dur is None:
            return "--"
        if dur < 60:
            return f"{dur:.0f}s"
        elif dur < 3600:
            return f"{dur/60:.1f}m"
        else:
            return f"{dur/3600:.1f}h"


@dataclass
class MetricPoint:
    """A single metric data point."""
    step: int
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class AppState:
    """Global application state."""
    
    # Job tracking
    jobs: dict[str, JobState] = field(default_factory=dict)
    active_job_id: Optional[str] = None
    
    # Metrics history (job_id -> metric_name -> deque of points)
    metrics_history: dict[str, dict[str, deque]] = field(default_factory=dict)
    
    # UI state
    sidebar_collapsed: bool = False
    current_page: str = "dashboard"
    
    def create_job(
        self,
        job_type: JobType,
        name: str,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> JobState:
        """Create and register a new job."""
        job_id = str(uuid.uuid4())[:8]
        job = JobState(
            id=job_id,
            type=job_type,
            status="pending",
            name=name,
            config_path=config_path,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job
        self.metrics_history[job_id] = {
            "loss": deque(maxlen=1000),
            "lr": deque(maxlen=1000),
            "grad_norm": deque(maxlen=1000),
        }
        return job
    
    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus):
        """Update job status with timestamp."""
        job = self.jobs.get(job_id)
        if job:
            job.status = status
            if status == "running" and job.started_at is None:
                job.started_at = datetime.now()
            elif status in ("completed", "failed", "stopped"):
                job.completed_at = datetime.now()
    
    def add_metric(self, job_id: str, metric_name: str, step: int, value: float):
        """Add a metric data point for a job."""
        if job_id in self.metrics_history:
            if metric_name not in self.metrics_history[job_id]:
                self.metrics_history[job_id][metric_name] = deque(maxlen=1000)
            self.metrics_history[job_id][metric_name].append(
                MetricPoint(step=step, value=value)
            )
    
    def get_active_jobs(self) -> list[JobState]:
        """Get all currently running jobs."""
        return [j for j in self.jobs.values() if j.status == "running"]
    
    def get_recent_jobs(self, n: int = 10) -> list[JobState]:
        """Get most recent jobs sorted by creation time."""
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )
        return sorted_jobs[:n]
    
    def get_jobs_by_status(self, status: JobStatus) -> list[JobState]:
        """Get all jobs with a specific status."""
        return [j for j in self.jobs.values() if j.status == status]


# Global singleton instance
state = AppState()
