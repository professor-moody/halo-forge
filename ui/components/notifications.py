"""
Centralized notification system for halo-forge UI.

Provides typed, semantic notification helpers for consistent user feedback.
"""

from nicegui import ui
from enum import Enum


class NotificationType(Enum):
    """Notification types mapping to NiceGUI's type values."""
    SUCCESS = "positive"
    ERROR = "negative"
    WARNING = "warning"
    INFO = "info"


def notify(
    message: str,
    type: NotificationType = NotificationType.INFO,
    duration: int = 3000,
):
    """
    Show a toast notification.
    
    Args:
        message: The notification message
        type: NotificationType enum value
        duration: How long to show notification in milliseconds
    """
    ui.notify(
        message,
        type=type.value,
        timeout=duration,
        close_button=True,
        position='top-right',
    )


def notify_job_started(job_name: str):
    """Notify that a training job has started."""
    notify(f"Started: {job_name}", NotificationType.SUCCESS)


def notify_job_completed(job_name: str):
    """Notify that a training job has completed successfully."""
    notify(f"Completed: {job_name}", NotificationType.SUCCESS, duration=5000)


def notify_job_failed(job_name: str, error: str):
    """Notify that a training job has failed."""
    notify(f"Failed: {job_name}\n{error}", NotificationType.ERROR, duration=10000)


def notify_checkpoint_saved(path: str):
    """Notify that a checkpoint has been saved."""
    notify(f"Checkpoint saved: {path}", NotificationType.INFO)


def notify_training_stopped(job_name: str):
    """Notify that training was manually stopped."""
    notify(f"Training stopped: {job_name}", NotificationType.WARNING)


def notify_validation_error(message: str):
    """Notify about a validation error."""
    notify(message, NotificationType.ERROR, duration=5000)


def notify_config_saved(filename: str):
    """Notify that a config file was saved."""
    notify(f"Config saved: {filename}", NotificationType.SUCCESS)


def notify_verification_passed(verifier: str, reward: float):
    """Notify that code verification passed."""
    notify(f"✓ {verifier} passed (reward: {reward:.2f})", NotificationType.SUCCESS)


def notify_verification_failed(verifier: str, reason: str):
    """Notify that code verification failed."""
    notify(f"✗ {verifier} failed: {reason}", NotificationType.ERROR, duration=5000)
