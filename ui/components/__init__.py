"""
halo-forge UI Components

Reusable UI components for the web interface.
"""

from ui.components.sidebar import Sidebar
from ui.components.header import Header
from ui.components.notifications import (
    NotificationType,
    notify,
    notify_job_started,
    notify_job_completed,
    notify_job_failed,
    notify_checkpoint_saved,
    notify_training_stopped,
    notify_validation_error,
    notify_config_saved,
    notify_verification_passed,
    notify_verification_failed,
)

__all__ = [
    "Sidebar",
    "Header",
    # Notifications
    "NotificationType",
    "notify",
    "notify_job_started",
    "notify_job_completed",
    "notify_job_failed",
    "notify_checkpoint_saved",
    "notify_training_stopped",
    "notify_validation_error",
    "notify_config_saved",
    "notify_verification_passed",
    "notify_verification_failed",
]
