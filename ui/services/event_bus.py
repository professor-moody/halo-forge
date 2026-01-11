"""
Event Bus for Real-Time UI Updates

Provides a pub/sub system for pushing updates from backend services
to UI components without polling.

Usage:
    from ui.services.event_bus import get_event_bus, Event, EventType
    
    # Subscribe to events
    bus = get_event_bus()
    unsub = bus.subscribe(EventType.METRICS_UPDATE, my_handler)
    
    # Emit events
    await bus.emit(Event(
        type=EventType.METRICS_UPDATE,
        job_id="job-123",
        data={'loss': 0.5, 'step': 100}
    ))
    
    # Cleanup
    unsub()  # Unsubscribe when done
"""

from typing import Callable, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class EventType(Enum):
    """Types of events that can be emitted."""
    
    # Job lifecycle events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_STOPPED = "job_stopped"
    
    # Training events
    METRICS_UPDATE = "metrics_update"
    LOG_LINE = "log_line"
    CHECKPOINT_SAVED = "checkpoint_saved"
    
    # Hardware events
    GPU_UPDATE = "gpu_update"


@dataclass
class Event:
    """An event to be emitted through the EventBus."""
    
    type: EventType
    job_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class EventBus:
    """
    Singleton event bus for real-time UI updates.
    
    Allows services to publish events and UI components to subscribe
    to those events for instant updates without polling.
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls) -> 'EventBus':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._initialized = True
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[['Event'], Any]
    ) -> Callable[[], None]:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is emitted.
                     Can be sync or async.
        
        Returns:
            Unsubscribe function - call it to stop receiving events
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        
        self._subscribers[event_type].add(callback)
        
        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe(event_type, callback)
        
        return unsubscribe
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Unsubscribe from events.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: The callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
    
    async def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: The event to emit
        """
        if event.type not in self._subscribers:
            return
        
        # Copy the set to avoid modification during iteration
        callbacks = list(self._subscribers[event.type])
        
        for callback in callbacks:
            try:
                result = callback(event)
                # Support both sync and async callbacks
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # Log error but don't stop other callbacks
                print(f"[EventBus] Handler error for {event.type.value}: {e}")
    
    def emit_sync(self, event: Event) -> None:
        """
        Emit an event synchronously (fire-and-forget for async handlers).
        
        Useful when called from sync code that can't await.
        Creates a task for async callbacks.
        
        Args:
            event: The event to emit
        """
        if event.type not in self._subscribers:
            return
        
        callbacks = list(self._subscribers[event.type])
        
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    # Schedule async callback as task
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop - skip async callbacks
                        pass
            except Exception as e:
                print(f"[EventBus] Handler error for {event.type.value}: {e}")
    
    def subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, set()))
    
    def clear_subscribers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear subscribers.
        
        Args:
            event_type: If provided, clear only for this type.
                       If None, clear all subscribers.
        """
        if event_type:
            self._subscribers[event_type] = set()
        else:
            self._subscribers.clear()


# Singleton accessor
def get_event_bus() -> EventBus:
    """Get the global EventBus instance."""
    return EventBus()
