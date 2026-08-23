"""A serial, JSON-persisted background job abstraction for local Dataset Lab work."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

try:  # POSIX is the supported local-workstation runtime.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps thread safety.
    fcntl = None  # type: ignore[assignment]

from .errors import JobError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}


@dataclass
class DatasetJob:
    id: str
    kind: str
    payload: Dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    processed: int = 0
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    output_size_bytes: int = 0
    logs: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    retry_of: Optional[str] = None
    work_item_id: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetJob":
        known = cls.__dataclass_fields__
        return cls(**{key: copy.deepcopy(item) for key, item in value.items() if key in known})


class JobContext:
    def __init__(self, manager: "SerialJobManager", job_id: str):
        self._manager = manager
        self.job_id = job_id

    @property
    def checkpoint_data(self) -> Dict[str, Any]:
        return self._manager.get(self.job_id).checkpoint

    def cancelled(self) -> bool:
        return self._manager.get(self.job_id).cancel_requested

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise JobCancelled("job cancellation requested")

    def progress(
        self,
        *,
        stage: Optional[str] = None,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        accepted: Optional[int] = None,
        rejected: Optional[int] = None,
        output_size_bytes: Optional[int] = None,
    ) -> None:
        self._manager._update(
            self.job_id,
            **{
                key: value
                for key, value in locals().items()
                if key not in {"self"} and value is not None
            },
        )

    def log(self, message: str) -> None:
        self._manager._log(self.job_id, message)

    def checkpoint(self, **values: Any) -> None:
        self._manager._checkpoint(self.job_id, values)


class JobCancelled(Exception):
    pass


JobHandler = Callable[[JobContext, Dict[str, Any]], Any]


class SerialJobManager:
    """Run at most one data job while persisting visible state to JSON.

    Handlers are deliberately registered by kind after construction: persisted jobs
    remain serializable and an application restart can re-register handlers before retry.
    """

    def __init__(self, state_path: Path | str, *, max_workers: int = 1, recover: bool = True):
        if max_workers != 1:
            raise JobError("Dataset Lab v1 supports exactly one active data job")
        self.state_path = Path(state_path).expanduser().resolve()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.state_path.with_name(f".{self.state_path.name}.lock")
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-lab")
        self._handlers: Dict[str, JobHandler] = {}
        self._futures: Dict[str, Future[Any]] = {}
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._jobs: Dict[str, DatasetJob] = self._load()
        if recover:
            self.recover_interrupted()

    def _load(self) -> Dict[str, DatasetJob]:
        if not self.state_path.is_file():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {item["id"]: DatasetJob.from_dict(item) for item in payload.get("jobs", [])}
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise JobError(f"Invalid Dataset Lab job state {self.state_path}: {exc}") from exc

    @contextmanager
    def _state(self, *, persist: bool = False):
        """Reload and optionally publish state under a cross-process lock.

        Dataset jobs are intentionally stored beside the managed data rather
        than only in a process-local executor. The dashboard supervisor and a
        headless worker can therefore hold separate ``DatasetLab`` instances.
        Reloading under a small advisory lock prevents either instance from
        serving stale progress or overwriting a concurrent cancellation.
        """

        with self._lock:
            with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    self._jobs = self._load()
                    yield
                    if persist:
                        self._save()
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _save(self) -> None:
        payload = {"format_version": 1, "jobs": [job.to_dict() for job in self._jobs.values()]}
        fd, name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            os.replace(name, self.state_path)
        except Exception:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            raise

    def register(self, kind: str, handler: JobHandler) -> None:
        with self._lock:
            self._handlers[kind] = handler

    def start(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        job_id: Optional[str] = None,
        retry_of: Optional[str] = None,
        checkpoint: Optional[Mapping[str, Any]] = None,
        submit: bool = True,
        work_item_id: Optional[str] = None,
    ) -> DatasetJob:
        with self._state(persist=True):
            if kind not in self._handlers:
                raise JobError(f"No job handler registered for {kind!r}")
            identifier = job_id or uuid.uuid4().hex
            if identifier in self._jobs:
                raise JobError(f"Job ID already exists: {identifier}")
            job = DatasetJob(
                id=identifier,
                kind=kind,
                payload=copy.deepcopy(dict(payload)),
                retry_of=retry_of,
                checkpoint=copy.deepcopy(dict(checkpoint or {})),
                work_item_id=work_item_id,
                worker_pid=os.getpid() if submit else None,
                worker_id=self.worker_id if submit else None,
            )
            self._jobs[identifier] = job
            result = DatasetJob.from_dict(job.to_dict())
        if submit:
            with self._lock:
                self._futures[identifier] = self._executor.submit(self._run, identifier)
        return result

    def _run(self, job_id: str) -> None:
        with self._state(persist=True):
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = job.stage = "cancelled"
                job.finished_at = job.updated_at = _now()
                return
            job.status = "running"
            job.stage = "starting"
            job.worker_pid = os.getpid()
            job.worker_id = self.worker_id
            job.started_at = job.updated_at = _now()
            handler = self._handlers.get(job.kind)
        if handler is None:
            self._fail(job_id, f"No handler registered for recovered job kind {job.kind!r}")
            return
        context = JobContext(self, job_id)
        try:
            result = handler(context, copy.deepcopy(job.payload))
            context.check_cancelled()
        except JobCancelled:
            with self._state(persist=True):
                current = self._jobs[job_id]
                current.status = current.stage = "cancelled"
                current.finished_at = current.updated_at = _now()
        except Exception as exc:
            self._fail(job_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            with self._state(persist=True):
                current = self._jobs[job_id]
                current.result = (
                    result.to_dict() if hasattr(result, "to_dict") else copy.deepcopy(result)
                )
                current.status = "succeeded"
                current.stage = "complete"
                current.finished_at = current.updated_at = _now()

    def run_queued(self, job_id: str) -> DatasetJob:
        """Execute one durable queued/interrupted job in the current process."""

        job = self.get(job_id)
        if job.status not in {"queued", "interrupted"}:
            raise JobError(f"Only queued or interrupted jobs can be claimed (got {job.status})")
        self._run(job_id)
        return self.get(job_id)

    def _fail(self, job_id: str, error: str, trace: Optional[str] = None) -> None:
        with self._state(persist=True):
            job = self._jobs[job_id]
            job.status = "failed"
            job.stage = "failed"
            job.error = error
            if trace:
                job.logs.append(trace)
            job.finished_at = job.updated_at = _now()

    def _update(self, job_id: str, **values: Any) -> None:
        allowed = {"stage", "processed", "total", "accepted", "rejected", "output_size_bytes"}
        with self._state(persist=True):
            job = self._jobs[job_id]
            for key, value in values.items():
                if key in allowed:
                    setattr(job, key, value)
            job.updated_at = _now()

    def _log(self, job_id: str, message: str) -> None:
        with self._state(persist=True):
            job = self._jobs[job_id]
            job.logs.append(str(message))
            job.logs = job.logs[-1000:]
            job.updated_at = _now()

    def _checkpoint(self, job_id: str, values: Mapping[str, Any]) -> None:
        with self._state(persist=True):
            job = self._jobs[job_id]
            job.checkpoint.update(copy.deepcopy(dict(values)))
            job.updated_at = _now()

    def get(self, job_id: str) -> DatasetJob:
        with self._state():
            try:
                return DatasetJob.from_dict(self._jobs[job_id].to_dict())
            except KeyError as exc:
                raise JobError(f"Unknown Dataset Lab job: {job_id}") from exc

    def list(self, *, status: Optional[str] = None) -> List[DatasetJob]:
        with self._state():
            jobs = [
                DatasetJob.from_dict(job.to_dict())
                for job in self._jobs.values()
                if status is None or job.status == status
            ]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def cancel(self, job_id: str) -> DatasetJob:
        with self._state(persist=True):
            job = self._jobs.get(job_id)
            if job is None:
                raise JobError(f"Unknown Dataset Lab job: {job_id}")
            if job.status in TERMINAL_JOB_STATES:
                return DatasetJob.from_dict(job.to_dict())
            job.cancel_requested = True
            job.updated_at = _now()
            future = self._futures.get(job_id)
            if future and future.cancel():
                job.status = job.stage = "cancelled"
                job.finished_at = _now()
            elif job.work_item_id and job.status in {"queued", "interrupted"}:
                job.status = job.stage = "cancelled"
                job.finished_at = _now()
            return DatasetJob.from_dict(job.to_dict())

    def retry(self, job_id: str, *, new_job_id: Optional[str] = None) -> DatasetJob:
        old = self.get(job_id)
        if old.status not in {"failed", "cancelled", "interrupted"}:
            raise JobError(
                f"Only failed, cancelled, or interrupted jobs can be retried (got {old.status})"
            )
        return self.start(
            old.kind, old.payload, job_id=new_job_id, retry_of=old.id, checkpoint=old.checkpoint
        )

    def reset_for_retry(self, job_id: str) -> DatasetJob:
        """Reset a scheduler-owned job in place for its next durable attempt."""

        with self._state(persist=True):
            job = self._jobs.get(job_id)
            if job is None:
                raise JobError(f"Unknown Dataset Lab job: {job_id}")
            if job.status not in {"failed", "cancelled", "interrupted"}:
                raise JobError(
                    "Only failed, cancelled, or interrupted jobs can be retried "
                    f"(got {job.status})"
                )
            job.status = "queued"
            job.stage = "queued"
            job.error = None
            job.cancel_requested = False
            job.worker_pid = None
            job.worker_id = None
            job.started_at = None
            job.finished_at = None
            job.updated_at = _now()
            return DatasetJob.from_dict(job.to_dict())

    def recover_interrupted(self) -> int:
        changed = 0
        with self._state(persist=True):
            for job in self._jobs.values():
                scheduler_queued = job.status == "queued" and bool(job.work_item_id)
                if (
                    job.status in {"queued", "running"}
                    and not scheduler_queued
                    and not _pid_is_alive(job.worker_pid)
                ):
                    job.status = job.stage = "interrupted"
                    job.error = "application stopped before job completion"
                    job.finished_at = job.updated_at = _now()
                    changed += 1
        return changed

    def wait(self, job_id: str, timeout: Optional[float] = None) -> DatasetJob:
        future = self._futures.get(job_id)
        if future:
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                from concurrent.futures import TimeoutError

                if isinstance(exc, TimeoutError):
                    raise
                # Handler errors are captured into job state.
        return self.get(job_id)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = ["DatasetJob", "JobCancelled", "JobContext", "SerialJobManager", "TERMINAL_JOB_STATES"]
