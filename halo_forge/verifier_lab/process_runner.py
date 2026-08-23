"""Isolated invocation entry point for deterministic registry verifiers.

The normal calibration path starts one interpreter for each replicated pass
and streams that pass' records through it.  The two pass interpreters are
therefore genuinely fresh and independent without paying process-start cost
for every record.  The single-request mode remains available for diagnostics
and backwards-compatible direct invocation.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from typing import Any, Mapping

from .adapters import RegistryVerifierReliabilityAdapter


RESULT_PREFIX = "HALO_FORGE_VERIFIER_OBSERVATION="


def invoke_request(request: Mapping[str, Any]) -> dict[str, Any]:
    adapter = RegistryVerifierReliabilityAdapter(
        str(request["implementation_ref"]),
        configuration=dict(request.get("configuration") or {}),
        family="deterministic",
        modalities=(str(request.get("modality") or "text"),),
        tasks=(str(request.get("task_type") or "binary"),),
    )
    runtime = {
        **dict(request.get("runtime") or {}),
        "process_isolation": "fresh_interpreter",
        "process_id": os.getpid(),
    }
    return adapter.invoke(
        dict(request.get("item") or {}),
        contract=dict(request.get("reward_contract") or {}),
        runtime=runtime,
    ).to_dict()


def _emit(result: Mapping[str, Any]) -> None:
    print(
        RESULT_PREFIX + json.dumps(dict(result), sort_keys=True, allow_nan=False),
        flush=True,
    )


def _stream_main() -> int:
    """Serve newline-delimited requests for one isolated calibration pass."""

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("isolated verifier request must be an object")
            _emit(invoke_request(request))
        except Exception as exc:  # keep the pass alive; the error is evidence
            _emit({"error": f"{type(exc).__name__}: {exc}"})
    return 0


class IsolatedVerifierPass:
    """Parent-side handle for one fresh deterministic verifier pass."""

    def __init__(self, *, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = max(1.0, min(3600.0, float(timeout_seconds)))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "halo_forge.verifier_lab.process_runner", "--stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise RuntimeError("could not open isolated verifier process pipes")

    @property
    def process_id(self) -> int:
        return int(self.process.pid)

    def invoke(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError("isolated verifier pass exited unexpectedly")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(dict(request), sort_keys=True, allow_nan=False) + "\n")
        self.process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("isolated verifier pass timed out")
                if not selector.select(timeout=remaining):
                    raise TimeoutError("isolated verifier pass timed out")
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("isolated verifier pass closed its output")
                if not line.startswith(RESULT_PREFIX):
                    continue
                value = json.loads(line[len(RESULT_PREFIX) :])
                if not isinstance(value, Mapping):
                    raise RuntimeError("isolated verifier returned a non-object result")
                return dict(value)
        finally:
            selector.close()

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)

    def __enter__(self) -> "IsolatedVerifierPass":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def main() -> int:
    if "--stream" in sys.argv[1:]:
        return _stream_main()
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, Mapping):
            raise ValueError("isolated verifier request must be an object")
        result = invoke_request(request)
        _emit(result)
        return 0
    except Exception as exc:  # pragma: no cover - exercised by the parent boundary
        error = {"error": f"{type(exc).__name__}: {exc}"}
        _emit(error)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
