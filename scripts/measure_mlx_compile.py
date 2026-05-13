#!/usr/bin/env python3
"""Measure MLX eager vs compiled loss/reduction candidates.

This is intentionally a measurement harness, not a production training path.
It uses synthetic tensors so it can isolate candidate reductions without
downloading a model.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


CANDIDATES = (
    "dpo_reference_free_sigmoid",
    "dpo_reference_model_sigmoid",
    "grpo_advantage_loss",
)


@dataclass
class Measurement:
    name: str
    first_step_seconds: float
    steady_state_seconds_mean: float
    steady_state_seconds_p50: float
    steps: int
    warmup: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32, help="Single batch size when --batch-sizes is omitted")
    parser.add_argument(
        "--batch-sizes",
        default="32,128,512",
        help="Comma-separated batch sizes to measure (default: 32,128,512)",
    )
    parser.add_argument(
        "--candidate",
        choices=("all",) + CANDIDATES,
        default="all",
        help="Candidate to measure, or all candidates",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def _require_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise SystemExit(
            "MLX is not installed. Install halo-forge with `pip install -e '.[mlx]'` "
            "on Apple Silicon, then rerun this measurement."
        ) from exc
    return mx


def _mlx_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_metal_unavailable(exc: BaseException) -> bool:
    message = str(exc)
    return "No Metal device available" in message or "metal::load_device" in message


def _unavailable_result(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "candidate": args.candidate,
        "candidates": _selected_candidates(args),
        "status": "unavailable",
        "reason": reason,
        "shapes": [{"batch_size": batch_size} for batch_size in _batch_sizes(args)],
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "mlx_version": _mlx_version("mlx"),
        "mlx_lm_version": _mlx_version("mlx-lm"),
        "decision": "measurement_only",
    }


def _batch_sizes(args: argparse.Namespace) -> list[int]:
    raw = str(args.batch_sizes or "").strip()
    if not raw:
        return [args.batch_size]
    out: list[int] = []
    for part in raw.split(","):
        value = int(part.strip())
        if value <= 0:
            raise SystemExit("--batch-sizes values must be positive")
        out.append(value)
    return out


def _selected_candidates(args: argparse.Namespace) -> list[str]:
    if args.candidate == "all":
        return list(CANDIDATES)
    return [args.candidate]


def _dpo_loss_fn(mx: Any, beta: float, label_smoothing: float, *, reference_model: bool) -> Callable[..., Any]:
    def loss(chosen, rejected, ref_chosen=None, ref_rejected=None):
        margin = chosen - rejected
        if reference_model:
            margin = margin - (ref_chosen - ref_rejected)
        logits = beta * margin
        positive = mx.logaddexp(mx.array(0.0), -logits)
        if label_smoothing > 0:
            negative = mx.logaddexp(mx.array(0.0), logits)
            return ((1 - label_smoothing) * positive + label_smoothing * negative).mean()
        return positive.mean()

    return loss


def _grpo_loss_fn(mx: Any) -> Callable[..., Any]:
    def loss(logps, advantages):
        return -(logps * advantages).mean()

    return loss


def _candidate_inputs(mx: Any, candidate: str, batch_size: int, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    if candidate == "dpo_reference_free_sigmoid":
        fn = _dpo_loss_fn(mx, args.beta, args.label_smoothing, reference_model=False)
        return fn, (
            mx.random.normal((batch_size,)),
            mx.random.normal((batch_size,)),
        )
    if candidate == "dpo_reference_model_sigmoid":
        fn = _dpo_loss_fn(mx, args.beta, args.label_smoothing, reference_model=True)
        return fn, (
            mx.random.normal((batch_size,)),
            mx.random.normal((batch_size,)),
            mx.random.normal((batch_size,)),
            mx.random.normal((batch_size,)),
        )
    if candidate == "grpo_advantage_loss":
        rewards = mx.random.normal((batch_size,))
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return _grpo_loss_fn(mx), (
            mx.random.normal((batch_size,)),
            advantages,
        )
    raise ValueError(f"unknown candidate: {candidate}")


def _time_callable(mx: Any, fn: Callable[[], Any], *, steps: int, warmup: int, name: str) -> Measurement:
    times: list[float] = []
    first = 0.0
    total = steps + warmup
    for idx in range(total):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        elapsed = time.perf_counter() - start
        if idx == 0:
            first = elapsed
        if idx >= warmup:
            times.append(elapsed)
    return Measurement(
        name=name,
        first_step_seconds=first,
        steady_state_seconds_mean=statistics.fmean(times),
        steady_state_seconds_p50=statistics.median(times),
        steps=steps,
        warmup=warmup,
    )


def _metal_memory(mx: Any) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for label, attr in {
        "active_memory_bytes": "get_active_memory",
        "peak_memory_bytes": "get_peak_memory",
        "cache_memory_bytes": "get_cache_memory",
    }.items():
        getter = getattr(mx, attr, None)
        if getter is None:
            metal = getattr(mx, "metal", None)
            getter = getattr(metal, attr, None) if metal is not None else None
        if getter is not None:
            try:
                out[label] = int(getter())
            except Exception:
                out[label] = None
    return out


def _measure_candidate(mx: Any, args: argparse.Namespace, candidate: str, batch_size: int) -> dict[str, Any]:
    fn, inputs = _candidate_inputs(mx, candidate, batch_size, args)
    compiled_fn = getattr(mx, "compile", None)
    if compiled_fn is None:
        raise SystemExit("This MLX version does not expose mx.compile.")
    compiled = compiled_fn(fn)

    eager = _time_callable(
        mx,
        lambda: fn(*inputs),
        steps=args.steps,
        warmup=args.warmup,
        name="eager",
    )
    compiled_result = _time_callable(
        mx,
        lambda: compiled(*inputs),
        steps=args.steps,
        warmup=args.warmup,
        name="compiled",
    )
    return {
        "candidate": candidate,
        "shape": {"batch_size": batch_size},
        "status": "measured",
        "measurements": [asdict(eager), asdict(compiled_result)],
        "memory": _metal_memory(mx),
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    mx = _require_mlx()
    mx.random.seed(args.seed)

    results: list[dict[str, Any]] = []
    for candidate in _selected_candidates(args):
        for batch_size in _batch_sizes(args):
            try:
                results.append(_measure_candidate(mx, args, candidate, batch_size))
            except RuntimeError as exc:
                if _is_metal_unavailable(exc):
                    raise
                results.append(
                    {
                        "candidate": candidate,
                        "shape": {"batch_size": batch_size},
                        "status": "error",
                        "reason": str(exc),
                        "memory": _metal_memory(mx),
                    }
                )

    return {
        "candidate": args.candidate,
        "candidates": _selected_candidates(args),
        "shapes": [{"batch_size": batch_size} for batch_size in _batch_sizes(args)],
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "mlx_version": getattr(mx, "__version__", None),
        "mlx_lm_version": _mlx_version("mlx-lm"),
        "results": results,
        "decision": "measurement_only",
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_measurement(args)
    except RuntimeError as exc:
        if not _is_metal_unavailable(exc):
            raise
        result = _unavailable_result(args, str(exc))
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(2)
        print("MLX compile measurement unavailable")
        print(f"- reason: {result['reason']}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("MLX DPO sigmoid loss compile measurement")
    for item in result["results"]:
        shape = item["shape"]["batch_size"]
        if item["status"] != "measured":
            print(f"- {item['candidate']} batch={shape}: {item['status']} {item.get('reason', '')}")
            continue
        print(f"- {item['candidate']} batch={shape}")
        for entry in item["measurements"]:
            print(
                f"  {entry['name']}: first={entry['first_step_seconds']:.6f}s "
                f"mean={entry['steady_state_seconds_mean']:.6f}s "
                f"p50={entry['steady_state_seconds_p50']:.6f}s"
            )
        if item["memory"]:
            print(f"  memory={item['memory']}")


if __name__ == "__main__":
    main()
