#!/usr/bin/env python3
"""Measure MLX eager vs compiled DPO sigmoid loss.

This is intentionally a measurement harness, not a production training path.
It uses synthetic log-prob tensors so it can isolate the DPO loss/reduction
candidate without downloading a model.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


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
    parser.add_argument("--batch-size", type=int, default=32)
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


def _loss_fn(mx: Any, beta: float, label_smoothing: float) -> Callable[..., Any]:
    def loss(chosen, rejected, ref_chosen, ref_rejected):
        logits = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
        positive = mx.logaddexp(mx.array(0.0), -logits)
        if label_smoothing > 0:
            negative = mx.logaddexp(mx.array(0.0), logits)
            return ((1 - label_smoothing) * positive + label_smoothing * negative).mean()
        return positive.mean()

    return loss


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
    metal = getattr(mx, "metal", None)
    if metal is None:
        return {}
    out: dict[str, int | None] = {}
    for label, attr in {
        "active_memory_bytes": "get_active_memory",
        "peak_memory_bytes": "get_peak_memory",
        "cache_memory_bytes": "get_cache_memory",
    }.items():
        getter = getattr(metal, attr, None)
        if getter is not None:
            try:
                out[label] = int(getter())
            except Exception:
                out[label] = None
    return out


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    mx = _require_mlx()
    mx.random.seed(args.seed)

    chosen = mx.random.normal((args.batch_size,))
    rejected = mx.random.normal((args.batch_size,))
    ref_chosen = mx.random.normal((args.batch_size,))
    ref_rejected = mx.random.normal((args.batch_size,))
    eager_loss = _loss_fn(mx, args.beta, args.label_smoothing)
    compiled_loss = getattr(mx, "compile", None)
    if compiled_loss is None:
        raise SystemExit("This MLX version does not expose mx.compile.")
    compiled = compiled_loss(eager_loss)

    eager = _time_callable(
        mx,
        lambda: eager_loss(chosen, rejected, ref_chosen, ref_rejected),
        steps=args.steps,
        warmup=args.warmup,
        name="eager",
    )
    compiled_result = _time_callable(
        mx,
        lambda: compiled(chosen, rejected, ref_chosen, ref_rejected),
        steps=args.steps,
        warmup=args.warmup,
        name="compiled",
    )

    return {
        "candidate": "mlx_dpo_sigmoid_loss",
        "shape": {"batch_size": args.batch_size},
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "mlx_version": getattr(mx, "__version__", None),
        "measurements": [asdict(eager), asdict(compiled_result)],
        "memory": _metal_memory(mx),
        "decision": "measurement_only",
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_measurement(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("MLX DPO sigmoid loss compile measurement")
    for entry in result["measurements"]:
        print(
            f"- {entry['name']}: first={entry['first_step_seconds']:.6f}s "
            f"mean={entry['steady_state_seconds_mean']:.6f}s "
            f"p50={entry['steady_state_seconds_p50']:.6f}s"
        )
    if result["memory"]:
        print(f"memory={result['memory']}")


if __name__ == "__main__":
    main()
