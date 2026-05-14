#!/usr/bin/env python3
"""Measure MLX reference-model GRPO feasibility.

This is a Terminal-only measurement harness. It intentionally does not enable
or tune any production path; it proves whether an Apple Silicon process can
hold policy + frozen reference models and run one bounded GRPO-style update.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from halo_forge.backend.mlx_readiness import check_mlx_readiness


DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", help="Optional directory for temporary measurement files")
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def _is_metal_unavailable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "no metal device available" in message or "metal::load_device" in message


def _metal_memory(mx: Any) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for label, attr in {
        "active_memory_bytes": "get_active_memory",
        "peak_memory_bytes": "get_peak_memory",
        "cache_memory_bytes": "get_cache_memory",
    }.items():
        getter = getattr(mx, attr, None)
        if getter is not None:
            try:
                out[label] = int(getter())
            except Exception:
                out[label] = None
    return out


def _unavailable_result(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    readiness = check_mlx_readiness()
    return {
        "status": "unavailable",
        "reason": reason,
        "decision": "measurement_only",
        "model": args.model,
        "readiness": readiness.to_dict(),
        "platform": _platform(),
    }


def _platform() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
    }


def _count_blocks(model: Any) -> int:
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
    try:
        return len(layers) if layers is not None else 0
    except TypeError:
        return 0


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    readiness = check_mlx_readiness()
    base: dict[str, Any] = {
        "status": "measured",
        "decision": "measurement_only",
        "model": args.model,
        "readiness": readiness.to_dict(),
        "platform": _platform(),
        "config": {
            "num_generations": args.num_generations,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
        },
        "timings": {},
        "memory": {},
        "checks": {},
    }
    if not readiness.executable:
        base["status"] = "unavailable"
        base["reason"] = "MLX is not executable in this process environment."
        return base

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from halo_forge.dpo.mlx_trainer import _response_logprobs
    from halo_forge.grpo.mlx_trainer import _grpo_policy_loss
    from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator

    mx.random.seed(args.seed)
    prompt = "Give a concise answer: what is 2 + 2?"
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    temp_root = Path(args.output_dir).expanduser() if args.output_dir else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        cache_path = Path(tmp) / "rollouts.jsonl"
        start = time.perf_counter()
        rollout = MLXRolloutGenerator(args.model)
        samples = rollout.generate_samples(
            [prompt],
            num_samples=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=1,
            system_prompt="",
            cache_path=cache_path,
        )
        rollout.cleanup()
        timings["rollout_seconds"] = time.perf_counter() - start
        base["checks"]["rollout_samples"] = len(samples)

        start = time.perf_counter()
        policy, tokenizer = mlx_load(args.model)
        timings["policy_load_seconds"] = time.perf_counter() - start
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy.freeze()
        linear_to_lora_layers(
            policy,
            num_layers=_count_blocks(policy) or 16,
            config={
                "rank": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": 0.0,
                "scale": float(args.lora_alpha) / float(args.lora_r) if args.lora_r else 1.0,
            },
        )

        start = time.perf_counter()
        reference, _ = mlx_load(args.model)
        reference.freeze()
        timings["reference_load_seconds"] = time.perf_counter() - start

        optimizer = optim.AdamW(learning_rate=args.learning_rate)
        completion = samples[-1][1] or "4"
        prompt_tokens = mx.array(tokenizer.encode(prompt))
        completion_tokens = mx.array(tokenizer.encode(completion))
        if completion_tokens.shape[0] == 0:
            completion_tokens = mx.array(tokenizer.encode("4"))

        def loss_fn(model, prompt_ids, completion_ids, advantage):
            policy_logp = _response_logprobs(
                mx=mx,
                nn=nn,
                model=model,
                prompt_tokens=prompt_ids,
                response_tokens=completion_ids,
            )
            reference_logp = _response_logprobs(
                mx=mx,
                nn=nn,
                model=reference,
                prompt_tokens=prompt_ids,
                response_tokens=completion_ids,
            )
            kl = policy_logp - reference_logp
            loss = _grpo_policy_loss(
                policy_logp,
                advantage=float(advantage),
                beta=args.beta,
                reference_logp=reference_logp,
            )
            return loss, (policy_logp, reference_logp, kl)

        loss_and_grad = nn.value_and_grad(policy, loss_fn)
        start = time.perf_counter()
        (loss_value, aux), grads = loss_and_grad(
            policy,
            prompt_tokens,
            completion_tokens,
            1.0,
        )
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state)
        timings["one_update_seconds"] = time.perf_counter() - start

        policy_logp, reference_logp, kl_value = aux
        base["checks"]["one_update"] = True
        base["metrics"] = {
            "loss": float(loss_value),
            "policy_logp": float(policy_logp),
            "reference_logp": float(reference_logp),
            "kl": float(kl_value),
            "completion_tokens": int(completion_tokens.shape[0]),
        }

    timings["total_seconds"] = time.perf_counter() - total_start
    base["timings"] = timings
    base["memory"] = _metal_memory(mx)
    return base


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_measurement(args)
    except RuntimeError as exc:
        if not _is_metal_unavailable(exc):
            raise
        result = _unavailable_result(args, str(exc))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"MLX reference-model GRPO measurement: {result['status']}")
        if result.get("reason"):
            print(f"- reason: {result['reason']}")
        if result.get("metrics"):
            print(f"- metrics: {result['metrics']}")
        if result.get("memory"):
            print(f"- memory: {result['memory']}")
    if result["status"] == "measured":
        return 0
    if result["status"] == "unavailable":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
