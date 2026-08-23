"""Isolated qualification payload executed inside a managed image.

Every subcommand emits one JSON object and exits non-zero on missing or
non-finite evidence. This module is intentionally independent of the host API.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "datasets": "3.6.0",
    "trl": "0.29.1",
    "accelerate": "1.14.0",
}


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot enumerate a CUDA/HIP accelerator")
    return torch


def dependencies() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {name: [EXPECTED[name], value] for name, value in versions.items() if value != EXPECTED[name]}
    if mismatches:
        raise RuntimeError(f"dependency lock mismatch: {mismatches}")
    torch = importlib.metadata.version("torch")
    return {"versions": versions, "torch": torch}


def enumerate_gpu() -> dict[str, Any]:
    torch = _torch()
    return {
        "count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
    }


def kernels() -> dict[str, Any]:
    torch = _torch()
    fp32 = torch.ones((64, 64), device="cuda", dtype=torch.float32)
    bf16 = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    result = bf16 @ bf16
    torch.cuda.synchronize()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("BF16 matmul produced non-finite output")
    return {"fp32_bytes": fp32.numel() * fp32.element_size(), "bf16_finite": True}


def optimizer() -> dict[str, Any]:
    torch = _torch()
    layer = torch.nn.Linear(32, 16, device="cuda", dtype=torch.float32)
    before = layer.weight.detach().clone()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    loss = layer(torch.randn(8, 32, device="cuda")).square().mean()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    changed = not torch.equal(before, layer.weight.detach())
    if not changed or not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError("optimizer qualification did not produce a finite weight update")
    return {"loss": float(loss.detach().cpu()), "weights_updated": changed}


def sft_proof(output: Path) -> dict[str, Any]:
    torch = _torch()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    batch = tokenizer("User: return 2 + 2. Assistant: 4", return_tensors="pt").to("cuda")
    before = next(parameter for parameter in model.parameters() if parameter.requires_grad).detach().clone()
    opt = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4)
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    opt.step()
    changed = not torch.equal(before, next(parameter for parameter in model.parameters() if parameter.requires_grad).detach())
    if not changed or not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError("Qwen SFT proof did not produce a finite adapter update")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return {"model": model_id, "loss": float(loss.detach().cpu()), "weights_updated": True, "artifact": str(output)}


def reload_adapter(output: Path) -> dict[str, Any]:
    torch = _torch()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(output)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, output)
    with torch.no_grad():
        logits = model(**tokenizer("User: 2 + 2", return_tensors="pt").to("cuda")).logits
    finite = bool(torch.isfinite(logits).all())
    if not finite:
        raise RuntimeError("reloaded adapter produced non-finite logits")
    return {"reloaded": True, "finite_output": finite}


def trainer_diagnostic(trainer: str) -> dict[str, Any]:
    """Exercise each objective's tensor/update shape inside the image.

    This is the runtime layer of the contract. Guided exposure additionally
    requires the application's trainer and capacity adapter contract evidence;
    this result alone never advertises a scenario.
    """

    torch = _torch()
    trainer = trainer.lower()
    model = torch.nn.Linear(16, 8, device="cuda")
    before = model.weight.detach().clone()
    optimizer_value = torch.optim.AdamW(model.parameters(), lr=1e-3)
    left = model(torch.randn(4, 16, device="cuda"))
    if trainer in {"dpo", "orpo", "rm", "rerank"}:
        right = model(torch.randn(4, 16, device="cuda"))
        loss = -torch.nn.functional.logsigmoid(left.mean(-1) - right.mean(-1)).mean()
    elif trainer == "embed":
        right = model(torch.randn(4, 16, device="cuda"))
        loss = torch.nn.functional.cross_entropy(left @ right.T, torch.arange(4, device="cuda"))
    elif trainer in {"classify", "image_classify", "audio_classify"}:
        loss = torch.nn.functional.cross_entropy(left, torch.tensor([0, 1, 2, 3], device="cuda"))
    else:
        loss = left.square().mean()
    loss.backward()
    optimizer_value.step()
    changed = not torch.equal(before, model.weight.detach())
    if not changed or not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError(f"{trainer} runtime contract did not update weights")
    return {
        "trainer": trainer,
        "diagnostic_only": True,
        "runtime_tensor_contract": True,
        "weights_updated": True,
        "loss": float(loss.detach().cpu()),
        "certifies_guided_training": False,
    }


# Retain the old import for scripts written before V21, but make its evidence
# explicitly diagnostic and never consume it as training-path certification.
trainer_contract = trainer_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("dependencies", "enumerate", "kernels", "optimizer", "sft-proof", "adapter-reload", "trainer-contract", "trainer-diagnostic"))
    parser.add_argument("--output", type=Path, default=Path("/evidence/sft-proof"))
    parser.add_argument("--trainer")
    args = parser.parse_args()
    if args.step == "dependencies": result = dependencies()
    elif args.step == "enumerate": result = enumerate_gpu()
    elif args.step == "kernels": result = kernels()
    elif args.step == "optimizer": result = optimizer()
    elif args.step == "sft-proof": result = sft_proof(args.output)
    elif args.step == "adapter-reload": result = reload_adapter(args.output)
    else:
        if not args.trainer: raise ValueError("--trainer is required")
        result = trainer_diagnostic(args.trainer)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
