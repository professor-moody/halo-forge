---
title: "Hardware And Capability Notes"
description: "Truthful backend, accelerator, telemetry, and desktop-platform coverage"
weight: 3
---

Halo Forge uses the same commands across backends, but it does **not** assume
that every trainer, model adapter, quantizer, evaluator, or telemetry sensor is
available everywhere. The dashboard preflight and CLI capability check are the
authority for a particular launch.

## Runtime Matrix

| Runtime | Training truth | Important limits |
|---|---|---|
| NVIDIA CUDA + PyTorch | Managed CUDA profile; guided combinations appear only after real-hardware qualification | V20 remains release-blocked until the exact trainer/model/runtime update and reload pass on NVIDIA hardware |
| AMD ROCm + PyTorch | Pinned ROCm 7.2.1 managed profile; Strix Halo is certified first | Hardware detection alone is not readiness; local qualification must pass while independently idle |
| Apple MPS + PyTorch | Torch training for methods whose model adapter supports MPS | No bitsandbytes QLoRA; unsupported operators or third-party models can still fail preflight |
| Apple MLX | Native SFT, RAFT, DPO, and GRPO implementations | Not a universal VLM/audio/reasoning/agentic backend; advanced PEFT options are not silently emulated |
| CPU | Validation, metadata jobs, evaluation helpers, and tiny smoke runs | Not presented as a practical heavy-training target |

MLX DPO supports sigmoid, IPO, hinge, and KTO-pair paths. MLX GRPO has native
eager paths, but resumability and reward-audit boundaries may be more limited
than their PyTorch equivalents. The capability preview states the effective
behavior before launch.

## Desktop And Browser Matrix

| Surface/host | Current state |
|---|---|
| macOS arm64 desktop | DMG engineering contract complete; normal install requires signed/notarized release qualification |
| Linux x86-64 desktop | AppImage and Debian engineering contracts complete; unsigned candidates remain preview-only |
| Windows x86-64 desktop | NSIS, runtime sidecar, native picker, path/process, health, and proof-smoke contracts; unsigned candidates remain preview-only |
| Local browser | Supported on macOS, Linux, and Windows wherever the selected runtime/backend installs successfully |
| Remote browser | Same dashboard/API against a supported workstation host; bearer token required for non-loopback binding |
| CLI | Python 3.10–3.13; feature readiness still depends on backend, extras, model, memory, and disk |

Do not bypass Gatekeeper or SmartScreen for normal use. The release manifest,
not source metadata, determines whether a desktop artifact is trusted.

## What Preflight Checks

Before managed heavy work, Halo Forge resolves:

- trainer and backend compatibility;
- model/tokenizer and dataset-artifact compatibility;
- required optional packages and runtime identity;
- image/audio asset availability where applicable;
- RAM and disk headroom; and
- reward/verifier qualification and resumability when requested.

An accelerator can be detected while one of those checks still refuses the
launch. That is expected and safer than silently changing the requested method.

## Telemetry Truth

GPU utilization, device memory, power, and temperature are reported only when
the platform exposes them. **Unavailable does not mean zero.** Halo Forge never
fabricates a GPU metric and does not turn a missing sensor into a scientific or
cost measurement.

## AMD Strix Halo Defaults

Strix Halo uses unified memory. Start with:

```yaml
training:
  dataloader_num_workers: 0
  dataloader_pin_memory: false
  bf16: true
  fp16: false
```

In the tested workstation configuration, BF16 is the conservative starting
point. Do not assume QLoRA is faster merely because 4-bit loading is available;
the ROCm/bitsandbytes combination and the selected model must pass preflight.

Useful checks:

```bash
halo-forge doctor
rocm-smi --showmeminfo vram
ls -l /dev/dri /dev/kfd
```

For older kernels or custom ROCm installations, consult the platform/vendor
instructions before changing kernel parameters or device permissions. Halo
Forge does not automatically apply system-wide settings.

## Apple Silicon

Use MPS for PyTorch trainers and MLX only for trainers with a declared MLX
implementation. Prefer MLX-format model repositories for native MLX work. A
Hugging Face PyTorch model name does not guarantee an equivalent MLX artifact,
and MPS availability does not make bitsandbytes features available.

Run:

```bash
halo-forge doctor mlx --json
halo-forge models list --backend mlx
```

## Planning Memory

Model size alone is not a memory forecast. Training method, precision, adapter
mode, reference models, optimizer state, sequence length, candidate count,
media tensors, evaluation subjects, and serving leases all matter. Use the
request preview and workstation preflight; treat published size tiers as
orientation, not a capacity guarantee.
