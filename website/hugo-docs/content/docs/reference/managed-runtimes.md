---
title: "Managed Accelerator Runtimes"
description: "How Halo Forge verifies ROCm and CUDA before guided training"
weight: 24
---

Halo Forge does not treat a detected GPU as proof that training works. On ROCm
and CUDA hosts, guided training uses a pinned managed runtime that must pass a
real hardware qualification first.

The normal dashboard flow is:

```text
Setup → Prepare AMD/NVIDIA training → wait for verification → Train
```

No container knowledge is required. The image reference, digest, mounts, and
raw logs remain under **Technical details**.

## Strix Halo profile

The V19 profile pins the official ROCm 7.2.1 / PyTorch 2.9.1 image by digest
and builds a Halo Forge layer with pinned Transformers, PEFT, Datasets, TRL,
and Accelerate versions. The dashboard previews approximately 10.7 GB of
download and 31 GB installed storage before asking for confirmation.

Core qualification verifies device access, dependencies, GPU enumeration,
FP32 and BF16 kernels, and backward plus AdamW. V21 then separately sends a
pinned Qwen 0.5B fixture through Dataset Lab and the shipped SFT trainer,
checks before/after trainable-parameter hashes, and reloads the saved adapter.
Only that second record unlocks guided SFT. Fedora results are reported as **Locally verified**; AMD's
documented Ubuntu host combination can be reported as **Vendor supported**.

The earlier Strix proof established that the official image can perform a real
optimizer update and reload an adapter. It overlapped an unrelated `/dev/kfd`
owner, so it is functional evidence—not isolated capacity or performance
certification. Formal qualification reruns only after Halo Forge independently
observes an idle accelerator.

## CUDA profile

V20 uses the same contracts with the pinned PyTorch 2.9.1 CUDA 12.8 image.
Podman or Docker is selected only after non-root engine and NVIDIA device
checks. CUDA guided scenarios stay unavailable until the exact runtime,
trainer, model, optimizer update, and artifact reload have passed on real
NVIDIA hardware.

## Safe coexistence

Halo Forge samples accelerator occupancy three times, two seconds apart, and
checks again immediately before launch. Existing compute produces **Waiting
for accelerator** and does not consume a retry. Activity shows only the
executable basename, PID, and elapsed time; full command lines and data paths
are not exposed.

Halo Forge never terminates an external process. If new external work appears
after launch, reusable performance evidence is invalidated. Segmented work
stops before its next checkpoint segment can claim the accelerator;
non-resumable work continues with a visible contention warning.

## CLI

```bash
halo-forge runtime list
halo-forge runtime show RUNTIME_REVISION
halo-forge runtime prepare RUNTIME_REVISION
halo-forge runtime qualify RUNTIME_REVISION
halo-forge runtime verify QUALIFICATION_ID
halo-forge runtime paths --family rocm
halo-forge runtime certify TRAINING_PATH_REVISION

halo-forge accelerator status --family rocm
halo-forge accelerator wait --family rocm --timeout 3600
```

The Mac desktop or browser may operate Halo Forge running on a Strix host
through the existing remote-browser surface. The scheduler and training
process still run on the compute host; there is no SSH worker or distributed
scheduler.
