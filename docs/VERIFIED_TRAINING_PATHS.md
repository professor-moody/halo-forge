# Verified training paths

Halo Forge separates four readiness states that are easy to confuse:

1. **Runtime ready** — the pinned accelerator runtime passed device, BF16, backward, and AdamW checks.
2. **Path verified** — Dataset Lab rendered a pinned fixture, the shipped trainer changed trainable-parameter hashes, and the saved artifact reloaded with finite output.
3. **Plan ready** — the exact user dataset and compute shape passed the disposable capacity check.
4. **Beta qualified** — the complete own-data, proof, evaluation, recovery, coexistence, and soak workflow has current evidence.

A GPU being detected is not one of these states. The legacy per-trainer tensor probes are diagnostics only and never unlock guided training.

## Normal workflow

Setup's **Prepare AMD training** action prepares the pinned runtime, performs core qualification, and queues instruction-SFT certification. Additional paths can be verified on selection only after their real certification executor ships; until then they are visibly unavailable with an exact reason. If an external accelerator process is present, work stays in **Waiting for accelerator** without consuming retries. Halo Forge never terminates that process.

Train shows one next action:

- **Verify this training path** when the real trainer path lacks current evidence;
- **Prepare and check** after path verification;
- **Start proof run** after capacity verification.

Technical details retain the runtime digest, fixture hash, exact model commit, trainer and capacity-adapter versions, parameter hashes, artifact checksums, replay v14 identity, and recovery cursor.
Replay v14 therefore preserves the exact certification used by a managed run.

## CLI

```text
halo-forge runtime paths --family rocm
halo-forge runtime certify TRAINING_PATH_REVISION --wait
halo-forge runtime certification show CERTIFICATION_ID
halo-forge runtime certification steps CERTIFICATION_ID
halo-forge runtime certification verify CERTIFICATION_ID
halo-forge runtime certification retry CERTIFICATION_ID --reason "runtime repaired"
halo-forge release workstation-certify --runtime-revision RUNTIME_REVISION
halo-forge release workstation-report CERTIFICATION_ID
```

CUDA paths remain hardware-gated until the same evidence is produced on a real NVIDIA host. Fedora Strix evidence is reported as `local_verified`, not vendor-supported.
