---
title: "Hugging Face access"
description: "Workstation-scoped Hugging Face tokens for gated and private model downloads."
weight: 71
---

Halo Forge uses two separate tokens:

- **Halo Forge API tokens** (`hfk_...`) protect non-loopback workstation access.
- **Hugging Face tokens** (`hf_...`) let that workstation download gated or private Hugging Face models.

The dashboard never stores Hugging Face tokens in browser `localStorage`. Use **Connection → Hugging Face access** to paste, verify, and save a read token. The backend stores it on the workstation, then passes it to dashboard-managed serving and dashboard-launched training as `HF_TOKEN`.

## Storage order

Halo Forge resolves Hugging Face credentials in this order:

1. `HF_TOKEN` from the workstation process environment.
2. OS keyring, when available.
3. `~/.halo-forge/secrets/huggingface_token` with `0600` permissions as a Linux/headless fallback.

The API only returns sanitized status such as source, presence, and verified username. It never returns the token value.

If `HF_TOKEN` is set in the environment, the dashboard can use it but cannot clear it. Clear that value from the shell, launch agent, service unit, or desktop runtime environment that started Halo Forge.

## Gated model flow

Some models require accepting a license on Hugging Face before downloads work. Halo Forge will not accept licenses for you.

1. Open the model page from **Models**.
2. Accept the license on Hugging Face if required.
3. Return to **Connection → Hugging Face access** and save a read token.
4. Use **Check access** from Models before serving or training.

If serving fails because access is missing, Playground shows actions to connect Hugging Face, choose an open model, or open the model page.

## Remote workstation behavior

Remote workstation mode still controls one Halo Forge host. Hugging Face access belongs on that host because that is where model downloads happen. A browser client should only enter an `hf_...` token when it is allowed to manage credentials for the workstation it is connected to.

## What is never written

Halo Forge does not write Hugging Face token values into:

- API responses,
- `launch_context.json`,
- serve logs,
- runtime logs,
- run artifacts,
- dashboard screenshots.
