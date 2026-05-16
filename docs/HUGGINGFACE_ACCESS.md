# Hugging Face Access

Halo Forge has two separate token flows:

- **Halo Forge API token** (`hfk_...`) controls access to a non-loopback Halo Forge workstation.
- **Hugging Face token** (`hf_...`) lets that workstation download gated or private Hugging Face models.

The dashboard stores Hugging Face access on the workstation, not in the browser. Go to **Connection** and use **Hugging Face access** to paste, verify, and save a read token. The saved token is never returned by the API, written to `launch_context.json`, or shown in the dashboard after submission.

## Storage Precedence

Halo Forge resolves Hugging Face access in this order:

1. `HF_TOKEN` in the workstation process environment.
2. The OS keyring, when available.
3. `~/.halo-forge/secrets/huggingface_token` with `0600` permissions as a headless/Linux fallback.

If `HF_TOKEN` is set, the dashboard can use it but cannot clear it. Clear or rotate that credential in the shell, launch agent, service unit, or desktop runtime environment that started Halo Forge.

## Gated Models

Some models, such as Llama-family checkpoints, require accepting a license on Hugging Face before downloads work. Halo Forge does not accept licenses for you. Use the model page link in **Models**, accept the license on Hugging Face, then return to **Connection** and verify access.

When a model load fails because access is missing, Playground shows actions to:

- connect Hugging Face,
- choose an open model,
- open the model page.

Open Qwen/MLX catalog entries remain the safest first-serving path.

## Remote Workstations

Remote workstation mode still uses one Halo Forge host. The Hugging Face token belongs on that host because that is where training, serving, eval, and export subprocesses download artifacts. A client browser connecting over the network should paste the token into **Connection** only when it is allowed to manage credentials for that workstation.
