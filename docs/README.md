# halo forge Documentation

Full documentation is available at **[halo-forge.io/docs](https://halo-forge.io/docs)**

## Quick Links

| Section | Description |
|---------|-------------|
| [Quick Start](https://halo-forge.io/docs/getting-started/quickstart/) | Get running in 30 minutes |
| [How to Train](https://halo-forge.io/docs/training-pipeline/how-to-train/) | Complete step-by-step guide |
| [Command Index](https://halo-forge.io/docs/reference/command-index/) | Every command and flag |
| [Verifiers](https://halo-forge.io/docs/verifiers/) | Verification options |
| [Experimental](https://halo-forge.io/docs/experimental/) | VLM, Audio, Reasoning, Agentic |
| [Contributing](https://halo-forge.io/docs/contributing/) | How to contribute |

## Local Documentation

These files contain additional detail or unique content:

| Document | Description |
|----------|-------------|
| [VERIFIERS.md](VERIFIERS.md) | Verifier guide with safety considerations |
| [MODELS.md](MODELS.md) | Supported models reference |
| [HARDWARE_NOTES.md](HARDWARE_NOTES.md) | AMD Strix Halo configuration details |
| [GGUF_EXPORT.md](GGUF_EXPORT.md) | GGUF export guide for llama.cpp/Ollama |

## Experimental Configs

Experimental learning rate configurations are in [`configs/experimental/`](../configs/experimental/):

| Config | Description |
|--------|-------------|
| `raft_aggressive_decay.yaml` | Aggressive LR decay (0.7 factor) |
| `raft_constant_lr.yaml` | Constant LR baseline |
| `raft_decay_lr.yaml` | Standard LR decay (0.85 factor) |
