# Serving + conversion

Halo-forge ships three commands that close the train → ship loop without leaving the local machine:

| Command | What it does |
|---|---|
| `halo-forge serve` | OpenAI-compatible HTTP endpoint for any trained model. |
| `halo-forge convert` | HF ↔ MLX ↔ GGUF format conversion with normalized quant vocabulary. |
| `halo-forge merge` | Merge LoRA adapters into the base, or combine multiple adapters. |

The three compose: `merge → convert --verify → serve`.

## Serve

```bash
halo-forge serve --model ./models/sft/final_model
halo-forge serve --model mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx
halo-forge serve --model X --host 0.0.0.0 --port 8080
```

Spins up a FastAPI server on `127.0.0.1:8001` (overridable). Endpoints:

- `POST /v1/chat/completions` — OpenAI v1 chat-completions surface.
- `POST /v1/completions` — OpenAI v1 text-completions surface.
- `GET  /v1/models` — list the served model.
- `GET  /health` — liveness + identity check.

Drop-in for any client expecting an OpenAI endpoint. Point your `OPENAI_BASE_URL` at `http://127.0.0.1:8001/v1` and the same code that targets `api.openai.com` works unchanged.

**Sampling.** Standard knobs supported in v1: `temperature`, `top_p`, `max_tokens`, `stop`. Validation is bounded — `temperature ∈ [0, 2]`, `top_p ∈ (0, 1]`, `max_tokens ∈ [1, 8192]`.

**Backend dispatch.**
- `--backend mlx` — routes through `MLXInferenceAdapter`. Apple Silicon native.
- otherwise — PyTorch via `transformers.AutoModelForCausalLM` on the active backend (rocm/cuda/mps/cpu).

**Stop strings.** Honored at the FastAPI layer, post-truncating the adapter's output. Both backends share one implementation.

**Chat-template rendering.** The `/v1/chat/completions` endpoint prefers the model tokenizer's native chat template; falls back to ChatML for adapters that don't expose one (the MLX path keeps its tokenizer private).

**Streaming.** Set `stream: true` on chat or text completions to receive
OpenAI-compatible `text/event-stream` chunks ending in `data: [DONE]`. The v1
implementation streams the adapter result through the OpenAI wire shape; backend
native token streaming remains a future optimization.

**Not yet shipped (Track I-followups):**
- Continuous batching — arrives with I2 (vLLM gives this for free on CUDA; needs explicit wiring for MLX/llama.cpp).
- Speculative decoding with a draft model — opt-in future work after streaming.
- Embeddings, function-calling, vision endpoints — separate items.

## Convert

```bash
halo-forge convert --source Qwen/Qwen2.5-3B-Instruct --format mlx --quant q4 --output ./mlx-q4
halo-forge convert --source ./models/sft/final_model --format gguf --quant q4 --output ./out.gguf
halo-forge convert --source X --format hf --quant bf16 --output ./bf16/      # dtype recast
halo-forge convert --list                                                    # show formats + quants
```

Three target formats, one normalized vocabulary:

| Format | Use for | Quant options |
|---|---|---|
| `mlx` | Apple Silicon serving | `q4`, `q8`, `fp16`, `bf16`, `fp32` |
| `gguf` | llama.cpp / Ollama | `q4`, `q8`, `fp16`, `bf16`, `fp32` |
| `hf`   | dtype recast (bf16 ↔ fp16 ↔ fp32) | `bf16`, `fp16`, `fp32` |

The quant vocabulary is normalized — `--quant q4` means "4-bit affine, group size 64" regardless of which target you pick. The dispatcher translates to each tool's underlying argument shape (`q4_k_m` for GGUF, `q_bits=4` for MLX, etc.). Unsupported (format, quant) pairs raise typed errors listing the supported quants for the requested format.

**Round-trip verify.** Add `--verify` to run a fixed prompt set through both source and exported and flag drift:

```bash
halo-forge convert --source X --format mlx --quant q4 --output ./out --verify
```

Three drift signals:
- **Exact match rate** — fraction matching character-for-character.
- **Avg char overlap** — character-level Jaccard, robust to small numerical-precision drift.
- **First-token match rate** — single best signal for "exported model immediately disagrees".

GGUF round-trip verification is a typed `NotImplementedError` in v1 because halo-forge's serving adapter doesn't load GGUF directly yet. `halo-forge convert --verify --format gguf` exits non-zero rather than treating the skipped check as success. Use llama.cpp or Ollama as the GGUF-side loader and the `compare_generation` programmatic API as the bring-your-own-callable surface.

## Merge

```bash
# Single LoRA into base — output is a standard HF checkpoint
halo-forge merge --mode bake \
  --base Qwen/Qwen2.5-3B-Instruct --adapter ./my-lora --output ./shipped

# N adapters combined via TIES + DARE (current best)
halo-forge merge --mode combine \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapters lora_a,lora_b,lora_c --weights 0.5,0.3,0.2 \
  --method dare_ties --output ./blended

# Combine + bake in one step
halo-forge merge --mode combine ... --bake-after-merge --output ./shipped
```

**Bake** (`--mode bake`) merges a single LoRA into its base via peft's `merge_and_unload`. The output is a standard HuggingFace checkpoint loadable through `AutoModelForCausalLM.from_pretrained` — no LoRA infrastructure required at inference time.

**Combine** (`--mode combine`) merges N LoRAs through peft's `add_weighted_adapter` under a normalized method vocabulary:

| Method | What it does |
|---|---|
| `linear` | Straight weighted sum |
| `ties` | Tang et al. 2023 — resolves sign conflicts + keeps top-k magnitudes |
| `dare_linear` | DARE pruning + linear |
| `dare_ties` | DARE + TIES (current best general-purpose; default) |
| `magnitude_prune` | Drop the smallest-magnitude deltas |

Add `--bake-after-merge` to combine + merge into the base in one step. Output is a single merged checkpoint instead of an adapter directory.

## Programmatic API

```python
from halo_forge.serving.app import create_serving_app
from halo_forge.inference.convert import convert
from halo_forge.inference.merge import merge

app = create_serving_app(model_name="X", backend_name="mlx")
# uvicorn.run(app, host="127.0.0.1", port=8001)

convert_result = convert(
    source="Qwen/Qwen2.5-3B-Instruct",
    output_path="./out",
    target_format="mlx",
    quantization="q4",
)

merge_result = merge(
    operation="bake",
    base_model="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="./my-lora",
    output_path="./shipped",
)
```

## Roadmap

- **I2 — continuous batching** for the served endpoint on MLX / llama.cpp (CUDA gets it free via vLLM).
- **I3 follow-up — speculative decoding** with a draft model. Disabled by default until measured.
- **GGUF round-trip verify** — needs a GGUF-loading serving adapter; currently typed-NotImplementedError.
