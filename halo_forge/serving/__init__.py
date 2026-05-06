"""OpenAI-compatible serving (Track I1).

Spins up a FastAPI server that accepts ``/v1/chat/completions``,
``/v1/completions``, and ``/v1/models`` calls and routes generation
through the active backend (PyTorch / MPS / MLX / CPU). Drop-in for
any client expecting an OpenAI-shaped endpoint — point it at
``http://127.0.0.1:8001/v1`` and the same code that hits
``api.openai.com`` works unchanged.

v1 scope intentionally narrow:
  - non-streaming chat + completions
  - single-model serving (start one server per model)
  - basic sampling knobs (temperature, top_p, max_tokens)

Deliberately deferred:
  - streaming (SSE) — Track I3 lands alongside speculative decoding
  - continuous batching — Track I6
  - embeddings / function-calling / vision — separate items
  - multi-model serving — defer until F-J model registry lands
"""

from halo_forge.serving.adapter import ServingAdapter, build_serving_adapter
from halo_forge.serving.app import create_serving_app

__all__ = ["ServingAdapter", "build_serving_adapter", "create_serving_app"]
