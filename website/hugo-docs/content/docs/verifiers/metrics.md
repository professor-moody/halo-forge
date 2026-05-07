---
title: "Reference-metric verifiers"
description: "BLEU / ROUGE / chrF — score candidates against a reference string with the standard MT / summarization metrics."
weight: 40
---

Wraps the standard MT / summarization scoring metrics behind the halo-forge `Verifier` interface. The trainer scores candidates against a reference string without writing the bridge per-task.

All three normalize their score into reward ∈ `[0.0, 1.0]` so RAFT / GRPO can mix them with execution-based rewards in a chained verifier without rescaling math.

## bleu

SacreBLEU — the standard MT score, comparable across papers because tokenization is fixed.

```python
from halo_forge.rlvr.verifiers import get_verifier

v = get_verifier("bleu")(
    references="The quick brown fox jumps over the lazy dog.",
    success_threshold=0.3,  # BLEU/100 above which success=True
)
v.verify("The quick brown fox jumps over the dog.")
```

Multi-reference is supported (translation tasks where several valid translations exist):

```python
v = get_verifier("bleu")(
    references=[
        "The quick brown fox.",
        "A quick brown fox.",
    ],
)
```

Requires `pip install sacrebleu`.

## rouge

google-research's `rouge_score` — the standard summarization metric. ROUGE-L F-measure by default; ROUGE-1 and ROUGE-2 also available.

```python
v = get_verifier("rouge")(
    reference="A summary that captures the main points concisely.",
    rouge_type="rougeL",  # or "rouge1", "rouge2"
)
v.verify("Summary capturing main points concisely.")
```

Requires `pip install rouge_score`.

## chrf

Character n-gram F-score via SacreBLEU. More robust than BLEU on morphologically rich languages and increasingly the recommended MT baseline.

```python
v = get_verifier("chrf")(
    references="The quick brown fox jumps over the lazy dog.",
    word_order=2,  # chrF++ (default); 0 = plain chrF
)
```

Requires `pip install sacrebleu` (same package as BLEU).

## When to pick which

| Task | Recommended | Why |
|---|---|---|
| Translation | `chrf` | Modern MT baseline; robust to morphology |
| Summarization | `rouge` | Standard summarization metric |
| Paraphrase / generation | `bleu` | Most papers report BLEU; keep comparable |
| Strictly formatted output | `regex_format` (V3) | Reference-metric won't catch structure |
| Tool calls / JSON | `json_schema` (V3) | Same — structure first |

## Composing

Metrics compose with execution-based verifiers via `ChainedVerifier` — useful for "code that compiles AND matches a reference solution":

```python
from halo_forge.rlvr.verifiers import (
    ChainedVerifier,
    GCCVerifier,
    BLEUVerifier,
)

v = ChainedVerifier([
    GCCVerifier(),                          # 0.5 if it compiles
    BLEUVerifier(references=ref_solution),  # 0.5 if it's close to the reference
])
```

## Lazy imports

All three metric packages (`sacrebleu`, `rouge_score`) are lazy-imported. The verifier modules load on installs without those deps; the verifier's first `.verify()` call surfaces a clean `ImportError` pointing at the right `pip install` command.

Empty candidates short-circuit before lazy import so they fail with `error="empty_response"` regardless of dep availability.
