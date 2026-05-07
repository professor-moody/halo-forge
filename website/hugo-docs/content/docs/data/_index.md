---
title: "Data pipeline"
description: "Three operations close the gap between \"I have prompts\" and \"I have a training-ready dataset\":"
weight: 20
---


Three operations close the gap between "I have prompts" and "I have a training-ready dataset":

- **synthesize** (D1) — generate completions from seed prompts via a teacher model; verify and filter.
- **dedup** (D2) — drop literal + near-duplicates.
- **score** (D3) — filter by quality threshold or top-K%.

They compose: synthesize → dedup → score → filter is the full pre-finetune sequence (Distilabel + NeMo Curator + DEITA recipes folded into three commands).

## Synthesize

```bash
# SFT data: 1 completion per seed, kept if reward >= threshold
halo-forge data synthesize \
  --seeds prompts.txt --output sft_raw.jsonl \
  --teacher-model Qwen/Qwen2.5-3B-Instruct \
  --verifier execution --threshold 0.5

# Preference data: 4 completions per seed, best→chosen, worst→rejected
halo-forge data synthesize \
  --seeds prompts.jsonl --output pref_raw.jsonl \
  --teacher-model Qwen/Qwen2.5-3B-Instruct \
  --verifier llm_judge \
  --n-per-prompt 4 --kind preference
```

**The teacher** defaults to an OpenAI-compatible HTTP client targeting `http://127.0.0.1:8001/v1` — exactly what `halo-forge serve` exposes. So the implicit zero-config recipe is:

```bash
# Terminal 1
halo-forge serve --model Qwen/Qwen2.5-3B-Instruct

# Terminal 2
halo-forge data synthesize --seeds prompts.txt --output raw.jsonl --teacher-model X
```

Override the endpoint with `--base-url` and `--api-key` (or env vars `HALOFORGE_TEACHER_BASE_URL` / `HALOFORGE_TEACHER_API_KEY`) for hosted APIs or other local servers.

**The verifier** is any short name from the V1 plugin registry — `execution`, `llm_judge`, `bleu`, `json_schema`, `regex_format`, or anything you've registered yourself. The completion is scored by the verifier; rows with `reward >= --threshold` are kept.

**Output shape** depends on `--kind`:

- `sft` — `{"prompt": ..., "completion": ...}` per surviving completion.
- `preference` — `{"prompt": ..., "chosen": ..., "rejected": ..., "chosen_reward": ..., "rejected_reward": ...}` per group whose best completion clears the threshold AND beats the worst (no signal from tied groups). Requires `--n-per-prompt >= 2`.

**Seeds** can be a JSONL file (looks for `prompt` / `text` / `question` / `instruction` keys), a plain text file (one prompt per line), or a Python list when calling programmatically.

**Programmatic API.**
```python
from halo_forge.data.synthesize import synthesize_dataset

result = synthesize_dataset(
    seeds=["Explain X.", "Solve Y."],
    output_path="raw.jsonl",
    teacher=my_teacher_callable,    # any (prompt: str) -> str
    verifier_name="execution",
    n_per_prompt=4,
    output_kind="preference",
)
print(result.n_accepted, "/", result.n_generated, "kept")
```

A teacher exception on one prompt doesn't crash the run — it gets logged and the row is recorded with `rejected_reason="teacher_error"`.

## Dedup

```bash
halo-forge data dedup --input raw.jsonl --output deduped.jsonl --method exact
halo-forge data dedup --input raw.jsonl --output deduped.jsonl --method fuzzy --threshold 0.85
```

Two methods:

- **`exact`** — SHA-256 over a normalized (whitespace-collapsed, lowercased by default) version of the row's text. O(n). Removes literal duplicates including the most common cosmetic-only diffs. Always available.
- **`fuzzy`** — MinHash + LSH over word n-gram shingles (default n=5; matches FineWeb-Edu and NeMo Curator). Removes near-duplicates above a configurable Jaccard threshold (default 0.85). Sub-quadratic via the LSH index — scales to millions of rows. Requires `pip install datasketch`.

**Order is preserved**: the first occurrence wins; subsequent duplicates are dropped. This matches NeMo Curator's semantics so a halo-forge dedup run is reproducible against published recipes.

**Knobs.**
- `--key text` — when records are dicts, which field to dedup on.
- `--case-sensitive` — opt out of the lowercase-and-trim normalization.
- `--threshold 0.85` — Jaccard similarity above which fuzzy treats rows as duplicates.
- `--num-perm 128` — MinHash permutations (more = more precise estimate, more memory).
- `--shingle-n 5` — word n-gram size.

**Typical removal rates.** Exact dedup catches 5-15% on most public corpora. Fuzzy with 0.85 threshold typically catches another 10-30% of near-duplicates that exact missed. Together you can drop 30-40% of FineWeb-style data without losing diversity.

**Programmatic API.**
```python
from halo_forge.data.dedup import exact_dedup, fuzzy_dedup, dedup_file

result = exact_dedup(records, key="text")
print(result.kept_indices, result.removed_indices)
```

## Quality scoring

```bash
halo-forge data score --input deduped.jsonl --output clean.jsonl --threshold 0.5
halo-forge data score --input deduped.jsonl --output clean.jsonl --top-k-pct 0.5
```

Heuristic scorer composing five dependency-free signals:

- **`length`** — penalizes too-short and runaway-long completions; sweet spot 50-1500 chars.
- **`whitespace`** — fraction non-whitespace; catches blank rows.
- **`alpha_ratio`** — fraction alphabetic; catches punctuation noise.
- **`repetition`** — unique n-grams / total; catches stuck-loop generations ("the the the …").
- **`format`** — chat-shape / preference-shape / dict-record validity check.

A multiplicative penalty (×0.3) fires when any single component drops below 0.1, so a row that fully fails on one axis (blank, punctuation-only, fully repetitive) can't survive on the strength of its other axes.

**Two filter modes.**
- `--threshold 0.5` — composite score floor; rows below it are dropped.
- `--top-k-pct 0.5` — keep the top 50% by score; threshold is ignored. Useful when you want a fixed dataset size regardless of absolute quality.

**LLM-as-judge scoring.** For semantic quality (relevance, factuality, style) beyond what heuristics catch, the `score_with_judge` API takes any `(text) -> float` callable:

```python
from halo_forge.data.quality import score_records, score_with_judge

def my_judge(text: str) -> float:
    # call your local LLM, return [0, 1]
    ...

result = score_records(records, scorer=lambda r: score_with_judge(r, judge=my_judge))
```

The judge integrates naturally with the [V2 LLM-as-judge verifier](VERIFIERS.md#llm-as-judge): wrap the verifier's `verify()` reward field as the judge callable.

**Rejection diagnosis.** The result's `reasons` dict buckets rejections by which component scored lowest. Useful for "why is half my dataset getting filtered" — usually one axis (length, alpha_ratio, repetition) is the dominant cause.

## Composing dedup + score

```bash
# Step 1: dedup
halo-forge data dedup --input raw.jsonl --output deduped.jsonl --method fuzzy --threshold 0.85

# Step 2: filter to top 50% by quality
halo-forge data score --input deduped.jsonl --output train.jsonl --top-k-pct 0.5

# Step 3: train
halo-forge sft train --data train.jsonl --model Qwen/Qwen2.5-Coder-3B
```

DEITA et al. show top-K%-by-quality consistently beats training on the full dataset *at any size*. The standard recipe is dedup → score → keep top-30% to top-50%.

## Format conventions

Halo-forge's data tools accept three record shapes:

1. **Plain string** — treated as the text. Wrapped to `{"text": s}` on write.
2. **Dict with `text`** — the canonical SFT shape.
3. **Dict with `prompt` / `chosen` / `rejected`** — preference data (DPO).
4. **Dict with `messages: [...]`** — chat shape (used by UltraFeedback, Llama-style training).

The dedup hash and quality scorer extract the appropriate field automatically (`text` → `completion` → `chosen` → `prompt` fallback chain).

## Composing all three

```bash
# Step 0: get a teacher running (or skip to step 1 with --base-url pointing elsewhere)
halo-forge serve --model Qwen/Qwen2.5-3B-Instruct &

# Step 1: synthesize
halo-forge data synthesize --seeds prompts.txt --output raw.jsonl \
  --teacher-model Qwen/Qwen2.5-3B-Instruct --verifier execution --threshold 0.5

# Step 2: dedup
halo-forge data dedup --input raw.jsonl --output deduped.jsonl --method fuzzy

# Step 3: filter to top 50% by heuristic quality
halo-forge data score --input deduped.jsonl --output train.jsonl --top-k-pct 0.5

# Step 4: train
halo-forge sft train --data train.jsonl --model Qwen/Qwen2.5-Coder-3B
```

## Roadmap

- **D4 — dataset versioning + lineage**: content-addressed datasets, runs link to the exact version trained on.
- **D5 — web-source ingest helpers**: "pull 10k Python repos with permissive licenses", "the FineWeb-Edu subset for X".
- **Semantic dedup (SemDeDup)**: sentence-embedding clustering for near-duplicates that fuzzy MinHash misses. Needs an embedding-model dep that deserves its own opt-in.
