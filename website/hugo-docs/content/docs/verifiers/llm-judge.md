---
title: "LLM-as-judge verifier"
description: "Rubric-graded verifier for outputs without a programmatic ground truth. Pluggable judge model — local or hosted."
weight: 50
---

Rubric-graded verifier for outputs that don't have a programmatic ground truth — creative writing, summarization, open-ended dialog, anything where execution-based or schema-based verification breaks down.

```python
from halo_forge.rlvr.verifiers import get_verifier

v = get_verifier("llm_judge")(
    rubric="Score the response on conciseness and factual correctness.",
    scoring_scale=5,
    judge_model="meta-llama/Llama-3.2-3B-Instruct",
    prompt="Explain attention.",
)
result = v.verify("Attention is the mechanism that lets a model focus...")
# success=True (score >= midpoint), reward in [0.0, 1.0]
```

## How scoring works

1. The verifier builds a judge prompt from `rubric + prompt + candidate response`.
2. The judge generates a single integer in `[1, scoring_scale]`.
3. Halo-forge maps that score to a reward in `[0.0, 1.0]` so the RAFT / GRPO trainers can mix it with execution-based rewards.

A 5/5 maps to `reward=1.0`. 1/5 maps to `0.0`. Success (the boolean field) flips at the midpoint — score ≥ `(scale+1)/2` is success.

## Three knobs

| Knob | Default | What it controls |
|---|---|---|
| `rubric` | "Score the candidate response on overall quality, taking into account correctness, helpfulness, and clarity." | Free-text rubric the judge follows |
| `scoring_scale` | `5` | Top of the integer scale (must be ≥ 2). 1-5 Likert is canonical; 1-10 also common |
| `judge_callable` | OpenAI-compatible HTTP | Pluggable. Replace with any `(prompt: str) -> str` |

## Pluggable judge

The default judge is an OpenAI-compatible HTTP client targeting `http://127.0.0.1:8001/v1` — exactly what `halo-forge serve` exposes. So the implicit zero-config recipe is:

```bash
# Terminal 1: run a teacher / judge
halo-forge serve --model meta-llama/Llama-3.2-3B-Instruct

# Terminal 2: train with the judge as verifier
halo-forge grpo train \
  --data prompts.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --verifier llm_judge \
  --num-generations 8
```

Override the endpoint:

| | Where |
|---|---|
| `--base-url` | constructor / CLI arg |
| `HALOFORGE_JUDGE_BASE_URL` | env var |
| `HALOFORGE_JUDGE_API_KEY` | env var (bearer token) |

For a fully custom judge (vLLM with a specific sampling configuration, hosted API with custom auth, regression model that scores directly without prompting):

```python
def my_judge(prompt: str) -> str:
    # Call your scoring infra; return the score as a string.
    return str(score_with_my_model(prompt))

v = get_verifier("llm_judge")(
    rubric="...",
    judge_callable=my_judge,
)
```

## Defensive against noisy output

Real judges return `"4"`, `"Score: 4"`, `"4/5"`, full sentences. The score parser scans for the first in-range integer:

```
"4"                                    → 4
"Score: 4"                             → 4
"4/5"                                  → 4
"I would rate this 3 out of 5."        → 3
"100 is too high; my pick is 4"        → 4
"I cannot evaluate this"               → None → unparseable_score error
```

Hopeless / errored / empty responses fail with typed error codes (`empty_response` / `unparseable_score` / `judge_failure`) rather than fabricating a reward.

## cDPO label smoothing

`label_smoothing` ∈ `[0, 1]` softens the judgment so a confident-correct pair doesn't drive loss to ~0. Useful when the dataset has label noise.

## Composing

LLM-judge composes with structural verifiers — chain `json_schema` first to gate on shape, then `llm_judge` on the content.

## Reliability concerns

LLM-as-judge is *informative but stochastic*. Two passes through the same judge on the same prompt-response pair can disagree.

- For training (RAFT / GRPO): the noise tends to average out across the group; the algorithm tolerates it.
- For evaluation: pair `llm_judge` with the V7 judge-reliability harness (roadmap) to measure the judge's self-agreement before drawing conclusions.

## Roadmap

- **V7 judge reliability harness** — measure judge agreement vs human labels; flag judges that disagree with themselves.
- **Multi-rater consensus** — invoke N judges, return the median or majority score.
- **Constrained decoding** — force the judge to emit `[1, scale]` integers via the OpenAI-compatible structured-output mode where available.
