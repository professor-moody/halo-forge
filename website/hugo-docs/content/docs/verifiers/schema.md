---
title: "Schema verifiers"
description: "JSON-structure / JSON-schema / regex-format verifiers for tool-calling, structured output, and format-discipline finetunes."
weight: 30
---

Three verifiers for outputs that have a *structural* contract rather than a behavioral / programmatic one. All three plug into the V1 plugin registry — pick by short name from CLI / config / API.

## json_structure

Pass if the candidate parses as JSON. Tolerant of leading / trailing whitespace and code-fence wrapping (` ```json ... ``` `) — common artifacts of chat-model output that we don't want to penalize when the actual JSON inside is valid.

```python
from halo_forge.rlvr.verifiers import get_verifier

v = get_verifier("json_structure")()
result = v.verify('{"answer": 42}')
# success=True, reward=1.0
```

## json_schema

Pass if the candidate is JSON *and* validates against a [JSON Schema](https://json-schema.org/). Tool-calling and structured-output workflows are the canonical use cases.

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}
v = get_verifier("json_schema")(schema=schema)
v.verify('{"name": "Alice", "age": 30}')   # reward=1.0
v.verify('{"age": 30}')                     # reward=0.5 (partial credit by default)
```

`partial_credit=False` flips the failure case to `reward=0.0` — useful when you want a strict "valid or fail" signal for production training. Defaults to `True` so RAFT-style algorithms get gradient signal from "valid JSON, wrong shape" rows.

Requires `pip install jsonschema`. Lazy-imported — the verifier loads on installs without it; the failure path surfaces a clean install hint.

## regex_format

Pass if the candidate matches a configured regex. Useful for "the answer must look like `Final answer: <number>`" or similar format-discipline rules.

```python
v = get_verifier("regex_format")(pattern=r"Final answer:\s*(\d+)")
v.verify("Some reasoning... Final answer: 42")   # reward=1.0
```

Two semantics:

| Mode | What it requires |
|---|---|
| `re.search` (default) | Pattern matches *somewhere* in the candidate |
| `re.fullmatch` (`full_match=True`) | Entire candidate matches the pattern |

`partial_credit=True` under `full_match` mode gives a `reward=0.5` for "pattern present but not full-match" — a useful gradient signal when you're training a model toward a strict output format.

The regex is compiled once at constructor time so per-call verification is fast.

## Composability

Schema verifiers compose well with reference-metric verifiers (e.g. JSON-schema validates structure; BLEU validates content) via halo-forge's `ChainedVerifier`:

```python
from halo_forge.rlvr.verifiers import (
    ChainedVerifier,
    JSONSchemaVerifier,
    BLEUVerifier,
)

v = ChainedVerifier([
    JSONSchemaVerifier(schema=tool_call_schema),
    BLEUVerifier(references=expected_args_text, success_threshold=0.5),
])
```
