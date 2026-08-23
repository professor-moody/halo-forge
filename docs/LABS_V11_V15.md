# Halo Forge Labs V11–V15

Halo Forge now carries a model from owned data through outcome validation,
controlled adaptation research, grounded data creation, specialized task-model
training, and deterministic agent-environment evaluation.

## V11: outcome validation

After a guided proof run, choose **Assess proof outcome**. Halo Forge separates
technical evidence, quality evidence, and diagnostics. A full run requires a
compatible completed assessment. Continuing without that evidence requires a
reason that remains in the run lineage.

```bash
halo-forge outcome assess RUN_ID
halo-forge outcome show ASSESSMENT_ID
halo-forge outcome findings ASSESSMENT_ID
halo-forge outcome full-run-context RUN_ID --assessment ASSESSMENT_ID
```

## V12: adaptation studies

Experiments → **Studies** supports paired A/B, dose response, and a bounded 2×2
factorial design. The guided default is three paired seeds: 17, 42, and 101.
Domain uptake and general-capability retention remain separate. Protected
evidence cannot guide study selection.

```bash
halo-forge study create --spec study.json
halo-forge study revise STUDY_ID --spec protocol.json
halo-forge study materialize PROTOCOL_REVISION_ID
halo-forge study analyze PROTOCOL_REVISION_ID --spec observations.json
```

## V13: grounded corpus-to-dataset

From a completed corpus Dataset Version choose **Create grounded data**. Every
candidate keeps exact document, source-span, source-hash, teacher, and verifier
identity. Generation produces suggestions only; review, publication, dataset
or suite creation, and training remain separate actions.

```bash
halo-forge data ground profile create --spec profile.json
halo-forge data ground profile revise PROFILE_ID --spec revision.json
halo-forge data ground generate REVISION_ID --spec request.json
halo-forge data ground candidates BATCH_ID
halo-forge data ground review-proposal BATCH_ID
```

## V14: specialized task models

Guided Own Data includes single-label and multi-label classification,
embedding pairs, reranking, image classification, and audio classification.
The verified PyTorch trainers write Hugging Face-compatible model/processor
artifacts and the standard Halo Forge training summary.

```bash
halo-forge classify train --model distilbert/distilbert-base-uncased \
  --dataset train.jsonl --output runs/classifier
halo-forge embed train --model sentence-transformers/all-MiniLM-L6-v2 \
  --dataset pairs.jsonl --output runs/embedder
halo-forge rerank train --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --dataset relevance.jsonl --output runs/reranker
```

## V15: agent environments

Evaluate → **Environments** creates deterministic local state-machine
environments with content-addressed fixtures and no external writes. Trace
replay repeats recorded actions exactly. Reviewed completed episodes can be
published as immutable trajectory sets for existing SFT, preference,
reasoning, tool-correction, or RLVR adapters.

```bash
halo-forge env create --spec environment.json
halo-forge env revise ENVIRONMENT_ID --spec revision.json
halo-forge env run --spec episode.json
halo-forge env replay EPISODE_ID
halo-forge env trajectory publish --spec trajectories.json
```

## Storage and compatibility

Schema v18 is additive and replay v9 remains backward readable. New bundles
live under:

```text
~/.halo-forge/evaluations/outcomes/
~/.halo-forge/studies/
~/.halo-forge/grounding/
~/.halo-forge/environments/
~/.halo-forge/episodes/
~/.halo-forge/trajectories/
```
