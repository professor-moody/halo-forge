# Strix Halo Configuration Alignment

This document summarizes the alignment work done to ensure halo forge configurations match the production-tested settings from `strix-edr-training`.

---

## Critical Settings Aligned

### 1. Dataloader Configuration (Strix Halo Unified Memory)

**Required settings for stable training:**

```yaml
training:
  dataloader_num_workers: 0    # MUST be 0
  dataloader_pin_memory: false # MUST be false
```

**Why**: Strix Halo uses unified memory architecture where CPU and GPU share the same physical RAM. The standard PyTorch data loading optimizations (pinned memory, multiple workers) cause issues with this architecture.

**Files updated:**
- `halo_forge/rlvr/raft_trainer.py` - Added to TrainingArguments
- `configs/raft_example.yaml` - Documented with comments
- `configs/sft_example.yaml` - Documented with comments

---

### 2. Gradient Checkpointing

**Settings:**

```python
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False}
```

**Why**: The `use_reentrant=False` setting avoids warnings about inputs not requiring gradients and provides cleaner checkpointing behavior.

**Files updated:**
- `halo_forge/rlvr/raft_trainer.py`
- `halo_forge/benchmark/runner.py`
- `halo_forge/sft/trainer.py`

---

### 3. Device Mapping

**Changed from:**
```python
device_map="cpu",  # Load on CPU first
# ... then manually move to GPU
self.model = self.model.to('cuda')
```

**Changed to:**
```python
device_map="auto",  # Unified memory handles this optimally
```

**Why**: With Strix Halo's unified memory, there's no benefit to loading on CPU first. The automatic device mapping is cleaner and faster.

**Files updated:**
- `halo_forge/sft/trainer.py`

---

### 4. Trainer API Deprecation

**Changed from:**
```python
trainer = Trainer(
    ...,
    tokenizer=self.tokenizer,  # Deprecated
)
```

**Changed to:**
```python
trainer = Trainer(
    ...,
    processing_class=self.tokenizer,  # New API
)
```

**Files updated:**
- `halo_forge/rlvr/raft_trainer.py`
- `halo_forge/benchmark/runner.py`
- `halo_forge/sft/trainer.py`

---

### 5. Chunked Verification

**Added memory-efficient verification:**

```python
# CHUNKED verification to prevent memory exhaustion
chunk_size = 200
results = []

for i in range(0, len(completions), chunk_size):
    chunk_results = self.verifier.verify_batch(chunk_completions[i:i+chunk_size])
    results.extend(chunk_results)
    gc.collect()  # Force garbage collection
```

**Why**: Large batches (500+ samples) can exhaust memory during verification. Chunking with explicit garbage collection prevents OOM.

**Files updated:**
- `halo_forge/rlvr/raft_trainer.py`

---

### 6. Memory Cleanup on Cache Resume

**Added cleanup when loading from verification cache:**

```python
# FREE MEMORY: samples not needed when loading from verified cache
try:
    del samples
except NameError:
    pass
del kept_indices
gc.collect()
```

**Why**: When resuming a cycle from cached verification results, the generated samples are still in memory. Explicitly freeing them prevents OOM during training.

**Files updated:**
- `halo_forge/rlvr/raft_trainer.py`

---

## Configuration Comparison

| Setting | strix-edr-training | halo forge (after alignment) |
|---------|-------------------|-------------------------------|
| `bf16` | true | true |
| `gradient_checkpointing` | true | true |
| `attn_implementation` | "eager" | "eager" |
| `dataloader_num_workers` | (implicit 0) | 0 (explicit) |
| `dataloader_pin_memory` | (implicit false) | false (explicit) |
| `device_map` | "auto" | "auto" |
| `learning_rate` (RAFT) | 5e-5 | 5e-5 |
| `gradient_accumulation_steps` | 16 | 16 |
| `batch_size` | 2 | 2 |
| `samples_per_prompt` | 8 | 8 |
| `max_new_tokens` | 1024 | 1024 |

---

## Remaining Differences (Intentional)

### 1. System Prompt

**strix-edr-training:**
```
"You are an expert C++ programmer specializing in Windows malware development and EDR evasion techniques."
```

**halo forge:**
```
"You are an expert programmer."
```

**Reason**: halo forge is domain-agnostic. Users configure their own system prompts.

### 2. Verifier Type

**strix-edr-training**: `JointVerifier` (compile + ThreatCheck)

**halo-forge**: Pluggable verifiers (GCC, MinGW, MSVC, pytest, etc.)

**Reason**: halo forge supports multiple domains beyond EDR evasion.

---

## Changelog

- **2025-01-02**: Initial alignment completed
  - Added dataloader settings to RAFT trainer
  - Updated device_map to "auto" in SFT trainer
  - Fixed deprecation warnings in all trainers
  - Added chunked verification with garbage collection
  - Added reward distribution tracking

