#!/usr/bin/env python3
"""
Test-Verified Training Example

Train a Python code generation model with pytest verification.
The model learns to generate code that passes tests.

Usage:
    python examples/test_verified/train.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from halo_forge.data.public_datasets import DatasetPreparer, get_dataset_spec
from halo_forge.sft.trainer import SFTTrainer, SFTConfig
from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
from halo_forge.rlvr.verifiers import PytestVerifier
from halo_forge.benchmark.pass_at_k import Benchmark


def create_test_prompts():
    """Create prompts that include test cases."""
    prompts = [
        {
            "prompt": """Write a Python function `factorial(n)` that returns the factorial of n.

Include tests:
```python
def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
```"""
        },
        {
            "prompt": """Write a Python function `is_palindrome(s)` that checks if a string is a palindrome.

Include tests:
```python
def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True
```"""
        },
        {
            "prompt": """Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array.

Include tests:
```python
def test_binary_search():
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3], 4) == -1
    assert binary_search([], 1) == -1
```"""
        },
        {
            "prompt": """Write a Python function `merge_sorted(a, b)` that merges two sorted lists.

Include tests:
```python
def test_merge_sorted():
    assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]
    assert merge_sorted([], [1]) == [1]
    assert merge_sorted([1], []) == [1]
```"""
        },
        {
            "prompt": """Write a Python function `count_vowels(s)` that counts vowels in a string.

Include tests:
```python
def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("xyz") == 0
    assert count_vowels("AEIOUaeiou") == 10
```"""
        }
    ]
    return prompts


def main():
    # Paths
    data_dir = Path("data")
    model_dir = Path("models/test_verified")
    results_dir = Path("results")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ====== Step 1: Prepare Data ======
    print("\n" + "=" * 60)
    print("STEP 1: PREPARE DATA")
    print("=" * 60)
    
    train_file = data_dir / "mbpp.jsonl"
    
    if not train_file.exists():
        print("Downloading MBPP dataset...")
        spec = get_dataset_spec("mbpp")
        preparer = DatasetPreparer(spec)
        preparer.prepare(str(train_file), template="qwen")
    else:
        print(f"Using existing data: {train_file}")
    
    # ====== Step 2: SFT Training ======
    print("\n" + "=" * 60)
    print("STEP 2: SFT TRAINING")
    print("=" * 60)
    
    sft_output = model_dir / "sft"
    
    if not (sft_output / "final_model").exists():
        sft_config = SFTConfig(
            model_name="Qwen/Qwen2.5-Coder-7B",
            train_file=str(train_file),
            output_dir=str(sft_output),
            num_epochs=2,
            batch_size=2,
            gradient_accumulation_steps=16,
            max_seq_length=2048
        )
        
        trainer = SFTTrainer(sft_config)
        sft_checkpoint = trainer.train()
    else:
        print(f"Using existing SFT model: {sft_output}/final_model")
        sft_checkpoint = str(sft_output / "final_model")
    
    # ====== Step 3: Create Test Prompts ======
    print("\n" + "=" * 60)
    print("STEP 3: CREATE TEST PROMPTS")
    print("=" * 60)
    
    prompts = create_test_prompts()
    prompts_file = data_dir / "python_test_prompts.jsonl"
    
    with open(prompts_file, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')
    
    print(f"Created {len(prompts)} test prompts")
    
    # ====== Step 4: RAFT Training ======
    print("\n" + "=" * 60)
    print("STEP 4: RAFT TRAINING")
    print("=" * 60)
    
    raft_output = model_dir / "raft"
    
    # Create pytest verifier
    verifier = PytestVerifier(
        timeout=60,
        max_workers=4
    )
    
    # RAFT config
    raft_config = RAFTConfig(
        sft_checkpoint=sft_checkpoint,
        output_dir=str(raft_output),
        num_cycles=3,
        samples_per_prompt=8,
        reward_threshold=0.5,
        keep_top_percent=0.5,
        system_prompt="You are an expert Python programmer. Write clean code that passes all tests."
    )
    
    # Train
    prompt_texts = [p['prompt'] for p in prompts]
    raft_trainer = RAFTTrainer(verifier=verifier, config=raft_config)
    final_model = raft_trainer.run(prompt_texts, num_cycles=3)
    
    # ====== Step 5: Benchmark ======
    print("\n" + "=" * 60)
    print("STEP 5: BENCHMARK")
    print("=" * 60)
    
    benchmark = Benchmark(
        model_path=final_model,
        verifier=verifier,
        base_model="Qwen/Qwen2.5-Coder-7B",
        system_prompt="You are an expert Python programmer."
    )
    
    result = benchmark.run(
        prompts=prompt_texts,
        samples_per_prompt=10,
        k_values=[1, 5, 10],
        output_path=str(results_dir / "test_verified_benchmark.json")
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final model: {final_model}")
    print(f"Pass rate: {result.pass_rate:.1%}")
    for k, rate in result.pass_at_k.items():
        print(f"pass@{k}: {rate:.1%}")


if __name__ == "__main__":
    main()

