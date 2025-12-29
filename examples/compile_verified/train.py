#!/usr/bin/env python3
"""
Compile-Verified Training Example

Train a C++ code generation model with GCC verification.
This is the simplest RLVR example - just verify code compiles.

Usage:
    python examples/compile_verified/train.py
"""

import json
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from halo_forge.data.public_datasets import DatasetPreparer, get_dataset_spec
from halo_forge.sft.trainer import SFTTrainer, SFTConfig
from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
from halo_forge.rlvr.verifiers import GCCVerifier
from halo_forge.benchmark.pass_at_k import Benchmark


def main():
    # Paths
    data_dir = Path("data")
    model_dir = Path("models/compile_verified")
    results_dir = Path("results")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ====== Step 1: Prepare Data ======
    print("\n" + "=" * 60)
    print("STEP 1: PREPARE DATA")
    print("=" * 60)
    
    train_file = data_dir / "codeforces_cpp.jsonl"
    
    if not train_file.exists():
        print("Downloading CodeForces C++ dataset...")
        spec = get_dataset_spec("codeforces_cpp")
        preparer = DatasetPreparer(spec)
        preparer.prepare(str(train_file), template="qwen")
    else:
        print(f"Using existing data: {train_file}")
    
    # Count examples
    with open(train_file) as f:
        num_examples = sum(1 for _ in f)
    print(f"Total examples: {num_examples}")
    
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
    
    # ====== Step 3: Extract Prompts ======
    print("\n" + "=" * 60)
    print("STEP 3: EXTRACT PROMPTS")
    print("=" * 60)
    
    prompts_file = data_dir / "cpp_prompts.jsonl"
    
    if not prompts_file.exists():
        print("Extracting prompts from training data...")
        prompts = []
        
        with open(train_file) as f:
            for i, line in enumerate(f):
                if i >= 200:  # Use 200 prompts for RAFT
                    break
                    
                data = json.loads(line)
                text = data.get('text', '')
                
                # Extract user prompt from chat format
                if '<|im_start|>user' in text:
                    start = text.find('<|im_start|>user') + len('<|im_start|>user')
                    end = text.find('<|im_end|>', start)
                    if end > start:
                        prompt = text[start:end].strip()
                        prompts.append({'prompt': prompt})
        
        with open(prompts_file, 'w') as f:
            for p in prompts:
                f.write(json.dumps(p) + '\n')
        
        print(f"Extracted {len(prompts)} prompts")
    else:
        print(f"Using existing prompts: {prompts_file}")
    
    # ====== Step 4: RAFT Training ======
    print("\n" + "=" * 60)
    print("STEP 4: RAFT TRAINING")
    print("=" * 60)
    
    raft_output = model_dir / "raft"
    
    # Create GCC verifier
    verifier = GCCVerifier(max_workers=8)
    
    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt'])
    
    # RAFT config
    raft_config = RAFTConfig(
        sft_checkpoint=sft_checkpoint,
        output_dir=str(raft_output),
        num_cycles=3,
        samples_per_prompt=4,
        reward_threshold=0.5,
        keep_top_percent=0.5
    )
    
    # Train
    raft_trainer = RAFTTrainer(verifier=verifier, config=raft_config)
    final_model = raft_trainer.run(prompts, num_cycles=3)
    
    # ====== Step 5: Benchmark ======
    print("\n" + "=" * 60)
    print("STEP 5: BENCHMARK")
    print("=" * 60)
    
    benchmark = Benchmark(
        model_path=final_model,
        verifier=verifier,
        base_model="Qwen/Qwen2.5-Coder-7B"
    )
    
    result = benchmark.run(
        prompts=prompts[:50],  # Benchmark on subset
        samples_per_prompt=5,
        k_values=[1, 5],
        output_path=str(results_dir / "compile_verified_benchmark.json")
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final model: {final_model}")
    print(f"Pass rate: {result.pass_rate:.1%}")
    print(f"pass@1: {result.pass_at_k.get(1, 0):.1%}")
    print(f"pass@5: {result.pass_at_k.get(5, 0):.1%}")


if __name__ == "__main__":
    main()

