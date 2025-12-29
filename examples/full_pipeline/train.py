#!/usr/bin/env python3
"""
Full Pipeline Example

Complete training pipeline from data to benchmark.

Usage:
    python train.py                # Run all steps
    python train.py --step data    # Just data prep
    python train.py --step sft     # Just SFT
    python train.py --step raft    # Just RAFT
    python train.py --step bench   # Just benchmark
"""

import argparse
import json
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def step_data(config):
    """Prepare training data."""
    from halo_forge.data.public_datasets import DatasetPreparer, get_dataset_spec
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    data_dir = Path(config['data']['output_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    datasets = config['data']['datasets']
    output_files = []
    
    for ds_name in datasets:
        output_file = data_dir / f"{ds_name}.jsonl"
        
        if not output_file.exists():
            print(f"\nDownloading {ds_name}...")
            spec = get_dataset_spec(ds_name)
            preparer = DatasetPreparer(spec)
            preparer.prepare(str(output_file), template=config['data']['template'])
        else:
            print(f"\nUsing existing: {output_file}")
        
        output_files.append(output_file)
    
    # Combine datasets
    combined_file = data_dir / "train.jsonl"
    
    print(f"\nCombining to {combined_file}...")
    total = 0
    with open(combined_file, 'w') as out:
        for f in output_files:
            with open(f) as inp:
                for line in inp:
                    out.write(line)
                    total += 1
    
    print(f"Total training examples: {total}")
    
    # Create prompts file for RAFT
    prompts_file = data_dir / "prompts.jsonl"
    
    print(f"\nExtracting prompts to {prompts_file}...")
    prompts = []
    
    with open(combined_file) as f:
        for i, line in enumerate(f):
            if i >= config['raft']['max_prompts']:
                break
            
            data = json.loads(line)
            text = data.get('text', '')
            
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
    
    return combined_file, prompts_file


def step_sft(config, train_file):
    """Run SFT training."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    print("\n" + "=" * 60)
    print("SFT TRAINING")
    print("=" * 60)
    
    output_dir = Path(config['sft']['output_dir'])
    final_model = output_dir / "final_model"
    
    if final_model.exists():
        print(f"\nUsing existing SFT model: {final_model}")
        return str(final_model)
    
    sft_config = SFTConfig(
        model_name=config['model']['name'],
        train_file=str(train_file),
        output_dir=str(output_dir),
        num_epochs=config['sft']['num_epochs'],
        batch_size=config['sft']['batch_size'],
        gradient_accumulation_steps=config['sft']['gradient_accumulation_steps'],
        max_seq_length=config['sft']['max_seq_length'],
        learning_rate=config['sft']['learning_rate']
    )
    
    trainer = SFTTrainer(sft_config)
    return trainer.train()


def step_raft(config, sft_checkpoint, prompts_file):
    """Run RAFT training."""
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
    from halo_forge.rlvr.verifiers import GCCVerifier
    
    print("\n" + "=" * 60)
    print("RAFT TRAINING")
    print("=" * 60)
    
    output_dir = Path(config['raft']['output_dir'])
    final_cycle = output_dir / f"cycle_{config['raft']['num_cycles']}_final"
    
    if final_cycle.exists():
        print(f"\nUsing existing RAFT model: {final_cycle}")
        return str(final_cycle)
    
    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt'])
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Create verifier
    verifier = GCCVerifier(max_workers=config['raft']['max_workers'])
    
    # RAFT config
    raft_config = RAFTConfig(
        base_model=config['model']['name'],
        sft_checkpoint=sft_checkpoint,
        output_dir=str(output_dir),
        num_cycles=config['raft']['num_cycles'],
        samples_per_prompt=config['raft']['samples_per_prompt'],
        reward_threshold=config['raft']['reward_threshold'],
        keep_top_percent=config['raft']['keep_top_percent']
    )
    
    # Train
    trainer = RAFTTrainer(verifier=verifier, config=raft_config)
    return trainer.run(prompts)


def step_benchmark(config, model_path, prompts_file):
    """Run benchmark."""
    from halo_forge.benchmark.pass_at_k import Benchmark
    from halo_forge.rlvr.verifiers import GCCVerifier
    
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)
    
    results_dir = Path(config['benchmark']['output_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt'])
    
    # Subset for benchmark
    prompts = prompts[:config['benchmark']['max_prompts']]
    
    # Create verifier
    verifier = GCCVerifier(max_workers=8)
    
    # Create benchmark
    benchmark = Benchmark(
        model_path=model_path,
        verifier=verifier,
        base_model=config['model']['name']
    )
    
    # Run
    result = benchmark.run(
        prompts=prompts,
        samples_per_prompt=config['benchmark']['samples_per_prompt'],
        k_values=config['benchmark']['k_values'],
        output_path=str(results_dir / "benchmark.json")
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Example")
    parser.add_argument('--step', choices=['data', 'sft', 'raft', 'bench', 'all'], default='all')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    
    if not config_path.exists():
        # Create default config
        config = {
            'model': {'name': 'Qwen/Qwen2.5-Coder-7B'},
            'data': {
                'datasets': ['codeforces_cpp'],
                'output_dir': 'data',
                'template': 'qwen'
            },
            'sft': {
                'output_dir': 'models/full_pipeline/sft',
                'num_epochs': 2,
                'batch_size': 2,
                'gradient_accumulation_steps': 16,
                'max_seq_length': 2048,
                'learning_rate': 2e-4
            },
            'raft': {
                'output_dir': 'models/full_pipeline/raft',
                'num_cycles': 3,
                'samples_per_prompt': 4,
                'reward_threshold': 0.5,
                'keep_top_percent': 0.5,
                'max_prompts': 100,
                'max_workers': 8
            },
            'benchmark': {
                'output_dir': 'results',
                'samples_per_prompt': 10,
                'k_values': [1, 5, 10],
                'max_prompts': 50
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Created default config: {config_path}")
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Run steps
    data_dir = Path(config['data']['output_dir'])
    train_file = data_dir / "train.jsonl"
    prompts_file = data_dir / "prompts.jsonl"
    
    if args.step in ['data', 'all']:
        train_file, prompts_file = step_data(config)
    
    sft_checkpoint = str(Path(config['sft']['output_dir']) / "final_model")
    
    if args.step in ['sft', 'all']:
        if not train_file.exists():
            print("Run data step first: --step data")
            return
        sft_checkpoint = step_sft(config, train_file)
    
    raft_checkpoint = str(Path(config['raft']['output_dir']) / f"cycle_{config['raft']['num_cycles']}_final")
    
    if args.step in ['raft', 'all']:
        if not Path(sft_checkpoint).exists():
            print("Run sft step first: --step sft")
            return
        raft_checkpoint = step_raft(config, sft_checkpoint, prompts_file)
    
    if args.step in ['bench', 'all']:
        if not Path(raft_checkpoint).exists():
            print("Run raft step first: --step raft")
            return
        result = step_benchmark(config, raft_checkpoint, prompts_file)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Final model: {raft_checkpoint}")
        print(f"Pass rate: {result.pass_rate:.1%}")
        for k, rate in result.pass_at_k.items():
            print(f"pass@{k}: {rate:.1%}")


if __name__ == "__main__":
    main()

