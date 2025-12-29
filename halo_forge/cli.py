#!/usr/bin/env python3
"""
halo-forge CLI

Unified command-line interface for the halo-forge framework.

Usage:
    halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
    halo-forge data generate --topic rust_async --backend deepseek --output data/rust.jsonl
    halo-forge sft train --config configs/sft.yaml
    halo-forge raft train --config configs/raft.yaml
    halo-forge benchmark run --model models/raft/cycle_3 --prompts data/test.jsonl
    halo-forge info  # Show hardware info
"""

import argparse
import sys
import json
from pathlib import Path


def cmd_data_prepare(args):
    """Prepare dataset from public sources."""
    from halo_forge.data.public_datasets import (
        DatasetPreparer,
        get_dataset_spec,
        list_datasets
    )
    
    if args.list:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return
    
    if not args.dataset:
        print("Error: --dataset required")
        print("Use --list to see available datasets")
        sys.exit(1)
    
    spec = get_dataset_spec(args.dataset)
    preparer = DatasetPreparer(spec, system_prompt=args.system_prompt)
    
    output = args.output or f"data/{args.dataset}.jsonl"
    preparer.prepare(output, template=args.template)


def cmd_data_generate(args):
    """Generate data with LLM."""
    from halo_forge.data.llm_generate import (
        TrainingDataGenerator,
        get_backend,
        get_topic_spec,
        list_topics
    )
    
    if args.list:
        print("Available topics:")
        for name in list_topics():
            print(f"  - {name}")
        return
    
    if not args.topic:
        print("Error: --topic required")
        print("Use --list to see available topics")
        sys.exit(1)
    
    spec = get_topic_spec(args.topic)
    backend = get_backend(args.backend, model=args.model)
    generator = TrainingDataGenerator(backend, spec)
    
    output = args.output or f"data/{args.topic}_generated.jsonl"
    generator.generate_all(output, template=args.template)


def cmd_sft_train(args):
    """Run SFT training."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    if args.config:
        config = SFTConfig.from_yaml(args.config)
    else:
        config = SFTConfig(
            train_file=args.data,
            output_dir=args.output,
            num_epochs=args.epochs
        )
    
    trainer = SFTTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume)


def cmd_raft_train(args):
    """Run RAFT training."""
    import yaml
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
    from halo_forge.rlvr.verifiers import GCCVerifier, MinGWVerifier, RemoteMSVCVerifier
    
    # Load config
    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
    else:
        cfg_dict = {}
    
    # Setup verifier
    verifier_type = args.verifier or cfg_dict.get('verifier', {}).get('type', 'gcc')
    
    if verifier_type == 'gcc':
        verifier = GCCVerifier()
    elif verifier_type == 'mingw':
        verifier = MinGWVerifier()
    elif verifier_type == 'msvc':
        verifier = RemoteMSVCVerifier(
            host=cfg_dict.get('verifier', {}).get('host', 'localhost'),
            user=cfg_dict.get('verifier', {}).get('user', 'user'),
            ssh_key=cfg_dict.get('verifier', {}).get('ssh_key', '~/.ssh/id_rsa')
        )
    else:
        print(f"Unknown verifier: {verifier_type}")
        sys.exit(1)
    
    # Create config
    config = RAFTConfig(
        sft_checkpoint=args.checkpoint or cfg_dict.get('sft_checkpoint', 'models/sft/final_model'),
        output_dir=args.output or cfg_dict.get('output_dir', 'models/raft'),
        num_cycles=args.cycles or cfg_dict.get('num_cycles', 3)
    )
    
    # Load prompts
    prompts = []
    prompts_file = args.prompts or cfg_dict.get('prompts')
    if prompts_file:
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get('prompt', data.get('text', '')))
    
    if not prompts:
        print("Error: No prompts provided")
        print("Use --prompts or set in config")
        sys.exit(1)
    
    # Run
    trainer = RAFTTrainer(verifier=verifier, config=config)
    trainer.run(prompts, num_cycles=config.num_cycles)


def cmd_benchmark(args):
    """Run benchmark."""
    from halo_forge.benchmark.pass_at_k import Benchmark
    from halo_forge.rlvr.verifiers import GCCVerifier, MinGWVerifier, RemoteMSVCVerifier
    
    # Setup verifier
    if args.verifier == 'gcc':
        verifier = GCCVerifier()
    elif args.verifier == 'mingw':
        verifier = MinGWVerifier()
    elif args.verifier == 'msvc':
        verifier = RemoteMSVCVerifier(
            host=args.host,
            user=args.user,
            ssh_key=args.ssh_key
        )
    else:
        print(f"Unknown verifier: {args.verifier}")
        sys.exit(1)
    
    # Create benchmark
    benchmark = Benchmark(
        model_path=args.model,
        verifier=verifier,
        base_model=args.base_model,
        system_prompt=args.system_prompt
    )
    
    # Parse k values
    k_values = [int(k) for k in args.k.split(',')]
    
    # Run
    result = benchmark.run(
        prompts=args.prompts,
        samples_per_prompt=args.samples,
        k_values=k_values,
        max_prompts=args.max_prompts,
        output_path=args.output
    )


def cmd_info(args):
    """Show hardware info."""
    from halo_forge.utils.hardware import print_hardware_info
    print_hardware_info()


def main():
    parser = argparse.ArgumentParser(
        prog='halo-forge',
        description='Complete RLVR training framework for AMD Strix Halo'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # data command
    data_parser = subparsers.add_parser('data', help='Data preparation')
    data_subparsers = data_parser.add_subparsers(dest='data_command', required=True)
    
    # data prepare
    prepare_parser = data_subparsers.add_parser('prepare', help='Prepare public dataset')
    prepare_parser.add_argument('--dataset', '-d', help='Dataset name')
    prepare_parser.add_argument('--output', '-o', help='Output file path')
    prepare_parser.add_argument('--template', default='qwen', help='Chat template')
    prepare_parser.add_argument('--system-prompt', help='Override system prompt')
    prepare_parser.add_argument('--list', action='store_true', help='List available datasets')
    
    # data generate
    generate_parser = data_subparsers.add_parser('generate', help='Generate with LLM')
    generate_parser.add_argument('--topic', '-t', help='Topic name')
    generate_parser.add_argument('--backend', '-b', default='deepseek', help='LLM backend')
    generate_parser.add_argument('--model', help='Model name for backend')
    generate_parser.add_argument('--output', '-o', help='Output file path')
    generate_parser.add_argument('--template', default='qwen', help='Chat template')
    generate_parser.add_argument('--list', action='store_true', help='List available topics')
    
    # sft command
    sft_parser = subparsers.add_parser('sft', help='SFT training')
    sft_subparsers = sft_parser.add_subparsers(dest='sft_command', required=True)
    
    # sft train
    sft_train_parser = sft_subparsers.add_parser('train', help='Run SFT training')
    sft_train_parser.add_argument('--config', '-c', help='Config file path')
    sft_train_parser.add_argument('--data', help='Training data file')
    sft_train_parser.add_argument('--output', '-o', default='models/sft', help='Output directory')
    sft_train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    sft_train_parser.add_argument('--resume', help='Resume from checkpoint')
    
    # raft command
    raft_parser = subparsers.add_parser('raft', help='RAFT training')
    raft_subparsers = raft_parser.add_subparsers(dest='raft_command', required=True)
    
    # raft train
    raft_train_parser = raft_subparsers.add_parser('train', help='Run RAFT training')
    raft_train_parser.add_argument('--config', '-c', help='Config file path')
    raft_train_parser.add_argument('--checkpoint', help='SFT checkpoint path')
    raft_train_parser.add_argument('--prompts', '-p', help='Prompts file')
    raft_train_parser.add_argument('--output', '-o', default='models/raft', help='Output directory')
    raft_train_parser.add_argument('--cycles', type=int, help='Number of RAFT cycles')
    raft_train_parser.add_argument('--verifier', default='gcc', help='Verifier type')
    
    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmarking')
    bench_subparsers = bench_parser.add_subparsers(dest='bench_command', required=True)
    
    # benchmark run
    bench_run_parser = bench_subparsers.add_parser('run', help='Run benchmark')
    bench_run_parser.add_argument('--model', '-m', required=True, help='Model path')
    bench_run_parser.add_argument('--prompts', '-p', required=True, help='Prompts file')
    bench_run_parser.add_argument('--output', '-o', help='Output file path')
    bench_run_parser.add_argument('--samples', type=int, default=10, help='Samples per prompt')
    bench_run_parser.add_argument('--k', default='1,5,10', help='k values (comma-separated)')
    bench_run_parser.add_argument('--max-prompts', type=int, help='Max prompts to evaluate')
    bench_run_parser.add_argument('--verifier', default='gcc', help='Verifier type')
    bench_run_parser.add_argument('--base-model', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    bench_run_parser.add_argument('--system-prompt', default='You are an expert programmer.', help='System prompt')
    bench_run_parser.add_argument('--host', help='MSVC host')
    bench_run_parser.add_argument('--user', help='MSVC user')
    bench_run_parser.add_argument('--ssh-key', help='MSVC SSH key')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show hardware info')
    
    # Parse
    args = parser.parse_args()
    
    # Route to handler
    if args.command == 'data':
        if args.data_command == 'prepare':
            cmd_data_prepare(args)
        elif args.data_command == 'generate':
            cmd_data_generate(args)
    elif args.command == 'sft':
        if args.sft_command == 'train':
            cmd_sft_train(args)
    elif args.command == 'raft':
        if args.raft_command == 'train':
            cmd_raft_train(args)
    elif args.command == 'benchmark':
        if args.bench_command == 'run':
            cmd_benchmark(args)
    elif args.command == 'info':
        cmd_info(args)


if __name__ == '__main__':
    main()

