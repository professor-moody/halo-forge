#!/usr/bin/env python3
"""
Prepare HumanEval and MBPP datasets for RLVR training with pytest verification.

This script creates dataset files that include:
- prompts: For model generation
- tests: For pytest verification
- entry_point: Function name (HumanEval)
- task_id: For tracking

Output format designed for halo-forge RAFT training with PytestVerifier.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Any


def prepare_humaneval(output_dir: Path, split: str = "test") -> Dict[str, Any]:
    """
    Prepare HumanEval dataset for RLVR training.
    
    HumanEval has 164 problems, each with:
    - prompt: Function signature + docstring
    - test: Test code with assertions
    - entry_point: Function name
    - canonical_solution: Reference implementation
    
    Returns:
        Stats dict with counts
    """
    print("Loading HumanEval from HuggingFace...")
    ds = load_dataset("openai/openai_humaneval", split=split)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompts file (for RAFT generation)
    prompts_file = output_dir / "humaneval_prompts.jsonl"
    
    # Full dataset with tests (for verification)
    full_file = output_dir / "humaneval_full.jsonl"
    
    # Reference solutions (for analysis only)
    solutions_file = output_dir / "humaneval_solutions.jsonl"
    
    prompts = []
    full_records = []
    solutions = []
    
    for item in ds:
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        canonical = item["canonical_solution"]
        
        # Prompt record (what model sees during generation)
        prompts.append({
            "task_id": task_id,
            "prompt": prompt,
            "entry_point": entry_point
        })
        
        # Full record (for verification)
        # The test code in HumanEval is structured as:
        #   def check(candidate):
        #       assert candidate(...) == ...
        #   check(entry_point)
        full_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "tests": test_code,
            "entry_point": entry_point
        })
        
        # Solution (for reference/analysis)
        solutions.append({
            "task_id": task_id,
            "prompt": prompt,
            "solution": canonical,
            "entry_point": entry_point
        })
    
    # Write files
    with open(prompts_file, 'w') as f:
        for record in prompts:
            f.write(json.dumps(record) + '\n')
    
    with open(full_file, 'w') as f:
        for record in full_records:
            f.write(json.dumps(record) + '\n')
    
    with open(solutions_file, 'w') as f:
        for record in solutions:
            f.write(json.dumps(record) + '\n')
    
    print(f"Wrote {len(prompts)} problems to:")
    print(f"  - {prompts_file} (prompts only)")
    print(f"  - {full_file} (prompts + tests)")
    print(f"  - {solutions_file} (reference solutions)")
    
    return {
        "dataset": "humaneval",
        "count": len(prompts),
        "prompts_file": str(prompts_file),
        "full_file": str(full_file),
        "solutions_file": str(solutions_file)
    }


def prepare_mbpp(
    output_dir: Path, 
    split: str = "train",
    include_sanitized: bool = True
) -> Dict[str, Any]:
    """
    Prepare MBPP dataset for RLVR training.
    
    MBPP has ~974 problems, each with:
    - text: Problem description
    - code: Reference solution
    - test_list: List of assertion strings
    - challenge_test_list: Harder test cases
    
    Args:
        output_dir: Where to save files
        split: Dataset split (train/test/validation)
        include_sanitized: Also load mbpp-sanitized for cleaner subset
        
    Returns:
        Stats dict
    """
    print(f"Loading MBPP ({split}) from HuggingFace...")
    ds = load_dataset("google-research-datasets/mbpp", split=split)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_file = output_dir / f"mbpp_{split}_prompts.jsonl"
    full_file = output_dir / f"mbpp_{split}_full.jsonl"
    solutions_file = output_dir / f"mbpp_{split}_solutions.jsonl"
    
    prompts = []
    full_records = []
    solutions = []
    
    for item in ds:
        task_id = item["task_id"]
        text = item["text"]
        code = item["code"]
        test_list = item["test_list"]
        challenge_tests = item.get("challenge_test_list", [])
        
        # Extract function name from code (heuristic)
        entry_point = extract_function_name(code)
        
        # Build test code from assertions
        # MBPP gives us assertion strings like: "assert func(1) == 2"
        test_code = build_test_code(test_list, challenge_tests)
        
        prompts.append({
            "task_id": task_id,
            "prompt": text,
            "entry_point": entry_point
        })
        
        full_records.append({
            "task_id": task_id,
            "prompt": text,
            "tests": test_code,
            "entry_point": entry_point,
            "test_count": len(test_list) + len(challenge_tests)
        })
        
        solutions.append({
            "task_id": task_id,
            "prompt": text,
            "solution": code,
            "entry_point": entry_point
        })
    
    # Write files
    with open(prompts_file, 'w') as f:
        for record in prompts:
            f.write(json.dumps(record) + '\n')
    
    with open(full_file, 'w') as f:
        for record in full_records:
            f.write(json.dumps(record) + '\n')
    
    with open(solutions_file, 'w') as f:
        for record in solutions:
            f.write(json.dumps(record) + '\n')
    
    print(f"Wrote {len(prompts)} problems to:")
    print(f"  - {prompts_file}")
    print(f"  - {full_file}")
    print(f"  - {solutions_file}")
    
    stats = {
        "dataset": "mbpp",
        "split": split,
        "count": len(prompts),
        "prompts_file": str(prompts_file),
        "full_file": str(full_file),
        "solutions_file": str(solutions_file)
    }
    
    # Optionally also prepare sanitized subset
    if include_sanitized and split == "train":
        print("\nAlso preparing sanitized subset...")
        sanitized_stats = prepare_mbpp_sanitized(output_dir)
        stats["sanitized"] = sanitized_stats
    
    return stats


def prepare_mbpp_sanitized(output_dir: Path) -> Dict[str, Any]:
    """
    Prepare the sanitized MBPP subset (cleaner, higher quality).
    
    The sanitized version has ~427 problems with cleaner test cases.
    """
    try:
        ds = load_dataset("mbpp", "sanitized", split="test")
    except Exception:
        # Fallback to bigcode version
        print("Using bigcode/mbpp-sanitized...")
        ds = load_dataset("bigcode/mbpp", split="test")
    
    prompts_file = output_dir / "mbpp_sanitized_prompts.jsonl"
    full_file = output_dir / "mbpp_sanitized_full.jsonl"
    
    prompts = []
    full_records = []
    
    for item in ds:
        task_id = item.get("task_id", item.get("id"))
        text = item.get("text", item.get("prompt"))
        code = item.get("code", item.get("canonical_solution", ""))
        test_list = item.get("test_list", [])
        
        entry_point = extract_function_name(code)
        test_code = build_test_code(test_list, [])
        
        prompts.append({
            "task_id": task_id,
            "prompt": text,
            "entry_point": entry_point
        })
        
        full_records.append({
            "task_id": task_id,
            "prompt": text,
            "tests": test_code,
            "entry_point": entry_point
        })
    
    with open(prompts_file, 'w') as f:
        for record in prompts:
            f.write(json.dumps(record) + '\n')
    
    with open(full_file, 'w') as f:
        for record in full_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Wrote {len(prompts)} sanitized problems")
    
    return {
        "count": len(prompts),
        "prompts_file": str(prompts_file),
        "full_file": str(full_file)
    }


def extract_function_name(code: str) -> str:
    """Extract the main function name from Python code."""
    import re
    
    # Look for def statements
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if match:
        return match.group(1)
    
    return "solution"


def build_test_code(test_list: List[str], challenge_tests: List[str]) -> str:
    """
    Build pytest-compatible test code from assertion lists.
    
    MBPP provides assertions like:
        ["assert func(1) == 2", "assert func(2) == 4"]
    
    We convert to:
        def test_solution():
            assert func(1) == 2
            assert func(2) == 4
    """
    all_tests = test_list + challenge_tests
    
    if not all_tests:
        return "def test_solution():\n    pass"
    
    # Build test function
    lines = ["def test_solution():"]
    for assertion in all_tests:
        # Clean up the assertion
        assertion = assertion.strip()
        if not assertion.startswith("assert"):
            assertion = f"assert {assertion}"
        lines.append(f"    {assertion}")
    
    return "\n".join(lines)


def create_validation_subset(
    full_file: Path,
    output_file: Path,
    n: int = 20,
    seed: int = 42
) -> None:
    """
    Create a small validation subset for quick testing.
    
    Args:
        full_file: Full dataset file
        output_file: Where to write subset
        n: Number of problems to sample
        seed: Random seed
    """
    import random
    
    with open(full_file) as f:
        records = [json.loads(line) for line in f]
    
    random.seed(seed)
    subset = random.sample(records, min(n, len(records)))
    
    with open(output_file, 'w') as f:
        for record in subset:
            f.write(json.dumps(record) + '\n')
    
    print(f"Created validation subset: {output_file} ({len(subset)} problems)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HumanEval and MBPP for RLVR training"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/rlvr"),
        help="Output directory"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["humaneval", "mbpp", "both"],
        default="both",
        help="Which dataset(s) to prepare"
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=20,
        help="Size of validation subset for quick testing"
    )
    
    args = parser.parse_args()
    
    stats = {}
    
    if args.dataset in ["humaneval", "both"]:
        print("\n" + "="*60)
        print("PREPARING HUMANEVAL")
        print("="*60)
        stats["humaneval"] = prepare_humaneval(args.output_dir)
        
        # Create validation subset
        create_validation_subset(
            args.output_dir / "humaneval_full.jsonl",
            args.output_dir / "humaneval_validation.jsonl",
            n=args.validation_size
        )
    
    if args.dataset in ["mbpp", "both"]:
        print("\n" + "="*60)
        print("PREPARING MBPP")
        print("="*60)
        stats["mbpp"] = prepare_mbpp(args.output_dir, split="train")
        
        # Create validation subset
        create_validation_subset(
            args.output_dir / "mbpp_train_full.jsonl",
            args.output_dir / "mbpp_validation.jsonl",
            n=args.validation_size
        )
    
    # Write summary
    summary_file = args.output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(json.dumps(stats, indent=2))
    print(f"\nSummary written to: {summary_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
For 3B validation run:
    halo-forge raft train \\
        --prompts data/rlvr/humaneval_validation.jsonl \\
        --verifier pytest \\
        --cycles 3

For 7B production run:
    halo-forge raft train \\
        --prompts data/rlvr/mbpp_train_full.jsonl \\
        --verifier pytest \\
        --cycles 5

For benchmarking:
    halo-forge benchmark run \\
        --prompts data/rlvr/humaneval_full.jsonl \\
        --verifier pytest
""")


if __name__ == "__main__":
    main()