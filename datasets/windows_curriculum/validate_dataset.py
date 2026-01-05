#!/usr/bin/env python3
"""
Windows Curriculum Dataset Validator

This script validates that all problems in the dataset:
1. Have valid JSONL format
2. Have required fields
3. Solutions compile with MSVC (when run on Windows)
4. Pass their test cases

Usage:
    python validate_dataset.py                    # Check format only
    python validate_dataset.py --compile          # Check format + compile
    python validate_dataset.py --compile --run    # Full validation
"""

import json
import argparse
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class DatasetValidator:
    def __init__(self, dataset_path: Path, compile_test: bool = False, run_test: bool = False):
        self.dataset_path = dataset_path
        self.compile_test = compile_test
        self.run_test = run_test
        self.problems: List[Dict] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.compile_results: Dict[str, bool] = {}
        self.run_results: Dict[str, Tuple[bool, str]] = {}
    
    def load_dataset(self) -> bool:
        """Load and parse the JSONL dataset."""
        print(f"Loading dataset: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            self.errors.append(f"Dataset file not found: {self.dataset_path}")
            return False
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    problem = json.loads(line.strip())
                    problem['_line'] = i
                    self.problems.append(problem)
                except json.JSONDecodeError as e:
                    self.errors.append(f"Line {i}: Invalid JSON - {e}")
        
        print(f"Loaded {len(self.problems)} problems")
        return len(self.errors) == 0
    
    def validate_fields(self) -> bool:
        """Check that all required fields are present."""
        print("Validating required fields...")
        
        required_fields = ['id', 'prompt', 'solution', 'tier', 'category', 'difficulty']
        optional_fields = ['test_cases', 'subcategory', 'api', 'tags', 'verification_strategy']
        
        for p in self.problems:
            line = p.get('_line', '?')
            pid = p.get('id', f'line_{line}')
            
            for field in required_fields:
                if field not in p:
                    self.errors.append(f"{pid}: Missing required field '{field}'")
                elif not p[field]:
                    self.warnings.append(f"{pid}: Empty field '{field}'")
            
            # Validate tier
            tier = p.get('tier')
            if tier not in [1, 2, 3, 4]:
                self.errors.append(f"{pid}: Invalid tier '{tier}' (must be 1-4)")
            
            # Validate difficulty
            diff = p.get('difficulty')
            if diff not in ['beginner', 'intermediate', 'advanced']:
                self.errors.append(f"{pid}: Invalid difficulty '{diff}'")
            
            # Check solution is C++ code
            solution = p.get('solution', '')
            if '#include' not in solution:
                self.warnings.append(f"{pid}: Solution doesn't appear to be C++ code")
            if 'int main' not in solution and 'void main' not in solution:
                self.warnings.append(f"{pid}: Solution missing main() function")
        
        return len(self.errors) == 0
    
    def compile_solutions(self) -> bool:
        """Compile all solutions with MSVC (Windows only)."""
        if sys.platform != 'win32':
            print("Skipping compile test (not on Windows)")
            return True
        
        print("Compiling solutions with MSVC...")
        
        # Check for cl.exe
        try:
            subprocess.run(['cl', '/?'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.warnings.append("MSVC (cl.exe) not found in PATH. Run from Developer Command Prompt.")
            return True
        
        compiled = 0
        failed = 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for p in self.problems:
                pid = p.get('id', 'unknown')
                solution = p.get('solution', '')
                
                # Write source file
                src_path = Path(tmpdir) / f"{pid}.cpp"
                exe_path = Path(tmpdir) / f"{pid}.exe"
                
                with open(src_path, 'w', encoding='utf-8') as f:
                    f.write(solution)
                
                # Compile
                result = subprocess.run(
                    ['cl', '/nologo', '/EHsc', '/W3', '/Fe:', str(exe_path), str(src_path)],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir
                )
                
                if result.returncode == 0:
                    compiled += 1
                    self.compile_results[pid] = True
                else:
                    failed += 1
                    self.compile_results[pid] = False
                    self.errors.append(f"{pid}: Compile failed - {result.stderr[:200]}")
        
        print(f"Compiled: {compiled}/{len(self.problems)}, Failed: {failed}")
        return failed == 0
    
    def run_solutions(self) -> bool:
        """Run compiled solutions and check test cases (Windows only)."""
        if sys.platform != 'win32':
            print("Skipping run test (not on Windows)")
            return True
        
        if not self.compile_test:
            print("Skipping run test (compile not enabled)")
            return True
        
        print("Running solutions and checking test cases...")
        
        passed = 0
        failed = 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for p in self.problems:
                pid = p.get('id', 'unknown')
                solution = p.get('solution', '')
                test_cases = p.get('test_cases', [])
                
                if not self.compile_results.get(pid, False):
                    continue  # Skip if compile failed
                
                # Write and compile
                src_path = Path(tmpdir) / f"{pid}.cpp"
                exe_path = Path(tmpdir) / f"{pid}.exe"
                
                with open(src_path, 'w', encoding='utf-8') as f:
                    f.write(solution)
                
                compile_result = subprocess.run(
                    ['cl', '/nologo', '/EHsc', '/Fe:', str(exe_path), str(src_path)],
                    capture_output=True,
                    cwd=tmpdir
                )
                
                if compile_result.returncode != 0:
                    continue
                
                # Run
                try:
                    run_result = subprocess.run(
                        [str(exe_path)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=tmpdir
                    )
                    output = run_result.stdout + run_result.stderr
                    
                    # Check test cases
                    all_passed = True
                    for tc in test_cases:
                        tc_type = tc.get('type', '')
                        tc_value = tc.get('value', '')
                        
                        if tc_type == 'output_contains':
                            if tc_value not in output:
                                all_passed = False
                                self.warnings.append(f"{pid}: Output missing '{tc_value}'")
                        elif tc_type == 'exit_code':
                            expected = tc.get('expected', 0)
                            if run_result.returncode != expected:
                                all_passed = False
                                self.warnings.append(f"{pid}: Exit code {run_result.returncode} != {expected}")
                    
                    if all_passed:
                        passed += 1
                        self.run_results[pid] = (True, "")
                    else:
                        failed += 1
                        self.run_results[pid] = (False, "Test case failed")
                    
                except subprocess.TimeoutExpired:
                    failed += 1
                    self.run_results[pid] = (False, "Timeout")
                    self.warnings.append(f"{pid}: Execution timeout")
                except Exception as e:
                    failed += 1
                    self.run_results[pid] = (False, str(e))
        
        print(f"Passed: {passed}, Failed: {failed}")
        return failed == 0
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        print(f"\nDataset: {self.dataset_path}")
        print(f"Total problems: {len(self.problems)}")
        
        # Tier distribution
        tiers = {}
        for p in self.problems:
            tier = p.get('tier', 0)
            tiers[tier] = tiers.get(tier, 0) + 1
        
        print("\nTier Distribution:")
        for t in sorted(tiers.keys()):
            print(f"  Tier {t}: {tiers[t]}")
        
        # Category distribution
        cats = {}
        for p in self.problems:
            cat = p.get('category', 'unknown')
            cats[cat] = cats.get(cat, 0) + 1
        
        print(f"\nCategories: {len(cats)}")
        
        if self.compile_test and self.compile_results:
            passed = sum(1 for v in self.compile_results.values() if v)
            print(f"\nCompile Results: {passed}/{len(self.compile_results)} passed")
        
        if self.run_test and self.run_results:
            passed = sum(1 for v, _ in self.run_results.values() if v)
            print(f"Run Results: {passed}/{len(self.run_results)} passed")
        
        if self.errors:
            print(f"\n[X] ERRORS ({len(self.errors)}):")
            for e in self.errors[:20]:
                print(f"  - {e}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more")
        
        if self.warnings:
            print(f"\n[!] WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                print(f"  - {w}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        if not self.errors:
            print("\n[OK] Dataset validation passed!")
        else:
            print(f"\n[FAIL] Dataset validation failed with {len(self.errors)} errors")
        
        print("="*60)
    
    def validate(self) -> bool:
        """Run full validation."""
        if not self.load_dataset():
            self.print_report()
            return False
        
        self.validate_fields()
        
        if self.compile_test:
            self.compile_solutions()
        
        if self.run_test:
            self.run_solutions()
        
        self.print_report()
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate Windows curriculum dataset")
    parser.add_argument('dataset', nargs='?', 
                        default='windows_systems_full_rlvr.jsonl',
                        help='Dataset file to validate')
    parser.add_argument('--compile', action='store_true',
                        help='Test compilation with MSVC')
    parser.add_argument('--run', action='store_true',
                        help='Run solutions and check test cases')
    
    args = parser.parse_args()
    
    # Find dataset file
    script_dir = Path(__file__).parent
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = script_dir / args.dataset
    
    validator = DatasetValidator(
        dataset_path,
        compile_test=args.compile,
        run_test=args.run
    )
    
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

