#!/usr/bin/env python3
"""Get detailed compile errors for failing problems."""
import json
import subprocess
import tempfile
import os
import sys

def main():
    # Use command line arg or default
    dataset = sys.argv[1] if len(sys.argv) > 1 else "windows_curriculum_rlvr.jsonl"
    
    with open(dataset) as f:
        problems = [json.loads(line) for line in f]
    
    print(f"Testing {len(problems)} problems from {dataset}...")
    
    total = 0
    failed = 0
    
    for p in problems:
        total += 1
        pid = p.get("id", "unknown")
        solution = p.get("solution", "")
        
        # Write source
        with open("test_compile.cpp", "w", encoding="utf-8") as f:
            f.write(solution)
        
        # Compile
        result = subprocess.run(
            ["cl", "/nologo", "/EHsc", "/W3", "/Fe:test_compile.exe", "test_compile.cpp"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            failed += 1
            print(f"\n=== {pid} ===")
            # Get error lines from both stdout and stderr (MSVC outputs to stdout)
            output = result.stdout + result.stderr
            for line in output.split("\n"):
                if "error" in line.lower() or "fatal" in line.lower():
                    print(line.strip()[:200])
        
        # Cleanup
        for f in ["test_compile.cpp", "test_compile.exe", "test_compile.obj"]:
            try:
                os.remove(f)
            except:
                pass
    
    passed = total - failed
    print(f"\n=== SUMMARY ===")
    print(f"Total: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

