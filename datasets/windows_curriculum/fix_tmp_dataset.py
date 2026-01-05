#!/usr/bin/env python3
"""Fix missing library pragmas in the tmp/ dataset."""
import json
from pathlib import Path

TMP_RLVR = Path(__file__).parent.parent / "tmp" / "rlvr_dataset_200.jsonl"
TMP_SFT = Path(__file__).parent.parent / "tmp" / "sft_dataset_200.jsonl"

# APIs that need advapi32.lib
ADVAPI32_APIS = [
    "GetUserNameA", "GetUserNameW",
    "RegOpenKeyEx", "RegCloseKey", "RegQueryValueEx", "RegSetValueEx",
    "RegEnumKeyEx", "RegCreateKeyEx", "RegDeleteKey",
    "OpenProcessToken", "GetTokenInformation", "LookupPrivilegeValue",
    "AdjustTokenPrivileges", "SetTokenInformation",
    "ConvertSidToStringSid", "LookupAccountSid", "GetSidSubAuthority",
]

def needs_advapi32(solution: str) -> bool:
    """Check if solution uses APIs that need advapi32.lib."""
    for api in ADVAPI32_APIS:
        if api in solution:
            return True
    return False

def add_pragma(solution: str, lib: str) -> str:
    """Add a #pragma comment(lib, ...) if not present."""
    pragma = f'#pragma comment(lib, "{lib}")'
    if pragma in solution:
        return solution
    
    # Find the last #include or first line after all includes
    lines = solution.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            insert_idx = i + 1
    
    lines.insert(insert_idx, pragma)
    return '\n'.join(lines)

def main():
    if not TMP_RLVR.exists():
        print(f"tmp dataset not found: {TMP_RLVR}")
        return
    
    print(f"Fixing {TMP_RLVR}...")
    
    fixed = 0
    problems = []
    with open(TMP_RLVR) as f:
        for line in f:
            p = json.loads(line)
            solution = p.get("solution", "")
            
            if needs_advapi32(solution):
                p["solution"] = add_pragma(solution, "advapi32.lib")
                fixed += 1
            
            problems.append(p)
    
    # Write back
    with open(TMP_RLVR, 'w') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')
    
    print(f"Fixed {fixed} problems")
    print("Now re-run merge_datasets.py")

if __name__ == "__main__":
    main()

