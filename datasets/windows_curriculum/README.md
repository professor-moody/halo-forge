# Windows Systems Programming Curriculum Dataset

A comprehensive dataset for training code models on Windows systems programming, structured with curriculum learning principles.

## Overview

| Metric | Count |
|--------|-------|
| Total Problems | 361 |
| Tier 1 (Foundations) | 84 |
| Tier 2 (Core APIs) | 128 |
| Tier 3 (Intermediate) | 72 |
| Tier 4 (Advanced) | 77 |
| Categories | 24 |
| MSVC Compile Rate | **100%** |

## Dataset Files

| File | Description |
|------|-------------|
| `windows_systems_full_rlvr.jsonl` | Full merged dataset for RLVR training (361 problems) |
| `windows_systems_full_sft.jsonl` | SFT-formatted version with chat templates |
| `windows_curriculum_rlvr.jsonl` | New tiered curriculum problems only (156) |
| `windows_curriculum_sft.jsonl` | SFT version of curriculum problems |
| `curriculum_order_full.json` | Curriculum progression metadata |

## Tier Structure

### Tier 1: Foundations (Beginner)
- **Verification**: stdout comparison
- **Topics**: Process ID, environment variables, file I/O, registry read, time/date, string handling, error handling
- **Focus**: Basic Windows API usage

### Tier 2: Core APIs (Intermediate)
- **Verification**: output + side effects
- **Topics**: Process/module/thread enumeration, memory mapping, named pipes, events/mutexes, DLL loading, heap operations
- **Focus**: System programming fundamentals

### Tier 3: Intermediate (Medium-Hard)
- **Verification**: structured output
- **Topics**: PE parsing, token queries, SID/ACL, service control, I/O completion ports, thread pool, fibers, VEH, TLS
- **Focus**: Windows internals

### Tier 4: Advanced (Hard)
- **Verification**: behavioral checks
- **Topics**: ETW, native API (NtXxx), PEB walking, syscall stubs, hook detection, memory manipulation
- **Focus**: Low-level systems programming

## Categories

memory, pe, native, process, threading, security, registry, file, dll, evasion, internals, sync, ipc, syscall, sysinfo, console, environment, error, exception, services, string, time, misc

## Schema

Each problem includes:

```json
{
  "id": "win_1_0001",
  "tier": 1,
  "category": "process",
  "subcategory": "info",
  "api": "GetCurrentProcessId",
  "difficulty": "beginner",
  "prompt": "Write a C++ program that...",
  "solution": "#include <windows.h>...",
  "test_cases": [
    {"type": "output_contains", "value": "Process ID:"}
  ],
  "tags": ["GetCurrentProcessId", "process"],
  "verification_strategy": "stdout_contains"
}
```

## Validation

Run validation on Windows with MSVC:

```powershell
cd datasets\windows_curriculum
. $env:USERPROFILE\init-msvc.ps1  # If you have MSVC init script
python validate_dataset.py windows_systems_full_rlvr.jsonl --compile
```

Or format-only validation on any platform:

```bash
python validate_dataset.py windows_systems_full_rlvr.jsonl
```

## Usage in Training

### For SFT
```bash
halo-forge sft train --data datasets/windows_curriculum/windows_systems_full_sft.jsonl
```

### For RLVR/RAFT
```bash
halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier msvc
```

## Curriculum Training Order

For best results with curriculum learning, train in tier order:

1. Warm-up with Tier 1 (foundations)
2. Progress to Tier 2 (core APIs)
3. Challenge with Tier 3 (intermediate)
4. Specialize with Tier 4 (advanced)

Use the `curriculum_order_full.json` file to programmatically access tier-based batching.

## Scripts

- `generate_curriculum_dataset.py` - Generates new tiered problems
- `merge_datasets.py` - Merges with existing problems and adds tier metadata
- `validate_dataset.py` - Validates format and MSVC compilation
- `fix_tmp_dataset.py` - Auto-fix missing lib pragmas in older problems
- `get_errors.py` - Compile test utility with error reporting

## License

This dataset is provided for research and educational purposes.

