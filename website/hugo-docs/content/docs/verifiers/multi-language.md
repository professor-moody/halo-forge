---
title: "Multi-Language Verifier"
weight: 26
---

# Multi-Language Verifier

The `MultiLanguageVerifier` automatically detects the programming language from code patterns and routes to the appropriate language-specific verifier.

## Overview

Instead of manually specifying a verifier for each language, the multi-language verifier:

1. **Analyzes** the code for language-specific patterns
2. **Detects** the programming language
3. **Routes** to the appropriate verifier
4. **Returns** the verification result

## Supported Languages

| Language | Detection Patterns | Verifier Used |
|----------|-------------------|---------------|
| C++ | `#include <iostream>`, `std::`, `cout <<` | GCCVerifier |
| C | `#include <stdio.h>`, `printf(` | GCCVerifier |
| Python | `def `, `import `, `print(` | PytestVerifier |
| Rust | `fn main()`, `use std::`, `println!` | RustVerifier |
| Go | `package main`, `func main()`, `fmt.` | GoVerifier |
| C# | `using System`, `Console.WriteLine` | DotNetVerifier |
| PowerShell | `$var =`, `Write-Host`, `Get-` | PowerShellVerifier |

## Basic Usage

```python
from halo_forge.rlvr.verifiers import MultiLanguageVerifier

verifier = MultiLanguageVerifier()

# C++ code - auto-detected
cpp_code = '''
#include <iostream>
int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
'''
result = verifier.verify(cpp_code)
print(f"Detected: {result.metadata['detected_language']}")  # cpp

# Python code - auto-detected
python_code = '''
def hello():
    print("Hello")

if __name__ == "__main__":
    hello()
'''
result = verifier.verify(python_code)
print(f"Detected: {result.metadata['detected_language']}")  # python

# Rust code - auto-detected
rust_code = '''
fn main() {
    println!("Hello");
}
'''
result = verifier.verify(rust_code)
print(f"Detected: {result.metadata['detected_language']}")  # rust
```

## CLI Usage

Use `--verifier auto` to enable multi-language detection:

```bash
# Benchmark with auto-detection
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/multi_lang_prompts.jsonl \
  --verifier auto \
  --samples 10 \
  --output results/multi_lang.json

# RAFT training with auto-detection
halo-forge raft train \
  --prompts data/mixed_prompts.jsonl \
  --verifier auto \
  --model Qwen/Qwen2.5-Coder-3B \
  --cycles 6 \
  --output models/multi_lang_raft
```

## Explicit Language Override

You can override auto-detection:

```python
verifier = MultiLanguageVerifier()

# Force Rust verification even if code looks like C++
result = verifier.verify(code, language='rust')
```

## Configuration Options

```python
verifier = MultiLanguageVerifier(
    default_language='python',      # Fallback if detection fails
    max_workers=8,                  # Parallel verification workers
    run_after_compile=False,        # Run binaries after compile
    binary_cache_dir='binaries/',   # Cache compiled binaries
)
```

## Language Detection Priority

Languages are checked in priority order:

1. **C++** (priority 10) - Most specific patterns
2. **Rust** (priority 9) - Distinctive syntax
3. **Go** (priority 9) - Distinctive syntax
4. **Python** (priority 8) - Common patterns
5. **C#** (priority 7) - .NET patterns
6. **PowerShell** (priority 6) - Cmdlet patterns
7. **C** (priority 5) - Basic C patterns

Higher priority languages are checked first. The first matching pattern wins.

## Custom Language Configuration

Add or modify language detection:

```python
from halo_forge.rlvr.verifiers import MultiLanguageVerifier, LanguageConfig

# Custom language config
my_configs = {
    'typescript': LanguageConfig(
        name='typescript',
        patterns=[r'^import .* from', r': string', r': number'],
        verifier_class='NodeVerifier',  # Your custom verifier
        priority=8
    )
}

verifier = MultiLanguageVerifier(language_configs=my_configs)
```

## Batch Verification

Verify multiple code samples with different languages:

```python
codes = [cpp_code, python_code, rust_code, go_code]
results = verifier.verify_batch(codes)

for code, result in zip(codes, results):
    print(f"Language: {result.metadata['detected_language']}")
    print(f"Success: {result.success}")
```

## Using with Mixed Datasets

For datasets containing multiple languages:

```python
import json
from halo_forge.rlvr.verifiers import MultiLanguageVerifier

verifier = MultiLanguageVerifier()

# Load mixed-language prompts
with open('data/mixed_lang_prompts.jsonl') as f:
    prompts = [json.loads(line) for line in f]

# The verifier handles each language automatically
for prompt in prompts:
    result = verifier.verify(prompt['completion'])
    print(f"{result.metadata['detected_language']}: {result.success}")
```

## Supported Languages

Check what languages are available:

```python
verifier = MultiLanguageVerifier()
print(verifier.supported_languages)
# ['cpp', 'c', 'python', 'rust', 'go', 'csharp', 'powershell']
```

## Performance Considerations

- **Lazy loading**: Verifiers are created on-demand, not at initialization
- **Caching**: Once a language verifier is created, it's reused
- **Parallel**: Batch verification uses thread pool

## Comparison with Single-Language Verifiers

| Aspect | Single-Language | MultiLanguage |
|--------|-----------------|---------------|
| Setup | Specify verifier | Auto-detect |
| Performance | Slightly faster | Small overhead |
| Flexibility | One language | All languages |
| Use case | Homogeneous data | Mixed datasets |

## Best Practices

1. **Use for mixed datasets** - When prompts may generate different languages
2. **Set default_language** - Provide fallback for ambiguous code
3. **Check detected_language** - Verify detection is correct in metadata
4. **Consider explicit language** - When you know the expected language

## Alias

`AutoVerifier` is an alias for `MultiLanguageVerifier`:

```python
from halo_forge.rlvr.verifiers import AutoVerifier

# Same as MultiLanguageVerifier
verifier = AutoVerifier()
```
