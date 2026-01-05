#!/usr/bin/env python3
"""
Extract CodeForces C++ sample data for quick testing.
Downloads from HuggingFace and saves 500 examples.
"""

import json
import re
import os
from pathlib import Path

def main():
    from datasets import load_dataset
    
    print("Loading CodeForces dataset from HuggingFace...")
    ds = load_dataset('open-r1/codeforces-cots', split='train')
    print(f"Total examples: {len(ds)}")
    
    # Process and extract C++ solutions
    samples = []
    code_pattern = re.compile(r'```(?:cpp|c\+\+)?\n(.*?)```', re.DOTALL)
    
    for ex in ds:
        generation = ex.get('generation', '')
        
        # Check if it contains C++ code
        if '```cpp' in generation or '#include' in generation:
            # Extract problem description
            problem = ex.get('description', '')
            if ex.get('input_format'):
                problem += '\n\nInput Format:\n' + ex.get('input_format', '')
            if ex.get('output_format'):
                problem += '\n\nOutput Format:\n' + ex.get('output_format', '')
            
            # Extract code from generation (after </think>)
            code = ''
            if '</think>' in generation:
                after_think = generation.split('</think>')[-1]
                code_match = code_pattern.search(after_think)
                if code_match:
                    code = code_match.group(1).strip()
            
            if not code:
                # Try to find any C++ code block
                code_match = code_pattern.search(generation)
                if code_match:
                    code = code_match.group(1).strip()
            
            if problem and code and len(code) > 50 and '#include' in code:
                samples.append({
                    'prompt': problem.strip(),
                    'response': code,
                    'title': ex.get('title', ''),
                    'difficulty': ex.get('index', ''),
                })
        
        if len(samples) >= 500:
            break
    
    print(f"Extracted {len(samples)} C++ examples")
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'codeforces_cpp_500.jsonl'
    
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    
    print(f"Saved to {output_path}")
    
    # Show sample
    if samples:
        print()
        print("Sample entry:")
        print(f"  Title: {samples[0]['title']}")
        print(f"  Prompt length: {len(samples[0]['prompt'])} chars")
        print(f"  Code length: {len(samples[0]['response'])} chars")


if __name__ == '__main__':
    main()

