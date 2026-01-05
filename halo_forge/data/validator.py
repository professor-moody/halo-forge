"""
Dataset Validation

Validate training data format and provide statistics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    
    valid: bool
    total_examples: int
    valid_examples: int
    invalid_examples: int
    
    # Format info
    detected_format: str  # 'sft', 'prompt_response', 'messages', 'unknown'
    has_text_field: int
    has_prompt_field: int
    has_response_field: int
    has_messages_field: int
    
    # Stats
    avg_prompt_length: float
    avg_response_length: float
    max_prompt_length: int
    max_response_length: int
    
    # Issues
    errors: List[str]
    warnings: List[str]
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'✓ VALID' if self.valid else '✗ INVALID'}",
            f"Format: {self.detected_format}",
            "",
            "Examples:",
            f"  Total:   {self.total_examples}",
            f"  Valid:   {self.valid_examples}",
            f"  Invalid: {self.invalid_examples}",
            "",
            "Fields Found:",
            f"  text:     {self.has_text_field}",
            f"  prompt:   {self.has_prompt_field}",
            f"  response: {self.has_response_field}",
            f"  messages: {self.has_messages_field}",
            "",
            "Length Statistics:",
            f"  Avg prompt:   {self.avg_prompt_length:.0f} chars",
            f"  Avg response: {self.avg_response_length:.0f} chars",
            f"  Max prompt:   {self.max_prompt_length} chars",
            f"  Max response: {self.max_response_length} chars",
        ]
        
        if self.errors:
            lines.extend(["", "Errors:"])
            for err in self.errors[:10]:
                lines.append(f"  ✗ {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
        
        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warn in self.warnings[:10]:
                lines.append(f"  ! {warn}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DatasetValidator:
    """
    Validate training dataset format and content.
    
    Supports:
    - SFT format: {"text": "..."}
    - Prompt/Response format: {"prompt": "...", "response": "..."}
    - Messages format: {"messages": [{"role": "...", "content": "..."}]}
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.examples: List[Dict] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> ValidationResult:
        """Run full validation and return results."""
        
        # Check file exists
        if not self.file_path.exists():
            return ValidationResult(
                valid=False,
                total_examples=0,
                valid_examples=0,
                invalid_examples=0,
                detected_format="unknown",
                has_text_field=0,
                has_prompt_field=0,
                has_response_field=0,
                has_messages_field=0,
                avg_prompt_length=0,
                avg_response_length=0,
                max_prompt_length=0,
                max_response_length=0,
                errors=[f"File not found: {self.file_path}"],
                warnings=[]
            )
        
        # Load and parse
        self._load_file()
        
        if not self.examples:
            return ValidationResult(
                valid=False,
                total_examples=0,
                valid_examples=0,
                invalid_examples=0,
                detected_format="unknown",
                has_text_field=0,
                has_prompt_field=0,
                has_response_field=0,
                has_messages_field=0,
                avg_prompt_length=0,
                avg_response_length=0,
                max_prompt_length=0,
                max_response_length=0,
                errors=self.errors or ["No valid examples found"],
                warnings=self.warnings
            )
        
        # Analyze format
        format_info = self._analyze_format()
        
        # Calculate stats
        stats = self._calculate_stats()
        
        # Check for issues
        self._check_issues()
        
        valid = len(self.errors) == 0 and format_info['detected_format'] != 'unknown'
        
        return ValidationResult(
            valid=valid,
            total_examples=len(self.examples),
            valid_examples=len(self.examples) - len([e for e in self.errors if 'line' in e.lower()]),
            invalid_examples=len([e for e in self.errors if 'line' in e.lower()]),
            detected_format=format_info['detected_format'],
            has_text_field=format_info['has_text'],
            has_prompt_field=format_info['has_prompt'],
            has_response_field=format_info['has_response'],
            has_messages_field=format_info['has_messages'],
            avg_prompt_length=stats['avg_prompt'],
            avg_response_length=stats['avg_response'],
            max_prompt_length=stats['max_prompt'],
            max_response_length=stats['max_response'],
            errors=self.errors,
            warnings=self.warnings
        )
    
    def _load_file(self):
        """Load and parse JSONL file."""
        try:
            with open(self.file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.examples.append(obj)
                    except json.JSONDecodeError as e:
                        self.errors.append(f"Line {i}: Invalid JSON - {e}")
        except Exception as e:
            self.errors.append(f"Failed to read file: {e}")
    
    def _analyze_format(self) -> Dict:
        """Analyze the format of examples."""
        has_text = sum(1 for ex in self.examples if 'text' in ex)
        has_prompt = sum(1 for ex in self.examples if 'prompt' in ex)
        has_response = sum(1 for ex in self.examples if 'response' in ex)
        has_messages = sum(1 for ex in self.examples if 'messages' in ex)
        
        # Determine format
        total = len(self.examples)
        if has_text > total * 0.9:
            detected = 'sft'
        elif has_prompt > total * 0.9 and has_response > total * 0.5:
            detected = 'prompt_response'
        elif has_prompt > total * 0.9:
            detected = 'prompts_only'
        elif has_messages > total * 0.9:
            detected = 'messages'
        else:
            detected = 'unknown'
        
        return {
            'detected_format': detected,
            'has_text': has_text,
            'has_prompt': has_prompt,
            'has_response': has_response,
            'has_messages': has_messages
        }
    
    def _calculate_stats(self) -> Dict:
        """Calculate length statistics."""
        prompt_lengths = []
        response_lengths = []
        
        for ex in self.examples:
            # Get prompt length
            if 'prompt' in ex:
                prompt_lengths.append(len(ex['prompt']))
            elif 'text' in ex:
                # For SFT format, estimate prompt from user section
                text = ex['text']
                if '<|im_start|>user' in text:
                    user_section = text.split('<|im_start|>user')[-1].split('<|im_end|>')[0]
                    prompt_lengths.append(len(user_section))
            elif 'messages' in ex:
                for msg in ex['messages']:
                    if msg.get('role') == 'user':
                        prompt_lengths.append(len(msg.get('content', '')))
            
            # Get response length
            if 'response' in ex:
                response_lengths.append(len(ex['response']))
            elif 'text' in ex:
                text = ex['text']
                if '<|im_start|>assistant' in text:
                    asst_section = text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0]
                    response_lengths.append(len(asst_section))
            elif 'messages' in ex:
                for msg in ex['messages']:
                    if msg.get('role') == 'assistant':
                        response_lengths.append(len(msg.get('content', '')))
        
        return {
            'avg_prompt': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            'avg_response': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'max_prompt': max(prompt_lengths) if prompt_lengths else 0,
            'max_response': max(response_lengths) if response_lengths else 0
        }
    
    def _check_issues(self):
        """Check for common issues."""
        for i, ex in enumerate(self.examples, 1):
            # Empty content
            if 'text' in ex and not ex['text'].strip():
                self.errors.append(f"Line {i}: Empty 'text' field")
            if 'prompt' in ex and not ex['prompt'].strip():
                self.warnings.append(f"Line {i}: Empty 'prompt' field")
            if 'response' in ex and not ex['response'].strip():
                self.warnings.append(f"Line {i}: Empty 'response' field")
            
            # Very short content
            if 'prompt' in ex and len(ex['prompt']) < 10:
                self.warnings.append(f"Line {i}: Very short prompt ({len(ex['prompt'])} chars)")
            if 'response' in ex and len(ex['response']) < 10:
                self.warnings.append(f"Line {i}: Very short response ({len(ex['response'])} chars)")
            
            # Very long content (may exceed context)
            if 'text' in ex and len(ex['text']) > 16000:
                self.warnings.append(f"Line {i}: Very long text ({len(ex['text'])} chars) - may exceed context")
    
    def preview(self, n: int = 3) -> str:
        """Return a preview of the first n examples."""
        if not self.examples:
            self._load_file()
        
        lines = [
            "=" * 60,
            "DATASET PREVIEW",
            "=" * 60,
            ""
        ]
        
        for i, ex in enumerate(self.examples[:n], 1):
            lines.append(f"--- Example {i} ---")
            
            if 'text' in ex:
                preview = ex['text'][:500]
                lines.append(f"Text: {preview}...")
            elif 'prompt' in ex:
                prompt_preview = ex['prompt'][:300]
                lines.append(f"Prompt: {prompt_preview}...")
                if 'response' in ex:
                    response_preview = ex['response'][:300]
                    lines.append(f"Response: {response_preview}...")
            elif 'messages' in ex:
                for msg in ex['messages'][:3]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]
                    lines.append(f"  [{role}]: {content}...")
            
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_dataset(file_path: str, preview: bool = False) -> ValidationResult:
    """
    Validate a dataset file.
    
    Args:
        file_path: Path to JSONL file
        preview: If True, also print preview
        
    Returns:
        ValidationResult with stats and issues
    """
    validator = DatasetValidator(file_path)
    result = validator.validate()
    
    print(result)
    
    if preview:
        print()
        print(validator.preview())
    
    return result

