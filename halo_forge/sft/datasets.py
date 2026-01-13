"""
SFT Dataset Loaders

Provides unified access to HuggingFace datasets for Supervised Fine-Tuning.
Supports short names and uses tokenizer's native chat template for proper formatting.

IMPORTANT: Always pass a tokenizer to load_sft_dataset() to ensure the correct
BOS token and chat template are used. Different models have different formats:
- LFM2.5: Requires <|startoftext|> prefix
- Qwen: Uses ChatML without BOS
- Llama: Uses different template entirely

Without a tokenizer, we fall back to hardcoded ChatML which may not match the
model's expected format, causing training/inference mismatches.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from functools import partial
from datasets import load_dataset, Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetSpec:
    """Specification for an SFT dataset."""
    name: str
    huggingface_id: str
    description: str
    domain: str  # code, vlm, audio, reasoning, agentic
    default_split: str = "train"
    subset: Optional[str] = None
    formatter: Optional[str] = None  # Name of formatting function
    size_hint: str = ""  # e.g., "20K", "395K"


# Registry of known SFT datasets
SFT_DATASETS: Dict[str, SFTDatasetSpec] = {
    # Code domain
    "codealpaca": SFTDatasetSpec(
        name="codealpaca",
        huggingface_id="sahil2801/CodeAlpaca-20k",
        description="20K instruction-following code examples",
        domain="code",
        formatter="format_alpaca",
        size_hint="20K"
    ),
    "code_instructions_122k": SFTDatasetSpec(
        name="code_instructions_122k",
        huggingface_id="TokenBender/code_instructions_122k",
        description="122K code instruction examples",
        domain="code",
        formatter="format_alpaca",
        size_hint="122K"
    ),
    "python_instructions": SFTDatasetSpec(
        name="python_instructions",
        huggingface_id="iamtarun/python_code_instructions_18k_alpaca",
        description="18K Python-specific instructions",
        domain="code",
        formatter="format_alpaca",
        size_hint="18K"
    ),
    
    # Reasoning domain
    "metamath": SFTDatasetSpec(
        name="metamath",
        huggingface_id="meta-math/MetaMathQA",
        description="395K math problems with chain-of-thought solutions",
        domain="reasoning",
        formatter="format_metamath",
        size_hint="395K"
    ),
    "gsm8k_sft": SFTDatasetSpec(
        name="gsm8k_sft",
        huggingface_id="gsm8k",
        subset="main",
        description="8.5K grade school math for SFT",
        domain="reasoning",
        formatter="format_gsm8k",
        size_hint="8.5K"
    ),
    
    # VLM domain
    "llava": SFTDatasetSpec(
        name="llava",
        huggingface_id="liuhaotian/LLaVA-Instruct-150K",
        description="150K visual instruction tuning examples",
        domain="vlm",
        formatter="format_llava",
        size_hint="150K"
    ),
    
    # Audio domain
    "librispeech_sft": SFTDatasetSpec(
        name="librispeech_sft",
        huggingface_id="librispeech_asr",
        subset="train.clean.100",
        description="100h clean English speech for ASR SFT",
        domain="audio",
        formatter="format_asr",
        size_hint="100h"
    ),
    "common_voice_sft": SFTDatasetSpec(
        name="common_voice_sft",
        huggingface_id="mozilla-foundation/common_voice_17_0",
        subset="en",
        description="Crowdsourced English speech",
        domain="audio",
        formatter="format_asr",
        size_hint="varies"
    ),
    
    # Agentic domain
    "xlam_sft": SFTDatasetSpec(
        name="xlam_sft",
        huggingface_id="Salesforce/xlam-function-calling-60k",
        description="60K function calling examples",
        domain="agentic",
        formatter="format_xlam",
        size_hint="60K"
    ),
    "glaive_sft": SFTDatasetSpec(
        name="glaive_sft",
        huggingface_id="glaiveai/glaive-function-calling-v2",
        description="113K function calling with irrelevance detection",
        domain="agentic",
        formatter="format_glaive",
        size_hint="113K"
    ),
}


def format_to_chatml(
    instruction: str,
    input_text: str = "",
    output: str = "",
    system_prompt: str = "You are a helpful assistant."
) -> str:
    """
    Format to ChatML format (FALLBACK - use format_with_tokenizer when possible).
    
    WARNING: This hardcodes ChatML tokens without BOS. Many models require
    additional tokens (e.g., LFM2.5 needs <|startoftext|>). Use tokenizer's
    apply_chat_template() for production training.
    """
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    if input_text:
        text += f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n"
    else:
        text += f"<|im_start|>user\n{instruction}<|im_end|>\n"
    
    text += f"<|im_start|>assistant\n{output}<|im_end|>"
    
    return text


def format_with_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, str]]
) -> str:
    """
    Format messages using tokenizer's native chat template.
    
    This ensures correct BOS tokens, special tokens, and format for the
    specific model being trained.
    
    Args:
        tokenizer: The model's tokenizer with chat_template defined
        messages: List of {"role": "...", "content": "..."} dicts
        
    Returns:
        Properly formatted text string with all required tokens
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # Include full conversation for training
    )


def format_alpaca(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format Alpaca-style datasets (instruction, input, output)."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Build user content
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    if tokenizer is not None:
        # Use tokenizer's native format (correct BOS, special tokens)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        return {"text": format_with_tokenizer(tokenizer, messages)}
    else:
        # Fallback to hardcoded ChatML (may lack BOS token)
        return {"text": format_to_chatml(instruction, input_text, output)}


def format_metamath(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format MetaMathQA dataset."""
    query = example.get("query", "")
    response = example.get("response", "")
    
    system = "You are a helpful math tutor. Solve problems step by step."
    
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        return {"text": format_with_tokenizer(tokenizer, messages)}
    else:
        return {"text": format_to_chatml(query, "", response, system)}


def format_gsm8k(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format GSM8K dataset for SFT."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    system = "You are a helpful math tutor. Solve problems step by step and put your final answer in \\boxed{}."
    
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        return {"text": format_with_tokenizer(tokenizer, messages)}
    else:
        return {"text": format_to_chatml(question, "", answer, system)}


def format_llava(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format LLaVA instruction dataset."""
    # LLaVA has conversations format
    conversations = example.get("conversations", [])
    
    if not conversations:
        return {"text": ""}
    
    # Build messages list
    messages = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
    
    for conv in conversations:
        role = "user" if conv.get("from") == "human" else "assistant"
        content = conv.get("value", "")
        messages.append({"role": role, "content": content})
    
    if tokenizer is not None:
        return {"text": format_with_tokenizer(tokenizer, messages)}
    else:
        # Fallback to hardcoded ChatML
        text = "<|im_start|>system\nYou are a helpful vision-language assistant.<|im_end|>\n"
        for conv in conversations:
            role = "user" if conv.get("from") == "human" else "assistant"
            content = conv.get("value", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text.strip()}


def format_asr(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format ASR dataset for Whisper-style training."""
    # Audio datasets have different structure - return transcript
    # No chat formatting needed for ASR
    text = example.get("text", example.get("sentence", ""))
    return {"text": text}


def format_xlam(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format xLAM function calling dataset."""
    query = example.get("query", "")
    tools = example.get("tools", "")
    answer = example.get("answers", "")
    
    system = f"""You are a helpful assistant with access to the following tools:
{tools}

Use tools by responding with <tool_call>{{"name": "tool_name", "arguments": {{}}}}</tool_call>"""
    
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(answer)}
        ]
        return {"text": format_with_tokenizer(tokenizer, messages)}
    else:
        return {"text": format_to_chatml(query, "", str(answer), system)}


def format_glaive(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """Format Glaive function calling dataset."""
    # Glaive uses system/user/assistant format already embedded in chat
    system = example.get("system", "You are a helpful assistant.")
    chat = example.get("chat", "")
    
    # For Glaive, the chat field already has the conversation
    # We can't easily parse it back to messages, so we construct manually
    if tokenizer is not None:
        # Try to use tokenizer's BOS token at minimum
        bos = getattr(tokenizer, 'bos_token', '') or ''
        text = f"{bos}<|im_start|>system\n{system}<|im_end|>\n{chat}"
        return {"text": text}
    else:
        text = f"<|im_start|>system\n{system}<|im_end|>\n{chat}"
        return {"text": text}


def format_messages(example: Dict[str, Any], tokenizer: Optional["PreTrainedTokenizer"] = None) -> Dict[str, str]:
    """
    Format a dataset with 'messages' column (list of {role, content} dicts).
    
    This is the standard format used by many SFT datasets including:
    - OpenAI fine-tuning format
    - ShareGPT-style datasets
    - Many HuggingFace chat datasets
    """
    messages = example.get("messages", [])
    
    if not messages:
        return {"text": ""}
    
    # Try to use tokenizer's chat template if available
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        except Exception:
            pass  # Fall back to manual formatting
    
    # Manual ChatML formatting
    bos = getattr(tokenizer, 'bos_token', '') if tokenizer else ''
    parts = [bos] if bos else []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return {"text": "\n".join(parts)}


# Formatter registry
FORMATTERS: Dict[str, Callable] = {
    "format_alpaca": format_alpaca,
    "format_metamath": format_metamath,
    "format_gsm8k": format_gsm8k,
    "format_llava": format_llava,
    "format_asr": format_asr,
    "format_xlam": format_xlam,
    "format_glaive": format_glaive,
    "format_messages": format_messages,
}


def list_sft_datasets(domain: Optional[str] = None) -> List[SFTDatasetSpec]:
    """List available SFT datasets, optionally filtered by domain."""
    datasets = list(SFT_DATASETS.values())
    if domain:
        datasets = [d for d in datasets if d.domain == domain]
    return datasets


def get_sft_dataset_spec(name: str) -> Optional[SFTDatasetSpec]:
    """Get dataset specification by name."""
    return SFT_DATASETS.get(name)


def is_huggingface_id(name: str) -> bool:
    """Check if name looks like a HuggingFace dataset ID (contains /)."""
    # HuggingFace IDs look like "owner/dataset" but NOT like file paths
    if "/" not in name:
        return False
    # File paths typically have extensions or multiple slashes
    if name.endswith(('.json', '.jsonl', '.csv', '.parquet')):
        return False
    # Check if it looks more like a path than a HF ID
    if name.count('/') > 1 or name.startswith(('.', '/')):
        return False
    return True


def is_local_file(path: str) -> bool:
    """Check if path is a local file that exists."""
    import os
    from pathlib import Path as FilePath
    return os.path.isfile(path) or FilePath(path).is_file()


def load_sft_dataset(
    name_or_id: str,
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    tokenizer: Optional["PreTrainedTokenizer"] = None
) -> Dataset:
    """
    Load an SFT dataset by short name or HuggingFace ID.
    
    Args:
        name_or_id: Short name (e.g., 'codealpaca') or HuggingFace ID
        max_samples: Maximum number of samples to load
        split: Dataset split to use (overrides default)
        streaming: Whether to use streaming mode
        tokenizer: Model tokenizer for proper chat template formatting.
                   IMPORTANT: Pass this to ensure correct BOS tokens and format!
        
    Returns:
        HuggingFace Dataset with 'text' column formatted for SFT
    """
    if tokenizer is None:
        logger.warning(
            "No tokenizer provided to load_sft_dataset(). "
            "Using fallback ChatML format which may lack proper BOS tokens. "
            "For production training, pass the model's tokenizer."
        )
    
    spec = get_sft_dataset_spec(name_or_id)
    
    if spec:
        # Known dataset with short name
        logger.info(f"Loading SFT dataset: {spec.name} ({spec.huggingface_id})")
        
        load_split = split or spec.default_split
        
        if spec.subset:
            ds = load_dataset(
                spec.huggingface_id,
                spec.subset,
                split=load_split,
                streaming=streaming,
                trust_remote_code=True
            )
        else:
            ds = load_dataset(
                spec.huggingface_id,
                split=load_split,
                streaming=streaming,
                trust_remote_code=True
            )
        
        # Apply formatter if specified
        if spec.formatter and spec.formatter in FORMATTERS:
            formatter = FORMATTERS[spec.formatter]
            logger.info(f"Applying formatter: {spec.formatter}")
            
            # Create partial function with tokenizer bound
            formatter_with_tok = partial(formatter, tokenizer=tokenizer)
            
            if streaming:
                ds = ds.map(formatter_with_tok)
            else:
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
        
    elif is_local_file(name_or_id):
        # Local JSONL/JSON file
        logger.info(f"Loading local dataset file: {name_or_id}")
        
        if name_or_id.endswith('.jsonl') or name_or_id.endswith('.json'):
            ds = load_dataset("json", data_files=name_or_id, split="train")
        elif name_or_id.endswith('.csv'):
            ds = load_dataset("csv", data_files=name_or_id, split="train")
        elif name_or_id.endswith('.parquet'):
            ds = load_dataset("parquet", data_files=name_or_id, split="train")
        else:
            raise ValueError(
                f"Unsupported local file format: {name_or_id}\n"
                f"Supported formats: .json, .jsonl, .csv, .parquet"
            )
        
        # Try to auto-detect format and apply formatter
        if len(ds) > 0:
            sample = ds[0]
            # Check if already has 'text' column (pre-formatted)
            if "text" in sample:
                logger.info("Dataset already has 'text' column, using as-is")
            elif "instruction" in sample and "output" in sample:
                logger.info("Detected Alpaca format, applying formatter")
                formatter_with_tok = partial(format_alpaca, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            elif "query" in sample and "response" in sample:
                logger.info("Detected MetaMath format, applying formatter")
                formatter_with_tok = partial(format_metamath, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            elif "question" in sample and "answer" in sample:
                logger.info("Detected QA format, applying formatter")
                formatter_with_tok = partial(format_gsm8k, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            elif "messages" in sample:
                logger.info("Detected messages format (ChatML), applying formatter")
                formatter_with_tok = partial(format_messages, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            else:
                logger.warning(
                    f"Could not auto-detect format for local file. "
                    f"Available columns: {list(sample.keys())}. "
                    f"Expected one of: text, instruction+output, query+response, question+answer, messages"
                )
    
    elif is_huggingface_id(name_or_id):
        # Direct HuggingFace ID
        logger.info(f"Loading HuggingFace dataset: {name_or_id}")
        
        load_split = split or "train"
        ds = load_dataset(
            name_or_id,
            split=load_split,
            streaming=streaming,
            trust_remote_code=True
        )
        
        # Try to auto-detect format and apply formatter
        if not streaming and len(ds) > 0:
            sample = ds[0]
            if "instruction" in sample and "output" in sample:
                logger.info("Detected Alpaca format, applying formatter")
                formatter_with_tok = partial(format_alpaca, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            elif "query" in sample and "response" in sample:
                logger.info("Detected MetaMath format, applying formatter")
                formatter_with_tok = partial(format_metamath, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
            elif "question" in sample and "answer" in sample:
                logger.info("Detected QA format, applying formatter")
                formatter_with_tok = partial(format_gsm8k, tokenizer=tokenizer)
                ds = ds.map(formatter_with_tok, remove_columns=ds.column_names)
    else:
        raise ValueError(
            f"Unknown dataset: {name_or_id}\n"
            f"Options:\n"
            f"  - Short name: {list(SFT_DATASETS.keys())[:5]}...\n"
            f"  - HuggingFace ID: e.g., 'sahil2801/CodeAlpaca-20k'\n"
            f"  - Local file: e.g., 'data/train.jsonl'"
        )
    
    # Limit samples if requested
    if max_samples and not streaming:
        if len(ds) > max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples")
            ds = ds.shuffle(seed=42).select(range(max_samples))
    
    return ds


def get_default_sft_dataset(domain: str) -> str:
    """Get the default SFT dataset for a domain."""
    defaults = {
        "code": "codealpaca",
        "vlm": "llava",
        "audio": "librispeech_sft",
        "reasoning": "metamath",
        "agentic": "xlam_sft",
    }
    return defaults.get(domain, "codealpaca")
