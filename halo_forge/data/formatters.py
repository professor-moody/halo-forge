"""
Chat Template Formatters

Format training examples for different model architectures.
Supports Qwen, Llama, and ChatML formats.
"""

from typing import Dict, List, Optional


def format_for_training(
    prompt: str,
    response: str,
    system_prompt: str = "You are an expert programmer.",
    template: str = "qwen",
    include_thinking: bool = False,
    thinking: Optional[str] = None
) -> str:
    """
    Format a prompt/response pair for training.
    
    Args:
        prompt: User prompt/question
        response: Model response/code
        system_prompt: System instruction
        template: Chat template format (qwen, llama, chatml)
        include_thinking: Wrap response in <think> tags
        thinking: Optional separate thinking content
        
    Returns:
        Formatted text for training
    """
    # Handle thinking/reasoning
    if include_thinking and thinking:
        formatted_response = f"<think>\n{thinking}\n</think>\n\n{response}"
    elif include_thinking:
        formatted_response = f"<code>\n{response}\n</code>"
    else:
        formatted_response = response
    
    if template == "qwen":
        return format_qwen(prompt, formatted_response, system_prompt)
    elif template == "llama":
        return format_llama(prompt, formatted_response, system_prompt)
    elif template == "chatml":
        return format_chatml(prompt, formatted_response, system_prompt)
    else:
        raise ValueError(f"Unknown template: {template}. Use 'qwen', 'llama', or 'chatml'")


def format_qwen(prompt: str, response: str, system_prompt: str) -> str:
    """
    Format for Qwen models.
    
    Uses <|im_start|> and <|im_end|> tokens.
    """
    return f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>"""


def format_llama(prompt: str, response: str, system_prompt: str) -> str:
    """
    Format for Llama models.
    
    Uses [INST] and <<SYS>> tags.
    """
    return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
{response}</s>"""


def format_chatml(prompt: str, response: str, system_prompt: str) -> str:
    """
    Format for ChatML standard.
    
    Used by many models including OpenAI-style fine-tunes.
    """
    return f"""<|system|>
{system_prompt}<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>
{response}<|end|>"""


def format_conversation(
    messages: List[Dict[str, str]],
    template: str = "qwen"
) -> str:
    """
    Format a multi-turn conversation.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        template: Chat template format
        
    Returns:
        Formatted conversation string
    """
    if template == "qwen":
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        return "\n".join(parts)
    
    elif template == "llama":
        result = "<s>"
        system_content = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_content = content
            elif role == "user":
                if system_content:
                    result += f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{content} [/INST]\n"
                    system_content = ""
                else:
                    result += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                result += f"{content}</s>"
        
        return result
    
    elif template == "chatml":
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}<|end|>")
        return "\n".join(parts)
    
    else:
        raise ValueError(f"Unknown template: {template}")


# System prompts for different domains
SYSTEM_PROMPTS = {
    "code_general": "You are an expert programmer. Write clean, efficient, and well-documented code.",
    
    "code_cpp": "You are an expert C++ programmer. Write modern, efficient C++ code with proper error handling.",
    
    "code_python": "You are an expert Python programmer. Write clean, Pythonic code following PEP 8 guidelines.",
    
    "code_rust": "You are an expert Rust programmer. Write safe, efficient Rust code with proper error handling.",
    
    "code_windows": "You are an expert Windows systems programmer. Write correct Win32 API code with proper error handling.",
    
    "competitive": "You are an expert competitive programmer. Solve problems efficiently with optimal time and space complexity.",
    
    "security": "You are an expert security researcher and systems programmer specializing in Windows internals.",
}


def get_system_prompt(domain: str) -> str:
    """Get a system prompt for a domain."""
    return SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["code_general"])

