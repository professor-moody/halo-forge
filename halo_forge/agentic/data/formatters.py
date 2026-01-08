"""
Hermes Format Conversion

Converts tool calling samples to Hermes format for training.
Hermes format is the standard for Qwen2.5, NousHermes, and most open models.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from halo_forge.agentic.data.loaders import ToolCallSample


@dataclass
class HermesFormatter:
    """Format tool calling samples to Hermes XML tags.
    
    The Hermes format uses:
    - <tools>...</tools> for tool schemas
    - <tool_call>...</tool_call> for function calls
    - <tool_response>...</tool_response> for results
    
    Example:
        formatter = HermesFormatter()
        formatted = formatter.format(sample)
    """
    
    SYSTEM_PREFIX: str = (
        "You are a function calling AI model. You may call one or more "
        "functions to assist with the user query. Don't make assumptions "
        "about what values to plug into functions."
    )
    
    include_tool_response: bool = False
    compact_json: bool = True
    
    def format(self, sample: ToolCallSample) -> str:
        """Format a sample to Hermes format.
        
        Args:
            sample: ToolCallSample to format.
            
        Returns:
            Formatted string in Hermes format.
        """
        output = []
        
        # System message with tools
        tools_json = self._format_tools(sample.tools)
        system_content = f"{self.SYSTEM_PREFIX}\n<tools>\n{tools_json}\n</tools>"
        output.append(f"<|im_start|>system\n{system_content}\n<|im_end|>")
        
        # Conversation messages
        for msg in sample.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool_call":
                # Tool call from assistant
                output.append(
                    f"<|im_start|>assistant\n<tool_call>\n{content}\n</tool_call><|im_end|>"
                )
            elif role == "tool_response" and self.include_tool_response:
                # Tool response
                output.append(
                    f"<|im_start|>tool\n<tool_response>\n{content}\n</tool_response>\n<|im_end|>"
                )
            else:
                # Regular message
                output.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Add expected tool calls as assistant turn if not already present
        if sample.expected_calls and not self._has_tool_call(sample.messages):
            calls_content = self._format_calls(sample.expected_calls)
            output.append(
                f"<|im_start|>assistant\n<tool_call>\n{calls_content}\n</tool_call><|im_end|>"
            )
        
        return "\n".join(output)
    
    def format_prompt(self, sample: ToolCallSample) -> str:
        """Format sample as a prompt (without expected output).
        
        Args:
            sample: ToolCallSample to format.
            
        Returns:
            Prompt string (system + user message only).
        """
        output = []
        
        # System message with tools
        tools_json = self._format_tools(sample.tools)
        system_content = f"{self.SYSTEM_PREFIX}\n<tools>\n{tools_json}\n</tools>"
        output.append(f"<|im_start|>system\n{system_content}\n<|im_end|>")
        
        # Only user messages (for generation)
        for msg in sample.messages:
            if msg.get("role") == "user":
                output.append(f"<|im_start|>user\n{msg.get('content', '')}<|im_end|>")
        
        # Add assistant prefix to prompt generation
        output.append("<|im_start|>assistant\n")
        
        return "\n".join(output)
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools list as JSON."""
        if self.compact_json:
            return json.dumps(tools, separators=(',', ':'))
        return json.dumps(tools, indent=2)
    
    def _format_calls(self, calls: List[Dict[str, Any]]) -> str:
        """Format tool calls as JSON."""
        if len(calls) == 1:
            call = calls[0]
            if self.compact_json:
                return json.dumps(call, separators=(',', ':'))
            return json.dumps(call, indent=2)
        
        # Multiple calls - format each
        formatted = []
        for call in calls:
            if self.compact_json:
                formatted.append(json.dumps(call, separators=(',', ':')))
            else:
                formatted.append(json.dumps(call, indent=2))
        return "\n".join(formatted)
    
    def _has_tool_call(self, messages: List[Dict[str, str]]) -> bool:
        """Check if messages already contain a tool call."""
        for msg in messages:
            if msg.get("role") == "tool_call":
                return True
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "<tool_call>" in content:
                    return True
        return False


def format_to_hermes(
    sample: ToolCallSample,
    include_response: bool = False
) -> str:
    """Convenience function to format a sample to Hermes format.
    
    Args:
        sample: ToolCallSample to format.
        include_response: Whether to include tool responses.
        
    Returns:
        Formatted string.
    """
    formatter = HermesFormatter(include_tool_response=include_response)
    return formatter.format(sample)


def create_training_sample(
    sample: ToolCallSample,
    formatter: Optional[HermesFormatter] = None
) -> Dict[str, str]:
    """Create a training sample dict with input and target.
    
    Args:
        sample: ToolCallSample to convert.
        formatter: Optional formatter to use.
        
    Returns:
        Dict with 'input' and 'target' keys.
    """
    if formatter is None:
        formatter = HermesFormatter()
    
    prompt = formatter.format_prompt(sample)
    
    # Target is the tool call
    if sample.expected_calls:
        target = formatter._format_calls(sample.expected_calls)
        target = f"<tool_call>\n{target}\n</tool_call><|im_end|>"
    else:
        # Irrelevant - no tool call
        target = "I cannot help with that request using the available tools.<|im_end|>"
    
    return {
        "input": prompt,
        "target": target,
        "full": formatter.format(sample),
    }
