"""
Tool Calling Dataset Loaders

Loads and preprocesses function calling datasets for RLVR training.
Standardizes all formats to Hermes-style XML tags.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolCallSample:
    """Single tool calling training sample."""
    
    messages: List[Dict[str, str]]  # Conversation history
    tools: List[Dict[str, Any]]     # Available tool schemas
    expected_calls: List[Dict]       # Expected tool_call outputs
    is_irrelevant: bool = False     # True if no tool should be called
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages": self.messages,
            "tools": self.tools,
            "expected_calls": self.expected_calls,
            "is_irrelevant": self.is_irrelevant,
            "metadata": self.metadata,
        }


# Dataset registry
AGENTIC_DATASETS = {
    "xlam": {
        "name": "xLAM-60k",
        "hf_path": "Salesforce/xlam-function-calling-60k",
        "description": "60k verified samples, 3,673 executable APIs",
        "size": "~60,000",
    },
    "glaive": {
        "name": "Glaive Function Calling v2",
        "hf_path": "glaiveai/glaive-function-calling-v2",
        "description": "113k samples, 7,500 irrelevance detection",
        "size": "~113,000",
    },
    "toolbench": {
        "name": "ToolBench",
        "hf_path": "ToolBench/ToolBench",
        "description": "188k samples, 16,464 RESTful APIs",
        "size": "~188,000",
    },
    "hermes": {
        "name": "Hermes Function Calling v1",
        "hf_path": "NousResearch/hermes-function-calling-v1",
        "description": "Format reference for Qwen/NousHermes",
        "size": "varies",
    },
}


def list_agentic_datasets() -> Dict[str, Dict[str, str]]:
    """Return available agentic/tool calling datasets."""
    return AGENTIC_DATASETS.copy()


class ToolCallingDataset(ABC):
    """Base class for tool calling datasets."""
    
    HERMES_SYSTEM_PREFIX = """You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions."""
    
    @abstractmethod
    def load(self, limit: Optional[int] = None) -> List[ToolCallSample]:
        """Load samples from the dataset."""
        pass
    
    def validate_sample(self, sample: ToolCallSample) -> bool:
        """Validate sample structure and JSON correctness."""
        try:
            # Validate tool schemas
            for tool in sample.tools:
                if "type" not in tool:
                    # Some datasets don't have type, add it
                    continue
                assert tool["type"] == "function"
                assert "function" in tool
                assert "name" in tool["function"]
            
            # Validate expected calls are valid JSON
            for call in sample.expected_calls:
                assert "name" in call
                if "arguments" in call:
                    # Ensure arguments can be serialized
                    if isinstance(call["arguments"], str):
                        json.loads(call["arguments"])
                    else:
                        json.dumps(call["arguments"])
            
            return True
        except (AssertionError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Sample validation failed: {e}")
            return False


class XLAMLoader(ToolCallingDataset):
    """Load Salesforce xLAM-60k dataset.
    
    The gold standard for tool calling training with 60k verified samples
    and 3,673 executable APIs.
    """
    
    def __init__(self, split: str = "train"):
        """Initialize loader.
        
        Args:
            split: Dataset split to load.
        """
        self.split = split
        self.dataset_info = AGENTIC_DATASETS["xlam"]
    
    def load(self, limit: Optional[int] = None) -> List[ToolCallSample]:
        """Load xLAM samples.
        
        Args:
            limit: Maximum number of samples to load.
            
        Returns:
            List of ToolCallSample objects.
        """
        from datasets import load_dataset
        
        logger.info(f"Loading xLAM-60k dataset (split={self.split})...")
        dataset = load_dataset(
            self.dataset_info["hf_path"],
            split=self.split
        )
        
        samples = []
        invalid_count = 0
        
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break
            
            try:
                # xLAM format: query, tools, answers
                tools = item.get("tools", "[]")
                if isinstance(tools, str):
                    tools = json.loads(tools)
                
                answers = item.get("answers", "[]")
                if isinstance(answers, str):
                    answers = json.loads(answers)
                
                # Normalize tool format
                normalized_tools = self._normalize_tools(tools)
                
                sample = ToolCallSample(
                    messages=[{"role": "user", "content": item.get("query", "")}],
                    tools=normalized_tools,
                    expected_calls=answers if isinstance(answers, list) else [answers],
                    is_irrelevant=len(answers) == 0 if isinstance(answers, list) else False,
                    metadata={"source": "xlam-60k", "index": i}
                )
                
                if self.validate_sample(sample):
                    samples.append(sample)
                else:
                    invalid_count += 1
                    
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse xLAM sample {i}: {e}")
                invalid_count += 1
                continue
        
        logger.info(f"Loaded {len(samples)} valid xLAM samples ({invalid_count} invalid)")
        return samples
    
    def _normalize_tools(self, tools: List[Dict]) -> List[Dict]:
        """Normalize tool format to standard schema."""
        normalized = []
        for tool in tools:
            if "function" in tool:
                # Already in correct format
                normalized.append(tool)
            elif "name" in tool:
                # Flat format, wrap it
                normalized.append({
                    "type": "function",
                    "function": tool
                })
            else:
                # Unknown format, skip
                continue
        return normalized


class GlaiveLoader(ToolCallingDataset):
    """Load Glaive Function Calling v2 dataset.
    
    Contains 113k samples including 7,500 irrelevance detection examples.
    """
    
    def __init__(self, split: str = "train"):
        """Initialize loader.
        
        Args:
            split: Dataset split to load.
        """
        self.split = split
        self.dataset_info = AGENTIC_DATASETS["glaive"]
    
    def load(
        self,
        limit: Optional[int] = None,
        include_irrelevant: bool = True
    ) -> List[ToolCallSample]:
        """Load Glaive samples.
        
        Args:
            limit: Maximum number of samples to load.
            include_irrelevant: Whether to include irrelevance samples.
            
        Returns:
            List of ToolCallSample objects.
        """
        from datasets import load_dataset
        
        logger.info(f"Loading Glaive v2 dataset (split={self.split})...")
        dataset = load_dataset(
            self.dataset_info["hf_path"],
            split=self.split
        )
        
        samples = []
        irrelevant_count = 0
        invalid_count = 0
        
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break
            
            try:
                # Parse Glaive format
                system = item.get("system", "")
                chat = item.get("chat", "")
                
                # Detect irrelevance samples
                is_irrelevant = "no function" in system.lower() or not chat
                
                if is_irrelevant:
                    irrelevant_count += 1
                    if not include_irrelevant:
                        continue
                
                # Parse conversation and tools
                messages, tools, expected_calls = self._parse_glaive_format(item)
                
                sample = ToolCallSample(
                    messages=messages,
                    tools=tools,
                    expected_calls=expected_calls,
                    is_irrelevant=is_irrelevant,
                    metadata={"source": "glaive-v2", "index": i}
                )
                
                if self.validate_sample(sample):
                    samples.append(sample)
                else:
                    invalid_count += 1
                    
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.debug(f"Failed to parse Glaive sample {i}: {e}")
                invalid_count += 1
                continue
        
        logger.info(
            f"Loaded {len(samples)} Glaive samples "
            f"({irrelevant_count} irrelevant, {invalid_count} invalid)"
        )
        return samples
    
    def _parse_glaive_format(
        self,
        item: Dict
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Parse Glaive's chat format into structured data.
        
        Args:
            item: Raw dataset item.
            
        Returns:
            Tuple of (messages, tools, expected_calls).
        """
        system = item.get("system", "")
        chat = item.get("chat", "")
        
        # Extract tools from system message
        tools = []
        if "functions" in system.lower():
            # Try to parse function definitions
            try:
                # Glaive often embeds JSON in the system message
                import re
                json_match = re.search(r'\[.*\]', system, re.DOTALL)
                if json_match:
                    tools = json.loads(json_match.group())
                    tools = self._normalize_tools(tools)
            except json.JSONDecodeError:
                pass
        
        # Parse chat into messages
        messages = []
        expected_calls = []
        
        # Simple parsing - split by role markers
        # Glaive uses USER: and ASSISTANT: format
        lines = chat.split("\n")
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("USER:"):
                if current_role:
                    content = " ".join(current_content).strip()
                    if content:
                        messages.append({"role": current_role, "content": content})
                current_role = "user"
                current_content = [line[5:].strip()]
            elif line.startswith("ASSISTANT:"):
                if current_role:
                    content = " ".join(current_content).strip()
                    if content:
                        messages.append({"role": current_role, "content": content})
                current_role = "assistant"
                current_content = [line[10:].strip()]
            elif line.startswith("FUNCTION RESPONSE:"):
                # Skip function responses for now
                current_role = None
                current_content = []
            elif current_role:
                current_content.append(line)
        
        # Add final message
        if current_role and current_content:
            content = " ".join(current_content).strip()
            if content:
                messages.append({"role": current_role, "content": content})
        
        # Extract tool calls from assistant messages
        for msg in messages:
            if msg["role"] == "assistant":
                calls = self._extract_tool_calls(msg["content"])
                expected_calls.extend(calls)
        
        return messages, tools, expected_calls
    
    def _normalize_tools(self, tools: List[Dict]) -> List[Dict]:
        """Normalize tool format to standard schema."""
        normalized = []
        for tool in tools:
            if "function" in tool:
                normalized.append(tool)
            elif "name" in tool:
                normalized.append({
                    "type": "function",
                    "function": tool
                })
        return normalized
    
    def _extract_tool_calls(self, content: str) -> List[Dict]:
        """Extract tool calls from assistant message content."""
        import re
        calls = []
        
        # Look for function call patterns
        # Pattern 1: <functioncall> {"name": "...", "arguments": {...}} </functioncall>
        fc_pattern = re.compile(
            r'<functioncall>\s*(\{.*?\})\s*</functioncall>',
            re.DOTALL
        )
        
        for match in fc_pattern.finditer(content):
            try:
                call = json.loads(match.group(1))
                if "name" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Bare JSON with name/arguments
        if not calls:
            json_pattern = re.compile(r'\{[^{}]*"name"[^{}]*\}')
            for match in json_pattern.finditer(content):
                try:
                    call = json.loads(match.group())
                    if "name" in call:
                        calls.append(call)
                except json.JSONDecodeError:
                    continue
        
        return calls
