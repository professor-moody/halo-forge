"""
Tool Calling Verifier

Provides deterministic reward signals for tool calling training:
- JSON structure validation
- Schema compliance checking
- Function name matching
- Argument validation
- Optional execution-based verification
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCallingVerifyConfig:
    """Configuration for tool calling verification."""
    
    # Reward values
    reward_correct: float = 1.0         # Correct function + arguments
    reward_correct_no_exec: float = 0.75  # Correct without execution test
    reward_correct_func: float = 0.5    # Correct function, wrong args
    reward_valid_json: float = 0.25     # Valid JSON, wrong function
    reward_no_call: float = 0.0         # No tool call when expected
    reward_false_positive: float = -0.25  # Called when shouldn't
    
    # Validation settings
    strict_schema: bool = False  # Require exact schema match
    allow_extra_args: bool = True  # Allow extra arguments
    case_sensitive: bool = True  # Case-sensitive function names
    
    # Execution settings
    execution_timeout: float = 5.0  # Timeout for execution tests
    max_workers: int = 8


@dataclass
class ToolCallVerifyResult(VerifyResult):
    """Extended result with tool-specific details."""
    
    json_valid: bool = False
    schema_valid: bool = False
    execution_success: Optional[bool] = None
    called_correct_function: bool = False
    arguments_correct: bool = False
    parsed_calls: List[Dict] = field(default_factory=list)
    expected_calls: List[Dict] = field(default_factory=list)


class ToolCallingVerifier(Verifier):
    """
    Verify tool calling outputs with multi-level rewards.
    
    Reward structure:
      1.0  - Correct function, correct arguments, executes successfully
      0.75 - Correct function, correct arguments, no execution test
      0.5  - Correct function, wrong/missing arguments
      0.25 - Valid JSON tool call, wrong function
      0.0  - Invalid JSON or no tool call when expected
     -0.25 - Tool call when none should be made (false positive)
    
    Example:
        verifier = ToolCallingVerifier()
        result = verifier.verify(
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
            expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}]
        )
    """
    
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL
    )
    
    # Alternative patterns for different formats
    ALT_PATTERNS = [
        re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL),
        re.compile(r'<function_call>\s*(\{.*?\})\s*</function_call>', re.DOTALL),
        re.compile(r'<functioncall>\s*(\{.*?\})\s*</functioncall>', re.DOTALL),
    ]
    
    def __init__(
        self,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        executor: Optional[Callable[[str, Dict], Any]] = None,
        config: Optional[ToolCallingVerifyConfig] = None,
    ):
        """Initialize the verifier.
        
        Args:
            available_tools: List of available tool schemas for validation.
            executor: Optional function to execute tool calls.
            config: Verification configuration.
        """
        self.config = config or ToolCallingVerifyConfig()
        super().__init__(max_workers=self.config.max_workers)
        
        self.available_tools = {}
        if available_tools:
            for tool in available_tools:
                if "function" in tool:
                    name = tool["function"]["name"]
                    self.available_tools[name] = tool
                elif "name" in tool:
                    self.available_tools[tool["name"]] = {"function": tool}
        
        self.executor = executor
    
    def verify(
        self,
        output: str,
        expected_calls: Optional[List[Dict]] = None,
        is_irrelevant: bool = False,
        **kwargs
    ) -> ToolCallVerifyResult:
        """
        Verify a tool calling output.
        
        Args:
            output: Model output string.
            expected_calls: List of expected tool calls.
            is_irrelevant: True if no tool should be called.
            
        Returns:
            ToolCallVerifyResult with reward and details.
        """
        expected_calls = expected_calls or []
        
        # Extract tool calls from output
        parsed_calls = self._extract_tool_calls(output)
        
        # Case: No tool should be called (irrelevance detection)
        if is_irrelevant:
            if parsed_calls:
                return ToolCallVerifyResult(
                    success=False,
                    reward=self.config.reward_false_positive,
                    details="False positive: called tool when none needed",
                    json_valid=True,
                    called_correct_function=False,
                    parsed_calls=parsed_calls,
                    expected_calls=expected_calls,
                )
            else:
                return ToolCallVerifyResult(
                    success=True,
                    reward=self.config.reward_correct,
                    details="Correctly declined to call tools",
                    json_valid=True,
                    parsed_calls=[],
                    expected_calls=[],
                )
        
        # Case: Tool should be called but wasn't
        if not parsed_calls:
            return ToolCallVerifyResult(
                success=False,
                reward=self.config.reward_no_call,
                details="No tool call found in output",
                json_valid=False,
                parsed_calls=[],
                expected_calls=expected_calls,
            )
        
        # Case: No expected calls to compare against
        if not expected_calls:
            return ToolCallVerifyResult(
                success=True,
                reward=self.config.reward_valid_json,
                details=f"Found {len(parsed_calls)} valid tool call(s), no expected to compare",
                json_valid=True,
                parsed_calls=parsed_calls,
                expected_calls=[],
            )
        
        # Validate each parsed call against expected
        total_reward = 0.0
        all_correct = True
        call_details = []
        
        # Match parsed calls to expected calls
        matched_expected = set()
        
        for parsed in parsed_calls:
            best_match = None
            best_reward = 0.0
            best_idx = -1
            
            for idx, expected in enumerate(expected_calls):
                if idx in matched_expected:
                    continue
                
                result = self._verify_single_call(parsed, expected)
                if result.reward > best_reward:
                    best_reward = result.reward
                    best_match = result
                    best_idx = idx
            
            if best_match:
                matched_expected.add(best_idx)
                total_reward += best_match.reward
                call_details.append(best_match.details)
                if not best_match.success:
                    all_correct = False
            else:
                # Extra call with no match
                total_reward += self.config.reward_valid_json
                call_details.append("Extra tool call with no expected match")
                all_correct = False
        
        # Penalty for missing expected calls
        missing = len(expected_calls) - len(matched_expected)
        if missing > 0:
            all_correct = False
            call_details.append(f"Missing {missing} expected call(s)")
        
        # Average reward across all calls
        total_calls = max(len(expected_calls), len(parsed_calls))
        avg_reward = total_reward / total_calls if total_calls > 0 else 0.0
        
        # Penalty for wrong number of calls
        if len(parsed_calls) != len(expected_calls):
            avg_reward *= 0.8
        
        return ToolCallVerifyResult(
            success=all_correct,
            reward=max(0.0, min(1.0, avg_reward)),  # Clamp to [0, 1]
            details="; ".join(call_details) if call_details else f"Verified {len(parsed_calls)} call(s)",
            json_valid=True,
            called_correct_function=all_correct,
            arguments_correct=all_correct,
            parsed_calls=parsed_calls,
            expected_calls=expected_calls,
        )
    
    def _extract_tool_calls(self, output: str) -> List[Dict]:
        """Extract and parse tool calls from output."""
        calls = []
        
        # Try primary pattern first
        for match in self.TOOL_CALL_PATTERN.finditer(output):
            try:
                call = json.loads(match.group(1))
                if self._is_valid_call(call):
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        
        # Try alternative patterns if primary found nothing
        if not calls:
            for pattern in self.ALT_PATTERNS:
                for match in pattern.finditer(output):
                    try:
                        call = json.loads(match.group(1))
                        if self._is_valid_call(call):
                            calls.append(call)
                    except json.JSONDecodeError:
                        continue
                if calls:
                    break
        
        # Last resort: look for bare JSON with name field
        if not calls:
            json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+[^{}]*\}')
            for match in json_pattern.finditer(output):
                try:
                    call = json.loads(match.group())
                    if self._is_valid_call(call):
                        calls.append(call)
                except json.JSONDecodeError:
                    continue
        
        return calls
    
    def _is_valid_call(self, call: Dict) -> bool:
        """Check if a parsed dict is a valid tool call."""
        if not isinstance(call, dict):
            return False
        return "name" in call
    
    def _verify_single_call(self, parsed: Dict, expected: Dict) -> VerifyResult:
        """Verify a single tool call against expected.
        
        Args:
            parsed: Parsed tool call from output.
            expected: Expected tool call.
            
        Returns:
            VerifyResult for this single call.
        """
        parsed_name = parsed.get("name", "")
        expected_name = expected.get("name", "")
        
        # Compare function names
        if self.config.case_sensitive:
            names_match = parsed_name == expected_name
        else:
            names_match = parsed_name.lower() == expected_name.lower()
        
        if not names_match:
            return VerifyResult(
                success=False,
                reward=self.config.reward_valid_json,
                details=f"Wrong function: {parsed_name} vs {expected_name}"
            )
        
        # Check arguments
        parsed_args = parsed.get("arguments", {})
        expected_args = expected.get("arguments", {})
        
        # Handle string arguments (some datasets store as JSON string)
        if isinstance(parsed_args, str):
            try:
                parsed_args = json.loads(parsed_args)
            except json.JSONDecodeError:
                parsed_args = {}
        
        if isinstance(expected_args, str):
            try:
                expected_args = json.loads(expected_args)
            except json.JSONDecodeError:
                expected_args = {}
        
        # Compare arguments
        if parsed_args == expected_args:
            # Full match
            if self.executor:
                try:
                    self.executor(parsed_name, parsed_args)
                    return VerifyResult(
                        success=True,
                        reward=self.config.reward_correct,
                        details="Execution successful"
                    )
                except Exception as e:
                    return VerifyResult(
                        success=False,
                        reward=self.config.reward_correct_no_exec,
                        details=f"Execution failed: {e}"
                    )
            
            return VerifyResult(
                success=True,
                reward=self.config.reward_correct_no_exec,
                details="Structural match (no execution test)"
            )
        
        # Partial credit for some correct arguments
        if expected_args:
            correct_args = sum(
                1 for k, v in expected_args.items()
                if parsed_args.get(k) == v
            )
            partial = correct_args / len(expected_args)
            
            # Interpolate between reward_correct_func and reward_correct_no_exec
            reward = self.config.reward_correct_func + (
                (self.config.reward_correct_no_exec - self.config.reward_correct_func) * partial
            )
            
            return VerifyResult(
                success=False,
                reward=reward,
                details=f"Arguments mismatch: {correct_args}/{len(expected_args)} correct"
            )
        
        return VerifyResult(
            success=False,
            reward=self.config.reward_correct_func,
            details="Arguments mismatch"
        )
    
    def validate_schema(self, call: Dict, tool_name: str) -> bool:
        """Validate call arguments against tool schema.
        
        Args:
            call: Tool call to validate.
            tool_name: Name of the tool.
            
        Returns:
            True if call matches schema.
        """
        if tool_name not in self.available_tools:
            return True  # No schema to validate against
        
        schema = self.available_tools[tool_name].get("function", {}).get("parameters", {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return False
        
        # Check required fields
        for field_name in required:
            if field_name not in args:
                return False
        
        # Check extra args if strict
        if self.config.strict_schema and not self.config.allow_extra_args:
            for key in args:
                if key not in properties:
                    return False
        
        # Check types (basic validation)
        for field_name, value in args.items():
            if field_name in properties:
                expected_type = properties[field_name].get("type")
                if expected_type and not self._type_matches(value, expected_type):
                    return False
        
        return True
    
    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check if value matches JSON schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        if expected_type not in type_map:
            return True
        return isinstance(value, type_map[expected_type])
