"""
LLM-Powered Data Generation

Generate training data using LLM APIs (DeepSeek, Claude, OpenAI, Ollama).
Useful for creating domain-specific training examples.
"""

import json
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Generator
from tqdm import tqdm

from halo_forge.data.formatters import format_for_training


@dataclass
class TopicSpec:
    """Specification for generating training data in a domain."""
    
    name: str                              # Identifier
    description: str                       # What kind of code to generate
    categories: List[str]                  # Categories to cover
    complexity_levels: List[str]           # Difficulty levels
    examples_per_category: int = 20        # Examples per category/complexity combo
    
    # Generation constraints
    required_patterns: List[str] = field(default_factory=list)  # Must include
    forbidden_patterns: List[str] = field(default_factory=list)  # Must not include
    style_guide: Optional[str] = None      # Additional instructions
    
    # Formatting
    system_prompt: str = "You are an expert programmer."


# Built-in topic specifications
TOPIC_REGISTRY: Dict[str, TopicSpec] = {}


def register_topic(spec: TopicSpec):
    """Register a topic specification."""
    TOPIC_REGISTRY[spec.name] = spec
    return spec


# Rust async programming
register_topic(TopicSpec(
    name="rust_async",
    description="Async Rust programming with tokio",
    categories=["http_client", "file_io", "channels", "timers", "tcp_server"],
    complexity_levels=["beginner", "intermediate", "advanced"],
    examples_per_category=20,
    required_patterns=["use tokio"],
    forbidden_patterns=["unwrap()"],
    style_guide="Use anyhow for error handling. Use async fn main with #[tokio::main]."
))

# Python testing
register_topic(TopicSpec(
    name="python_testing",
    description="Python unit testing with pytest",
    categories=["unit_tests", "fixtures", "mocking", "parametrize", "async_tests"],
    complexity_levels=["simple", "medium", "complex"],
    examples_per_category=30,
    required_patterns=["import pytest"],
    style_guide="Use pytest idioms, not unittest. Include docstrings."
))

# C++ algorithms
register_topic(TopicSpec(
    name="cpp_algorithms",
    description="C++ algorithm implementations",
    categories=["sorting", "graphs", "dynamic_programming", "trees", "strings"],
    complexity_levels=["easy", "medium", "hard"],
    examples_per_category=25,
    style_guide="Use modern C++17. Prefer STL containers and algorithms."
))

# Go concurrency
register_topic(TopicSpec(
    name="go_concurrency",
    description="Go concurrent programming patterns",
    categories=["goroutines", "channels", "mutexes", "waitgroups", "contexts"],
    complexity_levels=["simple", "intermediate", "advanced"],
    examples_per_category=20,
    required_patterns=["package main"],
    forbidden_patterns=["time.Sleep"],
    style_guide="Use contexts for cancellation. Properly close channels."
))


class LLMBackend(ABC):
    """Abstract LLM backend for generation."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        pass


class DeepSeekBackend(LLMBackend):
    """
    DeepSeek API backend.
    
    Free tier: 14M tokens/day, 1 request/second.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        import requests
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.requests = requests
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if not response.ok:
            raise Exception(f"DeepSeek API error: {response.text}")
        
        time.sleep(1)  # Rate limit
        return response.json()["choices"][0]["message"]["content"]


class OllamaBackend(LLMBackend):
    """
    Ollama local backend.
    
    Free, runs locally. No API costs.
    """
    
    def __init__(self, model: str = "codellama:13b", host: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.host = host
        self.requests = requests
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            },
            timeout=120
        )
        
        if not response.ok:
            raise Exception(f"Ollama error: {response.text}")
        
        return response.json()["response"]


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text


class TrainingDataGenerator:
    """
    Generate training data using LLM.
    
    Example:
        spec = TOPIC_REGISTRY["rust_async"]
        backend = DeepSeekBackend()
        generator = TrainingDataGenerator(backend, spec)
        generator.generate_all("data/rust_async.jsonl")
    """
    
    def __init__(self, backend: LLMBackend, spec: TopicSpec):
        self.backend = backend
        self.spec = spec
    
    def _build_code_prompt(self, category: str, complexity: str) -> str:
        """Build prompt for generating code."""
        prompt = f"""Generate a complete, compilable code example for: {self.spec.description}

Category: {category}
Complexity: {complexity}

Requirements:
- Complete, working code (not snippets)
- Include all necessary imports/headers
- Add comments explaining the approach
- Handle errors appropriately
- Complexity should match "{complexity}" level
"""
        
        if self.spec.required_patterns:
            prompt += f"\nMust include: {', '.join(self.spec.required_patterns)}"
        
        if self.spec.forbidden_patterns:
            prompt += f"\nDo NOT use: {', '.join(self.spec.forbidden_patterns)}"
        
        if self.spec.style_guide:
            prompt += f"\nStyle: {self.spec.style_guide}"
        
        prompt += "\n\nProvide ONLY the code, no explanations before or after."
        
        return prompt
    
    def _build_user_prompt_prompt(self, category: str, complexity: str) -> str:
        """Build prompt for generating the user prompt."""
        return f"""Generate a realistic user prompt that would ask for {self.spec.description} code.

Category: {category}
Complexity: {complexity}

The prompt should:
- Be specific enough to get working code
- Match the complexity level
- Sound like a real developer request
- Be 1-3 sentences

Output ONLY the prompt text, nothing else."""
    
    def generate_example(self, category: str, complexity: str) -> Optional[Dict]:
        """Generate a single training example."""
        try:
            # Generate user prompt
            user_prompt = self.backend.generate(
                self._build_user_prompt_prompt(category, complexity),
                max_tokens=200
            ).strip()
            
            # Generate code
            code = self.backend.generate(
                self._build_code_prompt(category, complexity),
                max_tokens=2000
            ).strip()
            
            return {
                "prompt": user_prompt,
                "response": code,
                "metadata": {
                    "source": f"llm_generated_{self.spec.name}",
                    "category": category,
                    "complexity": complexity
                }
            }
        except Exception as e:
            print(f"  Error generating example: {e}")
            return None
    
    def generate_all(
        self,
        output_path: str,
        template: str = "qwen",
        checkpoint_interval: int = 100
    ) -> int:
        """
        Generate all examples and save to file.
        
        Args:
            output_path: Output JSONL path
            template: Chat template format
            checkpoint_interval: Save checkpoint every N examples
            
        Returns:
            Number of examples generated
        """
        total = (
            len(self.spec.categories) *
            len(self.spec.complexity_levels) *
            self.spec.examples_per_category
        )
        
        print(f"Generating {total} examples for {self.spec.name}...")
        print(f"Categories: {self.spec.categories}")
        print(f"Complexity levels: {self.spec.complexity_levels}")
        print(f"Examples per combo: {self.spec.examples_per_category}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        examples = []
        failed = 0
        
        with tqdm(total=total, desc="Generating") as pbar:
            for category in self.spec.categories:
                for complexity in self.spec.complexity_levels:
                    for i in range(self.spec.examples_per_category):
                        example = self.generate_example(category, complexity)
                        
                        if example:
                            # Format for training
                            formatted = {
                                "text": format_for_training(
                                    prompt=example["prompt"],
                                    response=example["response"],
                                    system_prompt=self.spec.system_prompt,
                                    template=template
                                ),
                                "metadata": example["metadata"]
                            }
                            examples.append(formatted)
                        else:
                            failed += 1
                        
                        pbar.update(1)
                        
                        # Checkpoint
                        if len(examples) % checkpoint_interval == 0 and examples:
                            self._save_checkpoint(output_file, examples)
        
        # Final save
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        
        print(f"\nGenerated {len(examples)} examples (failed: {failed})")
        print(f"Saved to {output_path}")
        
        return len(examples)
    
    def _save_checkpoint(self, output_file: Path, examples: List[Dict]):
        """Save checkpoint during generation."""
        checkpoint_file = output_file.with_suffix('.checkpoint.jsonl')
        with open(checkpoint_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')


def get_backend(name: str, **kwargs) -> LLMBackend:
    """
    Get LLM backend by name.
    
    Args:
        name: Backend name (deepseek, ollama, openai, anthropic)
        **kwargs: Backend-specific arguments
        
    Returns:
        LLMBackend instance
    """
    backends = {
        "deepseek": DeepSeekBackend,
        "ollama": OllamaBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "claude": AnthropicBackend,
    }
    
    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(backends.keys())}")
    
    return backends[name](**kwargs)


def list_topics() -> List[str]:
    """List available topic names."""
    return list(TOPIC_REGISTRY.keys())


def get_topic_spec(name: str) -> TopicSpec:
    """Get a topic specification by name."""
    if name not in TOPIC_REGISTRY:
        raise ValueError(f"Unknown topic: {name}. Available: {list_topics()}")
    return TOPIC_REGISTRY[name]

