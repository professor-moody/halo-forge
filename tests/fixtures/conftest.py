"""
Shared pytest fixtures for halo-forge tests.

These fixtures are automatically available to all test files.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pytest
from PIL import Image


# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_prompts() -> List[Dict[str, Any]]:
    """Load sample code prompts from fixtures."""
    prompts = []
    prompts_file = FIXTURES_DIR / "sample_prompts.jsonl"
    
    if prompts_file.exists():
        with open(prompts_file) as f:
            for line in f:
                prompts.append(json.loads(line))
    
    return prompts


@pytest.fixture
def vlm_samples() -> List[Dict[str, Any]]:
    """Load VLM samples from fixtures."""
    samples = []
    samples_file = FIXTURES_DIR / "vlm_samples.jsonl"
    
    if samples_file.exists():
        with open(samples_file) as f:
            for line in f:
                samples.append(json.loads(line))
    
    return samples


@pytest.fixture
def mock_responses() -> Dict[str, Any]:
    """Load mock model responses from fixtures."""
    responses_file = FIXTURES_DIR / "mock_responses.json"
    
    if responses_file.exists():
        with open(responses_file) as f:
            return json.load(f)
    
    return {}


@pytest.fixture
def create_test_image():
    """Factory fixture for creating test images."""
    def _create_image(width: int = 224, height: int = 224, color: tuple = (128, 128, 128)) -> Image.Image:
        """Create a test image with specified dimensions and color."""
        return Image.new('RGB', (width, height), color=color)
    
    return _create_image


@pytest.fixture
def test_image_small(create_test_image) -> Image.Image:
    """Create a small test image (64x64)."""
    return create_test_image(64, 64, (255, 0, 0))


@pytest.fixture
def test_image_medium(create_test_image) -> Image.Image:
    """Create a medium test image (224x224)."""
    return create_test_image(224, 224, (0, 255, 0))


@pytest.fixture
def test_image_large(create_test_image) -> Image.Image:
    """Create a large test image (640x480)."""
    return create_test_image(640, 480, (0, 0, 255))


@pytest.fixture
def valid_cpp_code() -> str:
    """Return valid C++ code that compiles."""
    return '''
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
'''


@pytest.fixture
def invalid_cpp_code() -> str:
    """Return invalid C++ code with syntax error."""
    return '''
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl
    return 0;
}
'''


@pytest.fixture
def valid_python_code() -> str:
    """Return valid Python code with tests."""
    return '''
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
'''


@pytest.fixture
def vlm_completion_with_reasoning() -> str:
    """Return a VLM completion with proper reasoning."""
    return """
Looking at the image, I can observe several things.

Step 1: I see a person standing in the center of the image.
Step 2: The person is holding a red umbrella.
Step 3: It appears to be raining based on the wet ground.

Therefore, the answer is: a person with a red umbrella in the rain.
"""


@pytest.fixture
def vlm_completion_short() -> str:
    """Return a short VLM completion without reasoning."""
    return "The answer is yes."
