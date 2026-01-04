"""
Benchmark prompts for various languages.

Provides standard prompts for evaluating code generation quality across
different programming languages.
"""

from typing import List, Dict

# C++ Prompts (existing from runner.py)
CPP_PROMPTS = [
    {
        "id": "cpp_hello_world",
        "prompt": "Write a C++ program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!",
        "difficulty": "easy",
        "language": "cpp",
    },
    {
        "id": "cpp_sum_function",
        "prompt": "Write a C++ function that returns the sum of two integers a and b, then call it in main to print the result of 5 + 3.",
        "expected_output": "8",
        "difficulty": "easy",
        "language": "cpp",
    },
    {
        "id": "cpp_factorial",
        "prompt": "Write a C++ program that computes and prints the factorial of 6.",
        "expected_output": "720",
        "difficulty": "medium",
        "language": "cpp",
    },
    {
        "id": "cpp_fibonacci",
        "prompt": "Write a C++ program that prints the first 10 Fibonacci numbers, space-separated.",
        "expected_output": "0 1 1 2 3 5 8 13 21 34",
        "difficulty": "medium",
        "language": "cpp",
    },
    {
        "id": "cpp_binary_search",
        "prompt": "Write a C++ program that uses binary search to find the index of 7 in sorted array {1, 3, 5, 7, 9, 11} and prints it.",
        "expected_output": "3",
        "difficulty": "hard",
        "language": "cpp",
    },
]

# Rust Prompts
RUST_PROMPTS = [
    {
        "id": "rust_hello_world",
        "prompt": "Write a Rust program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!",
        "difficulty": "easy",
        "language": "rust",
    },
    {
        "id": "rust_sum_function",
        "prompt": "Write a Rust program with a function that returns the sum of two i32 values, then call it in main to print the result of 5 + 3.",
        "expected_output": "8",
        "difficulty": "easy",
        "language": "rust",
    },
    {
        "id": "rust_factorial",
        "prompt": "Write a Rust program that computes and prints the factorial of 6.",
        "expected_output": "720",
        "difficulty": "medium",
        "language": "rust",
    },
    {
        "id": "rust_fibonacci",
        "prompt": "Write a Rust program that prints the first 10 Fibonacci numbers, space-separated.",
        "expected_output": "0 1 1 2 3 5 8 13 21 34",
        "difficulty": "medium",
        "language": "rust",
    },
    {
        "id": "rust_vector_sum",
        "prompt": "Write a Rust program that calculates and prints the sum of a Vec containing [1, 2, 3, 4, 5].",
        "expected_output": "15",
        "difficulty": "easy",
        "language": "rust",
    },
    {
        "id": "rust_option_handling",
        "prompt": "Write a Rust program that safely gets the first element of a Vec<i32> containing [10, 20, 30] using Option and prints it.",
        "expected_output": "10",
        "difficulty": "medium",
        "language": "rust",
    },
    {
        "id": "rust_string_reverse",
        "prompt": "Write a Rust program that reverses the string 'hello' and prints it.",
        "expected_output": "olleh",
        "difficulty": "medium",
        "language": "rust",
    },
    {
        "id": "rust_prime_check",
        "prompt": "Write a Rust program that checks if 17 is prime and prints 'yes' or 'no'.",
        "expected_output": "yes",
        "difficulty": "medium",
        "language": "rust",
    },
    {
        "id": "rust_binary_search",
        "prompt": "Write a Rust program that uses binary search to find the index of 7 in slice [1, 3, 5, 7, 9, 11] and prints it.",
        "expected_output": "3",
        "difficulty": "hard",
        "language": "rust",
    },
    {
        "id": "rust_hashmap",
        "prompt": "Write a Rust program that creates a HashMap with keys 'a', 'b', 'c' mapping to values 1, 2, 3, then prints the value for 'b'.",
        "expected_output": "2",
        "difficulty": "medium",
        "language": "rust",
    },
]

# Go Prompts
GO_PROMPTS = [
    {
        "id": "go_hello_world",
        "prompt": "Write a Go program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!",
        "difficulty": "easy",
        "language": "go",
    },
    {
        "id": "go_sum_function",
        "prompt": "Write a Go program with a function that returns the sum of two ints, then call it in main to print the result of 5 + 3.",
        "expected_output": "8",
        "difficulty": "easy",
        "language": "go",
    },
    {
        "id": "go_factorial",
        "prompt": "Write a Go program that computes and prints the factorial of 6.",
        "expected_output": "720",
        "difficulty": "medium",
        "language": "go",
    },
    {
        "id": "go_fibonacci",
        "prompt": "Write a Go program that prints the first 10 Fibonacci numbers, space-separated.",
        "expected_output": "0 1 1 2 3 5 8 13 21 34",
        "difficulty": "medium",
        "language": "go",
    },
    {
        "id": "go_slice_sum",
        "prompt": "Write a Go program that calculates and prints the sum of slice []int{1, 2, 3, 4, 5}.",
        "expected_output": "15",
        "difficulty": "easy",
        "language": "go",
    },
    {
        "id": "go_string_reverse",
        "prompt": "Write a Go program that reverses the string 'hello' and prints it.",
        "expected_output": "olleh",
        "difficulty": "medium",
        "language": "go",
    },
    {
        "id": "go_prime_check",
        "prompt": "Write a Go program that checks if 17 is prime and prints 'yes' or 'no'.",
        "expected_output": "yes",
        "difficulty": "medium",
        "language": "go",
    },
    {
        "id": "go_binary_search",
        "prompt": "Write a Go program that uses binary search to find the index of 7 in slice []int{1, 3, 5, 7, 9, 11} and prints it.",
        "expected_output": "3",
        "difficulty": "hard",
        "language": "go",
    },
    {
        "id": "go_map_usage",
        "prompt": "Write a Go program that creates a map with keys 'a', 'b', 'c' mapping to values 1, 2, 3, then prints the value for 'b'.",
        "expected_output": "2",
        "difficulty": "medium",
        "language": "go",
    },
    {
        "id": "go_goroutine",
        "prompt": "Write a Go program that uses a goroutine to print 'done' after sleeping for 100ms. Use a channel to wait for it before exiting.",
        "expected_output": "done",
        "difficulty": "hard",
        "language": "go",
    },
]


def get_prompts_for_language(language: str) -> List[Dict]:
    """Get benchmark prompts for a specific language."""
    prompts = {
        "cpp": CPP_PROMPTS,
        "c++": CPP_PROMPTS,
        "rust": RUST_PROMPTS,
        "go": GO_PROMPTS,
        "golang": GO_PROMPTS,
    }
    return prompts.get(language.lower(), [])


def get_all_prompts() -> Dict[str, List[Dict]]:
    """Get all benchmark prompts organized by language."""
    return {
        "cpp": CPP_PROMPTS,
        "rust": RUST_PROMPTS,
        "go": GO_PROMPTS,
    }


# Convenience exports
ALL_LANGUAGES = ["cpp", "rust", "go"]

