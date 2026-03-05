#!/usr/bin/env python3
"""Verifier service regression tests for HumanEval/MBPP assertion validity."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ui.services.verifier_service import VerifierService


def test_humaneval_valid_solution_passes_assertion_harness():
    """HumanEval should pass only when bundled assertions pass."""
    service = VerifierService()
    prompt = service.VERIFIERS["HumanEval"]["example_prompt"]
    solution = service.VERIFIERS["HumanEval"]["example_solution"]

    result = asyncio.run(
        service.verify(
            verifier_name="HumanEval",
            prompt=prompt,
            solution=solution,
        )
    )

    assert result.passed is True
    assert result.reward == 1.0


def test_humaneval_invalid_solution_fails_assertion_harness():
    """HumanEval should fail logical-invalid code even when syntax is valid."""
    service = VerifierService()
    prompt = service.VERIFIERS["HumanEval"]["example_prompt"]
    invalid_solution = "    return False"

    result = asyncio.run(
        service.verify(
            verifier_name="HumanEval",
            prompt=prompt,
            solution=invalid_solution,
        )
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert "failed" in result.message.lower()


def test_mbpp_valid_solution_passes_assertion_harness():
    """MBPP should pass when bundled assertions pass."""
    service = VerifierService()
    prompt = service.VERIFIERS["MBPP"]["example_prompt"]
    solution = service.VERIFIERS["MBPP"]["example_solution"]

    result = asyncio.run(
        service.verify(
            verifier_name="MBPP",
            prompt=prompt,
            solution=solution,
        )
    )

    assert result.passed is True
    assert result.reward == 1.0


def test_mbpp_invalid_solution_fails_assertion_harness():
    """MBPP should fail logical-invalid code even when syntax is valid."""
    service = VerifierService()
    prompt = service.VERIFIERS["MBPP"]["example_prompt"]
    invalid_solution = (
        "def similar_elements(test_tup1, test_tup2):\n"
        "    return tuple(test_tup1)\n"
    )

    result = asyncio.run(
        service.verify(
            verifier_name="MBPP",
            prompt=prompt,
            solution=invalid_solution,
        )
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert "failed" in result.message.lower()


def test_verifiers_page_examples_match_backend_harness_expectations():
    """UI examples for HumanEval/MBPP should stay aligned with service test harness."""
    source = Path("ui/pages/verifiers.py").read_text(encoding="utf-8")
    assert "has_close_elements" in source
    assert "similar_elements" in source
