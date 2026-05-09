from types import SimpleNamespace

from halo_forge.training_eligibility import is_training_eligible


def test_training_eligibility_requires_success_even_with_reward():
    result = SimpleNamespace(success=False, reward=0.9, metadata={})

    assert is_training_eligible(result, 0.5) is False


def test_training_eligibility_rejects_compile_only_by_default():
    result = SimpleNamespace(success=True, reward=0.5, metadata={"stage": "compile"})

    assert is_training_eligible(result, 0.5) is False
    assert is_training_eligible(result, 0.5, allow_compile_only=True) is True
