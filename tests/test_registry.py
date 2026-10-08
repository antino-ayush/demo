import pytest

from evalsuite.registry import default_registry


def test_default_registry_covers_all_module_criteria():
    registry = default_registry(use_deepeval=False)
    expected = {
        "completeness", "conciseness", "faithfulness", "relevancy",
        "correctness", "coherence", "safety", "format_adherence",
    }
    assert expected <= set(registry.known_criteria())


def test_unknown_criterion_raises():
    registry = default_registry()
    with pytest.raises(KeyError):
        registry.get("not_a_real_criterion")
