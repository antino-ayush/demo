import pytest

from evalsuite.models import ModuleConfig


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ModuleConfig(module="x", criteria=["a", "b"], weights={"a": 0.5, "b": 0.6})


def test_weights_must_cover_all_criteria():
    with pytest.raises(ValueError):
        ModuleConfig(module="x", criteria=["a", "b"], weights={"a": 1.0})


def test_valid_config():
    config = ModuleConfig(module="x", criteria=["a", "b"], weights={"a": 0.4, "b": 0.6}, thresholds={"a": 0.5})
    assert config.threshold_for("a") == 0.5
    assert config.threshold_for("b") == 0.5  # falls back to the default
