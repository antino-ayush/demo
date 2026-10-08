from pathlib import Path

import pytest

from evalsuite.module_config import ModuleConfigRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_known_endpoints_resolve_to_the_right_module(tmp_path):
    # Synthetic configs, not the project's real configs/modules/ -- this test
    # is about the endpoint -> module resolution mechanism itself, not about
    # which modules happen to be configured for real right now.
    (tmp_path / "alpha.yaml").write_text(
        "module: alpha\nendpoints: [/api/v1/alpha]\ncriteria: [c]\nweights: {c: 1.0}\n"
    )
    (tmp_path / "beta.yaml").write_text(
        "module: beta\nendpoints: [/api/v1/beta, /api/v1/beta/legacy]\ncriteria: [c]\nweights: {c: 1.0}\n"
    )
    registry = ModuleConfigRegistry(tmp_path)
    assert registry.module_for_endpoint("/api/v1/alpha") == "alpha"
    assert registry.module_for_endpoint("/api/v1/beta") == "beta"
    assert registry.module_for_endpoint("/api/v1/beta/legacy") == "beta"


def test_unknown_endpoint_raises_with_known_list():
    registry = ModuleConfigRegistry(ROOT / "configs" / "modules")
    with pytest.raises(KeyError, match="no module config claims endpoint"):
        registry.module_for_endpoint("/api/v1/not-a-real-route")


def test_same_endpoint_in_two_configs_is_a_config_error(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "module: a\nendpoints: [/x]\ncriteria: [c]\nweights: {c: 1.0}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "module: b\nendpoints: [/x]\ncriteria: [c]\nweights: {c: 1.0}\n"
    )
    registry = ModuleConfigRegistry(tmp_path)
    with pytest.raises(ValueError, match="claimed by both"):
        registry.module_for_endpoint("/x")
