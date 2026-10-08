"""Loads ModuleConfig objects from YAML files -- the Module Config Registry
from the design doc's HLD. Adding a module means adding a YAML file here,
not a code change.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .models import ModuleConfig


class ModuleConfigRegistry:
    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._cache: dict[str, ModuleConfig] = {}
        self._endpoint_index: dict[str, str] | None = None

    def get(self, module: str) -> ModuleConfig:
        if module not in self._cache:
            self._cache[module] = self._load(module)
        return self._cache[module]

    def _load(self, module: str) -> ModuleConfig:
        path = self._config_dir / f"{module}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"no module config for {module!r} at {path}; add {path.name} under {self._config_dir}"
            )
        data = yaml.safe_load(path.read_text())
        return ModuleConfig(
            module=data["module"],
            criteria=data["criteria"],
            weights=data["weights"],
            thresholds=data.get("thresholds", {}),
        )

    def known_modules(self) -> list[str]:
        return sorted(p.stem for p in self._config_dir.glob("*.yaml"))

    def module_for_endpoint(self, endpoint: str) -> str:
        """Resolves an API endpoint identifier (a route path, or whatever
        name your services log -- e.g. "/api/v1/tickets/summarize") to the
        module that should score its responses, via each YAML's `endpoints`
        list. This is the endpoint -> module mapping: it lives in the
        configs, not in code, same as everything else about a module.

        Use this in whatever ingests real traffic/logs into a dataset --
        tag each captured row with the endpoint that produced it, then
        resolve `module` once here before writing the EvalRow, instead of
        hand-labeling `module` on every row.
        """
        if self._endpoint_index is None:
            self._build_endpoint_index()
        assert self._endpoint_index is not None
        try:
            return self._endpoint_index[endpoint]
        except KeyError as exc:
            raise KeyError(
                f"no module config claims endpoint {endpoint!r}; "
                f"known endpoints: {sorted(self._endpoint_index)}. "
                f"Add it to the `endpoints:` list in the right configs/modules/*.yaml."
            ) from exc

    def _build_endpoint_index(self) -> None:
        index: dict[str, str] = {}
        for path in sorted(self._config_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text())
            module = data["module"]
            for endpoint in data.get("endpoints", []):
                if endpoint in index and index[endpoint] != module:
                    raise ValueError(
                        f"endpoint {endpoint!r} is claimed by both {index[endpoint]!r} "
                        f"and {module!r} -- fix configs/modules/*.yaml, an endpoint can only "
                        f"belong to one module"
                    )
                index[endpoint] = module
        self._endpoint_index = index
