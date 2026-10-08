"""Core data models shared across the evaluation framework.

These mirror the LLD in the design doc: EvalRow is one (golden, generated)
pair to be scored; CriterionScore is one Scorer's verdict on one row;
ModuleConfig says which criteria apply to a module, how they're weighted,
and what score counts as a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalRow:
    """One row to evaluate: a single (input, golden, generated) triple."""

    row_id: str
    module: str
    input: str
    golden: str
    generated: str
    context: str | None = None  # retrieved context; populated for RAG-style modules

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "module": self.module,
            "input": self.input,
            "golden": self.golden,
            "generated": self.generated,
            "context": self.context,
        }


@dataclass
class CriterionScore:
    """One Scorer's verdict on one EvalRow."""

    criterion: str
    score: float  # normalized 0-1, higher is better
    rationale: str = ""
    judge_provider: str | None = None
    judge_model: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score for {self.criterion!r} must be in [0, 1], got {self.score}")


@dataclass
class ModuleConfig:
    """Which criteria apply to a module, their weights, and pass thresholds.

    This is the Module Config Registry's unit of data from the design doc's
    HLD: adding a module means adding one of these (usually via YAML), not
    writing new code.
    """

    module: str
    criteria: list[str]
    weights: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(self.criteria) - set(self.weights)
        if missing:
            raise ValueError(f"module {self.module!r} is missing weights for: {sorted(missing)}")
        extra = set(self.weights) - set(self.criteria)
        if extra:
            raise ValueError(f"module {self.module!r} has weights for unused criteria: {sorted(extra)}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"module {self.module!r} weights must sum to 1.0, got {total}")

    def threshold_for(self, criterion: str, default: float = 0.5) -> float:
        return self.thresholds.get(criterion, default)
