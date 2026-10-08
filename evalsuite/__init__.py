"""LLM response evaluation framework -- golden-set vs generated-set scoring
with a pluggable LLM judge, per-module criteria, and CI-gateable regression
reporting. See README.md for the quickstart, and the accompanying design
doc ("LLM Evaluation Framework -- Design & Architecture") for the full
HLD/LLD this package implements.
"""
from .models import CriterionScore, EvalRow, ModuleConfig
from .registry import ScorerRegistry, default_registry
from .scorer import Scorer

__all__ = [
    "CriterionScore",
    "EvalRow",
    "ModuleConfig",
    "Scorer",
    "ScorerRegistry",
    "default_registry",
]
