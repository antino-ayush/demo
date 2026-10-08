"""Scorer: one criterion, one interface -- the Strategy pattern from the
design doc's LLD.

Default flow is Template Method (build_prompt -> judge -> parse_response).
A Scorer that wraps an existing library (DeepEval/RAGAS) can override
`score()` directly instead and skip build_prompt/parse_response entirely --
both are valid Scorers as far as the Orchestrator and Aggregator care,
since they only ever call `.score()`.
"""
from __future__ import annotations

from abc import ABC

from .judge_provider import JudgeProvider
from .models import CriterionScore, EvalRow

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "0-10, higher is better"},
        "rationale": {"type": "string", "description": "brief chain-of-thought explanation"},
    },
    "required": ["score", "rationale"],
}


class Scorer(ABC):
    criterion_name: str = "unset"

    def score(self, row: EvalRow, judge: JudgeProvider) -> CriterionScore:
        prompt = self.build_prompt(row)                      # Template Method step 1
        raw = judge.judge(prompt, response_schema=self.schema())
        return self.parse_response(raw, judge)                # Template Method step 2

    def schema(self) -> dict:
        return RESPONSE_SCHEMA

    def build_prompt(self, row: EvalRow) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_prompt() or override score()"
        )

    def parse_response(self, raw: dict, judge: JudgeProvider) -> CriterionScore:
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_response() or override score()"
        )
