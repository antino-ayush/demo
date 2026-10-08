"""Score Aggregator: applies a module's weights to per-criterion scores to
get one composite score per row, and rolls rows up into a per-module,
per-run summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import CriterionScore, EvalRow, ModuleConfig


@dataclass
class RowResult:
    row: EvalRow
    scores: list[CriterionScore]
    composite: float
    passed: bool
    failed_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "row_id": self.row.row_id,
            "module": self.row.module,
            "input": self.row.input,
            "golden": self.row.golden,
            "generated": self.row.generated,
            "composite": round(self.composite, 4),
            "passed": self.passed,
            "failed_criteria": self.failed_criteria,
            "scores": [
                {
                    "criterion": s.criterion,
                    "score": round(s.score, 4),
                    "rationale": s.rationale,
                    "judge_provider": s.judge_provider,
                    "judge_model": s.judge_model,
                }
                for s in self.scores
            ],
        }


class ScoreAggregator:
    def aggregate(self, row: EvalRow, scores: list[CriterionScore], config: ModuleConfig) -> RowResult:
        composite = sum(s.score * config.weights[s.criterion] for s in scores)
        failed = [s.criterion for s in scores if s.score < config.threshold_for(s.criterion)]
        return RowResult(row=row, scores=scores, composite=composite, passed=not failed, failed_criteria=failed)

    def summarize(self, results: list[RowResult]) -> dict:
        if not results:
            return {"rows": 0}
        by_criterion: dict[str, list[float]] = {}
        for r in results:
            for s in r.scores:
                by_criterion.setdefault(s.criterion, []).append(s.score)
        return {
            "rows": len(results),
            "pass_rate": sum(r.passed for r in results) / len(results),
            "avg_composite": sum(r.composite for r in results) / len(results),
            "avg_by_criterion": {c: sum(v) / len(v) for c, v in by_criterion.items()},
        }
