"""Evaluation Orchestrator: for each row, looks up its module's config and
fans out one scoring task per configured criterion, via the ScorerScheduler.
"""
from __future__ import annotations

from collections.abc import Callable

from .aggregator import RowResult, ScoreAggregator
from .judge_provider import JudgeProvider
from .module_config import ModuleConfigRegistry
from .models import EvalRow
from .registry import ScorerRegistry
from .scheduler import ScorerScheduler

#: called once per row that permanently failed to score (ScorerScheduler's
#: own retries -- see RETRYABLE_EXCEPTIONS there -- already exhausted):
#: (row, exception).
FailureCallback = Callable[[EvalRow, Exception], None]


class EvaluationOrchestrator:
    def __init__(
        self,
        scorer_registry: ScorerRegistry,
        module_configs: ModuleConfigRegistry,
        judge: JudgeProvider,
        scheduler: ScorerScheduler | None = None,
    ) -> None:
        self._scorers = scorer_registry
        self._modules = module_configs
        self._judge = judge
        self._scheduler = scheduler or ScorerScheduler()
        self._aggregator = ScoreAggregator()

    def run(self, rows: list[EvalRow], on_failure: FailureCallback | None = None) -> list[RowResult]:
        """Scores every row, in order. A row whose scoring raises (e.g. a
        judge call that times out on every retry) is excluded from the
        result and reported via `on_failure(row, exc)` if given, rather than
        aborting every other row in `rows` -- callers that want the old
        fail-fast behavior can pass `on_failure=None` (the default) and
        check `len(results) < len(rows)` instead."""
        results: list[RowResult] = []
        for row in rows:
            config = self._modules.get(row.module)
            tasks = [(criterion, self._scorers.get(criterion)) for criterion in config.criteria]
            try:
                scores = self._scheduler.run(row, tasks, self._judge)
            except Exception as exc:
                if on_failure is None:
                    raise
                on_failure(row, exc)
                continue
            results.append(self._aggregator.aggregate(row, scores, config))
        return results
