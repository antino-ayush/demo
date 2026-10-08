"""Scorers that wrap DeepEval's validated RAG metrics.

These override `score()` directly and skip the Template Method's
build_prompt/parse_response -- DeepEval owns prompting internally. Per the
design doc's Build vs. Buy section: use these for criteria DeepEval already
validates (faithfulness, contextual recall/precision, relevancy), and
hand-written GEvalScorer subclasses (geval_scorers.py) for anything
module-specific.

Requires: pip install deepeval. The import is deferred to call time so the
rest of the package works without this optional dependency installed.
"""
from __future__ import annotations

from ..judge_provider import JudgeProvider
from ..models import CriterionScore, EvalRow
from ..scorer import Scorer


def _require_deepeval() -> None:
    try:
        import deepeval  # noqa: F401
    except ImportError as exc:
        raise ImportError("This Scorer wraps DeepEval. Install it with: pip install deepeval") from exc


class _DeepEvalScorer(Scorer):
    """Shared plumbing for every DeepEval-backed Scorer."""

    def _build_test_case(self, row: EvalRow):
        from deepeval.test_case import LLMTestCase

        return LLMTestCase(
            input=row.input,
            actual_output=row.generated,
            expected_output=row.golden,
            retrieval_context=[row.context] if row.context else None,
        )

    def _build_metric(self, judge: JudgeProvider):
        raise NotImplementedError

    def score(self, row: EvalRow, judge: JudgeProvider) -> CriterionScore:
        _require_deepeval()
        metric = self._build_metric(judge)
        case = self._build_test_case(row)
        metric.measure(case)
        return CriterionScore(
            criterion=self.criterion_name,
            score=float(metric.score),
            rationale=str(getattr(metric, "reason", "") or ""),
            judge_provider=judge.name,
            judge_model=judge.model_name,
        )


class FaithfulnessScorer(_DeepEvalScorer):
    """Hallucination, in the design doc's language -- wraps FaithfulnessMetric."""

    criterion_name = "faithfulness"

    def _build_metric(self, judge: JudgeProvider):
        from deepeval.metrics import FaithfulnessMetric

        return FaithfulnessMetric(threshold=0.5, model=judge.model_name)


class ContextualRecallScorer(_DeepEvalScorer):
    """Under-generation, in the design doc's language -- wraps ContextualRecallMetric."""

    criterion_name = "completeness"

    def _build_metric(self, judge: JudgeProvider):
        from deepeval.metrics import ContextualRecallMetric

        return ContextualRecallMetric(threshold=0.5, model=judge.model_name)


class ContextualPrecisionScorer(_DeepEvalScorer):
    """Over-generation, in the design doc's language -- wraps ContextualPrecisionMetric."""

    criterion_name = "conciseness"

    def _build_metric(self, judge: JudgeProvider):
        from deepeval.metrics import ContextualPrecisionMetric

        return ContextualPrecisionMetric(threshold=0.5, model=judge.model_name)


class RelevancyScorer(_DeepEvalScorer):
    """Going out of context, in the design doc's language -- wraps AnswerRelevancyMetric."""

    criterion_name = "relevancy"

    def _build_metric(self, judge: JudgeProvider):
        from deepeval.metrics import AnswerRelevancyMetric

        return AnswerRelevancyMetric(threshold=0.5, model=judge.model_name)
