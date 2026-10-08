"""ScorerRegistry: binds a criterion name to a concrete Scorer instance.

This is the literal switch behind the design doc's Build vs. Buy hybrid --
moving a criterion between "wrapped DeepEval" and "hand-written GEval" is
one registration line here, nowhere else in the system.
"""
from __future__ import annotations

from .scorer import Scorer


class ScorerRegistry:
    def __init__(self) -> None:
        self._scorers: dict[str, Scorer] = {}

    def register(self, criterion_name: str, scorer: Scorer) -> "ScorerRegistry":
        self._scorers[criterion_name] = scorer
        return self

    def get(self, criterion_name: str) -> Scorer:
        try:
            return self._scorers[criterion_name]
        except KeyError as exc:
            raise KeyError(
                f"no Scorer registered for criterion {criterion_name!r}; "
                f"registered: {sorted(self._scorers)}"
            ) from exc

    def known_criteria(self) -> list[str]:
        return sorted(self._scorers)


def default_registry(use_deepeval: bool = False) -> ScorerRegistry:
    """Builds the registry used by the CLI/tests.

    use_deepeval=False (the default, so the repo runs with zero optional
    dependencies): faithfulness/completeness/conciseness/relevancy are
    hand-written GEval-style scorers.

    use_deepeval=True: those same four criteria are swapped for DeepEval-
    backed Scorers instead -- same criterion names, same Scorer interface,
    different implementation, exactly per the design doc's hybrid
    recommendation. Requires `pip install deepeval`.
    """
    from .scorers.geval_scorers import (
        CoherenceScorer,
        CorrectnessScorer,
        FormatAdherenceScorer,
        GEvalScorer,
        SafetyScorer,
    )

    registry = ScorerRegistry()
    registry.register("coherence", CoherenceScorer())
    registry.register("correctness", CorrectnessScorer())
    registry.register("safety", SafetyScorer())
    registry.register(
        "format_adherence",
        FormatAdherenceScorer(format_description="valid JSON matching the module's schema"),
    )

    if use_deepeval:
        from .scorers.deepeval_scorers import (
            ContextualPrecisionScorer,
            ContextualRecallScorer,
            FaithfulnessScorer,
            RelevancyScorer,
        )

        registry.register("faithfulness", FaithfulnessScorer())
        registry.register("completeness", ContextualRecallScorer())
        registry.register("conciseness", ContextualPrecisionScorer())
        registry.register("relevancy", RelevancyScorer())
        return registry

    class FaithfulnessScorer(GEvalScorer):  # hand-written fallback (hallucination)
        criterion_name = "faithfulness"
        definition = (
            "Every claim in the generated answer must be traceable to the golden answer "
            "or context; unsupported claims are hallucinations."
        )
        evaluation_steps = [
            "List the factual claims made in the generated answer.",
            "Check each claim against the golden answer (and context, if given).",
            "Flag any claim not supported by the golden answer/context as a hallucination.",
            "Score = fraction of claims that are supported.",
        ]
        needs_context = True

    class CompletenessScorer(GEvalScorer):  # hand-written fallback (under-generation)
        criterion_name = "completeness"
        definition = "All key facts present in the golden answer should also be present in the generated answer."
        evaluation_steps = [
            "List the key facts in the golden answer.",
            "Check whether each is present in the generated answer.",
            "Score = fraction of golden facts that are present.",
        ]

    class ConcisenessScorer(GEvalScorer):  # hand-written fallback (over-generation)
        criterion_name = "conciseness"
        definition = "The generated answer should not add unsupported or unnecessary content beyond the golden answer."
        evaluation_steps = [
            "List content in the generated answer not present in the golden answer.",
            "Judge whether that extra content is unsupported padding, vs. reasonable elaboration.",
            "Score lower as more unsupported/unnecessary content is present.",
        ]

    class RelevancyScorer(GEvalScorer):  # hand-written fallback (out of context)
        criterion_name = "relevancy"
        definition = "The response should stay on-topic for the question and not include unrelated content."
        evaluation_steps = [
            "Re-read the input/question.",
            "Check whether the generated answer directly addresses it.",
            "Penalize tangents or content unrelated to the question.",
        ]

    registry.register("faithfulness", FaithfulnessScorer())
    registry.register("completeness", CompletenessScorer())
    registry.register("conciseness", ConcisenessScorer())
    registry.register("relevancy", RelevancyScorer())
    return registry
