from evalsuite.aggregator import ScoreAggregator
from evalsuite.models import CriterionScore, EvalRow, ModuleConfig


def make_row():
    return EvalRow(row_id="1", module="m", input="q", golden="g", generated="a")


def test_composite_is_weighted_average():
    config = ModuleConfig(module="m", criteria=["a", "b"], weights={"a": 0.25, "b": 0.75})
    scores = [CriterionScore("a", 1.0), CriterionScore("b", 0.0)]
    result = ScoreAggregator().aggregate(make_row(), scores, config)
    assert abs(result.composite - 0.25) < 1e-9


def test_failed_criteria_below_threshold():
    config = ModuleConfig(module="m", criteria=["a"], weights={"a": 1.0}, thresholds={"a": 0.8})
    scores = [CriterionScore("a", 0.5)]
    result = ScoreAggregator().aggregate(make_row(), scores, config)
    assert not result.passed
    assert result.failed_criteria == ["a"]


def test_to_dict_serializes_row_and_scores_for_output():
    config = ModuleConfig(module="m", criteria=["a"], weights={"a": 1.0}, thresholds={"a": 0.8})
    scores = [CriterionScore("a", 0.5, rationale="close but not quite", judge_provider="mock", judge_model="mock-v1")]
    result = ScoreAggregator().aggregate(make_row(), scores, config)

    d = result.to_dict()

    assert d["row_id"] == "1"
    assert d["module"] == "m"
    assert d["golden"] == "g"
    assert d["generated"] == "a"
    assert d["composite"] == 0.5
    assert d["passed"] is False
    assert d["failed_criteria"] == ["a"]
    assert d["scores"] == [
        {
            "criterion": "a",
            "score": 0.5,
            "rationale": "close but not quite",
            "judge_provider": "mock",
            "judge_model": "mock-v1",
        }
    ]
