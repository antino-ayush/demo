from pathlib import Path

from evalsuite.dataset_loader import load_dataset
from evalsuite.judge_provider import MockJudgeProvider
from evalsuite.module_config import ModuleConfigRegistry
from evalsuite.orchestrator import EvaluationOrchestrator
from evalsuite.registry import default_registry

ROOT = Path(__file__).resolve().parents[1]


def _write_sample_module_configs(config_dir: Path) -> Path:
    # Synthetic configs for exactly the modules data/sample_dataset.jsonl's
    # rows use (summarization/rag_qa/structured_extraction/chat) -- not the
    # project's real configs/modules/, since this test is about the
    # orchestrator/scoring pipeline itself, not about which modules happen to
    # be configured for real right now.
    (config_dir / "summarization.yaml").write_text(
        "module: summarization\ncriteria: [completeness, faithfulness]\n"
        "weights: {completeness: 0.5, faithfulness: 0.5}\n"
    )
    (config_dir / "rag_qa.yaml").write_text(
        "module: rag_qa\ncriteria: [faithfulness, completeness]\n"
        "weights: {faithfulness: 0.5, completeness: 0.5}\n"
    )
    (config_dir / "structured_extraction.yaml").write_text(
        "module: structured_extraction\ncriteria: [format_adherence, correctness]\n"
        "weights: {format_adherence: 0.5, correctness: 0.5}\n"
    )
    (config_dir / "chat.yaml").write_text(
        "module: chat\ncriteria: [coherence, relevancy]\n"
        "weights: {coherence: 0.5, relevancy: 0.5}\n"
    )
    return config_dir


def _build_orchestrator(tmp_path: Path):
    registry = default_registry(use_deepeval=False)
    module_configs = ModuleConfigRegistry(_write_sample_module_configs(tmp_path))
    return EvaluationOrchestrator(registry, module_configs, MockJudgeProvider()), module_configs


def test_full_pipeline_runs_offline(tmp_path):
    rows = load_dataset(ROOT / "data" / "sample_dataset.jsonl")
    orchestrator, module_configs = _build_orchestrator(tmp_path)

    results = orchestrator.run(rows)

    assert len(results) == len(rows)
    for r in results:
        assert 0.0 <= r.composite <= 1.0
        assert {s.criterion for s in r.scores} == set(module_configs.get(r.row.module).criteria)


def test_hallucination_row_scores_lower_on_faithfulness(tmp_path):
    """qa-2 in the sample dataset invents claims not in the context/golden
    answer -- the mock judge (token overlap) should reflect that as a lower
    faithfulness score than a faithful row like qa-1."""
    rows = {r.row_id: r for r in load_dataset(ROOT / "data" / "sample_dataset.jsonl")}
    orchestrator, _ = _build_orchestrator(tmp_path)

    results = {r.row.row_id: r for r in orchestrator.run([rows["qa-1"], rows["qa-2"]])}

    def faithfulness(row_id: str) -> float:
        return next(s.score for s in results[row_id].scores if s.criterion == "faithfulness")

    assert faithfulness("qa-1") > faithfulness("qa-2")


def test_format_violation_scores_lower_on_format_adherence(tmp_path):
    """ext-2 returns prose instead of the required JSON -- format_adherence
    should score it lower than ext-1, which matches the golden JSON exactly."""
    rows = {r.row_id: r for r in load_dataset(ROOT / "data" / "sample_dataset.jsonl")}
    orchestrator, _ = _build_orchestrator(tmp_path)

    results = {r.row.row_id: r for r in orchestrator.run([rows["ext-1"], rows["ext-2"]])}

    def format_score(row_id: str) -> float:
        return next(s.score for s in results[row_id].scores if s.criterion == "format_adherence")

    assert format_score("ext-1") > format_score("ext-2")
