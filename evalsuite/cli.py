"""Command-line entry point: run a batch evaluation over a dataset.

Example:
    python -m evalsuite.cli run \\
        --dataset data/sample_dataset.jsonl \\
        --config-dir configs/modules \\
        --judge mock \\
        --db eval_results.db \\
        --gate
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .dataset_loader import load_dataset
from .env import load_env
from .generator import DEFAULT_OUTPUT_DIR
from .judge_provider import build_judge_provider
from .module_config import ModuleConfigRegistry
from .orchestrator import EvaluationOrchestrator
from .registry import default_registry
from .reporting import diff_against_baseline
from .results_store import ResultsStore


def main(argv: list[str] | None = None) -> int:
    load_env()  # .env -> os.environ, before anything below reads endpoints/tokens

    parser = argparse.ArgumentParser(prog="evalsuite")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="score a dataset and store the results")
    run_p.add_argument("--dataset", required=True)
    run_p.add_argument("--config-dir", default="configs/modules")
    run_p.add_argument("--judge", default="mock", choices=["mock", "claude", "openai", "nemotron"])
    run_p.add_argument("--judge-model", default=None)
    run_p.add_argument("--db", default="eval_results.db")
    run_p.add_argument(
        "--use-deepeval",
        action="store_true",
        help="wrap DeepEval for faithfulness/completeness/conciseness/relevancy (pip install deepeval)",
    )
    run_p.add_argument("--gate", action="store_true", help="exit non-zero on regression or failing rows")
    run_p.add_argument("--regression-threshold", type=float, default=0.02)
    run_p.add_argument(
        "--scored-out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory to write each module's per-row scored JSONL to (default: {DEFAULT_OUTPUT_DIR}/<module>_scored.jsonl)",
    )

    gen_p = sub.add_parser(
        "generate",
        help="call each row's module service endpoint (see .env) to fill in `generated` for a golden dataset",
    )
    gen_p.add_argument("--dataset", required=True, help="golden dataset to fill in (input/golden present, generated empty)")
    gen_p.add_argument(
        "--out",
        default=None,
        help="where to write the filled-in dataset (default: output/<same filename>, so data/ golden sets stay input+golden-only)",
    )
    gen_p.add_argument("--config-dir", default="configs/modules", help="where to read each module's optional service: request/response contract from")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
    if args.command == "generate":
        return _generate(args)
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover


def _generate(args: argparse.Namespace) -> int:
    from .generator import fill_dataset

    out_path = args.out or str(Path(DEFAULT_OUTPUT_DIR) / Path(args.dataset).name)
    rows = fill_dataset(args.dataset, out_path, args.config_dir)
    print(f"generated {len(rows)} rows -> {out_path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    judge_kwargs = {"model": args.judge_model} if args.judge_model else {}
    judge = build_judge_provider(args.judge, **judge_kwargs)

    registry = default_registry(use_deepeval=args.use_deepeval)
    module_configs = ModuleConfigRegistry(args.config_dir)
    orchestrator = EvaluationOrchestrator(registry, module_configs, judge)
    store = ResultsStore(args.db)

    rows = load_dataset(args.dataset)
    rows_by_module: dict[str, list] = defaultdict(list)
    for row in rows:
        rows_by_module[row.module].append(row)

    overall_ok = True
    for module, module_rows in sorted(rows_by_module.items()):
        baseline_run_id = store.latest_run(module)
        run_id = store.start_run(module, baseline_run_id=baseline_run_id)

        results = orchestrator.run(module_rows)
        store.save_results(run_id, results)

        scored_path = Path(args.scored_out_dir) / f"{module}_scored.jsonl"
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        with scored_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")

        current_scores = store.avg_scores_for_run(run_id)
        baseline_scores = store.avg_scores_for_run(baseline_run_id)
        report = diff_against_baseline(
            current_scores, baseline_scores, run_id, baseline_run_id, args.regression_threshold
        )

        print(f"\n=== {module} ===")
        print(f"  scored rows -> {scored_path}")
        print(report.render())
        failing_rows = [r for r in results if not r.passed]
        if failing_rows:
            print(f"  {len(failing_rows)}/{len(results)} rows failed a per-criterion threshold")
            for r in failing_rows:
                print(f"    row {r.row.row_id}: failed {r.failed_criteria}")

        if args.gate and (report.has_regression or failing_rows):
            overall_ok = False

    store.close()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
