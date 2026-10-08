"""One-off generator for scripts/run_<module>.py -- not part of the runtime
pipeline, just used to stamp out the per-module runner files consistently.
Re-run this after editing TEMPLATE below to regenerate all of them.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# module -> (human-readable endpoints for the docstring, default judge)
MODULES = {
    "open_gpt": "/api/offline_chatbot/local_llm",
    "gpts": "/api/chat/chatbot/query",
}

TEMPLATE = '''#!/usr/bin/env python3
"""Single-command test runner for the '{module}' module.

Does both steps end-to-end, in one command. Generation still happens in
fixed-size BATCHES (default 50 rows, up to --generate-workers HTTP calls at
once against {endpoints}, per configs/modules/{module}.yaml), but each row
is handed to the judge the INSTANT it finishes generating -- not batched
with the rest of its chunk (pass --no-pipeline for the simpler, fully
batch-wise/sequential mode instead). See evalsuite/batch_runner.py
(`run_streaming` / `run_in_batches`) for the pipeline itself. Row-wise
hand-off matters when the judge is the slow/unreliable side: with
batch-level hand-off, one stuck row anywhere in a batch delays every other
row in that batch from even starting; here a stuck row only blocks itself.

There is ONE output file, --out (default output/{module}_dataset.jsonl),
row_id-keyed rather than an append-only log: each row's JSON entry is
UPDATED IN PLACE (the whole file is atomically rewritten from the current
in-memory state on every row) rather than appended fresh every time, so a
row is never duplicated across resumes and a row's status/answer/score all
live in the one record for that row_id, not scattered across files. Each
row carries a "status": "generated" (has an answer, not yet judged),
"judged" (has an answer AND a score -- "composite", "passed",
"failed_criteria", "scores" all present), "failed_generate", or
"failed_judge" (both carry an "error" field) -- plus "batch": N, the chunk
it was generated in, for provenance.

Per row, the flow is: 1) generate it -- 2) UPDATE its entry in --out to
status "generated", immediately, before judging even starts -- 3) THEN
judge it against this module's criteria (configs/modules/{module}.yaml:
criteria/weights/thresholds) -- 4) update its entry again to status
"judged" (or a failed_* status, with "error", if either step didn't work).
Step 2 happening before step 3 means a row's generated answer is safely on
disk even if judging it hangs, times out, or the run is killed before
reaching it -- you never lose already-successful generation work to a
judge-side problem. This -- and everything else in this paragraph -- also
already holds for --resume's recovered rows and the --end-retries pass
below, not just a row's first attempt: they all update the exact same
--out file through the same in-place mechanism.

On --resume, rows already "judged" in --out are skipped entirely; rows
already "generated" (has an answer, no score yet) are pushed onto the SAME
live judge queue as freshly-generated rows -- concurrently, not as a
separate blocking phase first -- so the judge starts working through a
resumed backlog immediately while fresh generation proceeds at the same
time, instead of the backlog gating when fresh progress can even begin.
See evalsuite/batch_runner.py's `run_streaming` (`preloaded_rows` argument)
for the mechanics. Rows with a failed_* status (or never attempted at all)
go through the full pipeline again.

Every row's status transitions (generated / judged / failed, with a
timestamp) are also logged to output/{module}_run.log -- tail that file to
see which row is currently being processed and its outcome, including the
judge's own retry/timeout detail (evalsuite.scheduler's warnings land
there too).

A row that fails -- either to generate (retried automatically on 429/5xx
first, see ResponseGenerator.RETRYABLE_STATUS_CODES in evalsuite/generator.py)
or to judge (retried on timeout/connection errors, see RETRYABLE_EXCEPTIONS
in evalsuite/scheduler.py; --judge-timeout / --judge-max-retries tune how
hard it tries before giving up) -- gets a failed_* status in --out rather
than crashing the whole run. Once every row has had its first attempt,
--end-retries (default 1) does ONE MORE pass over just the rows that failed
with a transient-looking error (timeout, dropped connection, HTTP 429/5xx
-- see evalsuite/errors.py:is_retryable_error) -- a row that failed with a
permanent-looking error (e.g. 422/401) is never retried, since it would
fail identically every time. Whatever's still failing after that is listed
in output/{module}_generation_failures.jsonl / output/{module}_judge_failures.jsonl
(a filtered summary view -- the authoritative record is still each row's
own entry in --out) and makes the run exit non-zero under --gate.

Never restart WITHOUT --resume against an existing --out, or its current
state will be overwritten with an empty one -- and only ever run ONE
instance of this script against the same --out at a time; two instances
both writing to it (or both hammering the same service/judge host at once)
will corrupt the output and starve each other's requests. Check `ps` if a
run behaves like it's stuck.

Usage:
    python scripts/run_{module}.py
    python scripts/run_{module}.py --batch-size 25 --generate-workers 10
    python scripts/run_{module}.py --resume            # continue a killed/crashed run
    python scripts/run_{module}.py --no-pipeline       # fully batch-wise/sequential instead
    python scripts/run_{module}.py --judge-timeout 20 --judge-max-retries 1  # fail fast on a flaky judge
    python scripts/run_{module}.py --judge mock       # offline, no .env / network needed
    python scripts/run_{module}.py --skip-generate     # judge an already-generated file
    python scripts/run_{module}.py --gate              # exit non-zero on regression/failures

Requires configs/modules/{module}.yaml's service endpoint(s) to be reachable
(SERVICE_BASE_URL + DEFAULT_SERVICE_TOKEN in .env, or a service.url/token
override in that YAML) -- unless --skip-generate is passed. The judge step
needs the .env NEMOTRON_* block filled in unless --judge mock is passed.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/run_{module}.py` works from any cwd

from evalsuite.batch_runner import run_in_batches, run_streaming  # noqa: E402
from evalsuite.dataset_loader import load_dataset  # noqa: E402
from evalsuite.env import load_env  # noqa: E402
from evalsuite.errors import is_retryable_error  # noqa: E402
from evalsuite.generator import DEFAULT_OUTPUT_DIR  # noqa: E402
from evalsuite.judge_provider import build_judge_provider  # noqa: E402
from evalsuite.models import EvalRow  # noqa: E402
from evalsuite.module_config import ModuleConfigRegistry  # noqa: E402
from evalsuite.orchestrator import EvaluationOrchestrator  # noqa: E402
from evalsuite.registry import default_registry  # noqa: E402
from evalsuite.reporting import diff_against_baseline  # noqa: E402
from evalsuite.request_schemas import GPT_VARIANTS  # noqa: E402
from evalsuite.results_store import ResultsStore  # noqa: E402
from evalsuite.scheduler import ScorerScheduler  # noqa: E402

MODULE = "{module}"
MODULE_VARIANTS = GPT_VARIANTS.get(MODULE, {{}})


def main(argv: list[str] | None = None) -> int:
    load_env()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "data" / f"{{MODULE}}_dataset.jsonl"),
        help="golden dataset (input/golden present, generated may be empty); "
        "rows for other modules in the same file are ignored",
    )
    parser.add_argument("--config-dir", default=str(REPO_ROOT / "configs" / "modules"))
    parser.add_argument(
        "--gpt-variant",
        choices=sorted(MODULE_VARIANTS) or None,
        default=next(iter(MODULE_VARIANTS), None),
        help="which dashboard variant of this module's shared service contract to target this run "
        "against -- loads a run-time patched copy of this module's yaml with that variant's request "
        "field overrides applied (see evalsuite.request_schemas.GPT_VARIANTS); has no effect if this "
        "module has no registered variants",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="rows generated concurrently per chunk (default: 50); each row is judged the instant it's "
        "generated, not batched with the rest of its chunk -- see --no-pipeline to change that",
    )
    parser.add_argument(
        "--generate-workers",
        type=int,
        default=8,
        help="max concurrent HTTP calls to the model-under-test service within one batch (default: 8)",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="disable row-wise hand-off: fully generate a batch, then judge that whole batch, before moving "
        "on -- simpler and more predictable, but one stuck row delays every other row in its batch",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=20,
        help="max already-generated rows allowed to queue up waiting for the judge (default: 20; ignored "
        "with --no-pipeline) -- raise it if generation runs faster than judging and the generation side "
        "would otherwise stall waiting for queue space",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"the single row_id-keyed output file -- generated answers, judged scores, and status all "
        "live in one entry per row, updated in place (default: {{DEFAULT_OUTPUT_DIR}}/{{MODULE}}_dataset.jsonl)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="skip calling the service; judge --dataset as-is (its `generated` field must already be filled in)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="pick up a previous (e.g. killed/crashed) run of this same --out: rows already 'judged' are "
        "skipped, rows already 'generated' are judged without regenerating, rows that previously failed "
        "are retried from scratch",
    )
    parser.add_argument("--judge", default="nemotron", choices=["mock", "claude", "openai", "nemotron"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="seconds before a judge call times out (default: provider's own default, e.g. 60 for nemotron); "
        "lower this to fail faster on a hung/overloaded judge backend, at the cost of false negatives on "
        "calls that were just slow, not stuck",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=None,
        help="retries per criterion on a judge timeout/connection error before skipping that row "
        "(default: 2, i.e. 3 attempts total; see ScorerScheduler). Lower to 0-1 to skip stuck rows "
        "faster instead of waiting out multiple full timeouts on each one",
    )
    parser.add_argument(
        "--end-retries",
        type=int,
        default=1,
        help="after the main run finishes, retry rows whose generation/judge failure looks transient "
        "(timeout, dropped connection, HTTP 429/5xx -- see evalsuite.errors.is_retryable_error) this "
        "many times (default: 1); rows that failed with a permanent-looking error (e.g. 422/401) are "
        "never retried, since they'd fail identically every time. 0 disables this pass.",
    )
    parser.add_argument(
        "--use-deepeval",
        action="store_true",
        help="wrap DeepEval for faithfulness/completeness/conciseness/relevancy (pip install deepeval)",
    )
    parser.add_argument("--db", default=str(REPO_ROOT / "eval_results.db"))
    parser.add_argument(
        "--log",
        default=None,
        help=f"where to write the per-row status log (default: {{DEFAULT_OUTPUT_DIR}}/{{MODULE}}_run.log)",
    )
    parser.add_argument("--gate", action="store_true", help="exit non-zero on regression vs. the previous run, or on any failing row")
    parser.add_argument("--regression-threshold", type=float, default=0.02)
    args = parser.parse_args(argv)

    variant_overrides = MODULE_VARIANTS.get(args.gpt_variant, {{}})
    out_variant_suffix = "" if args.gpt_variant in (None, next(iter(MODULE_VARIANTS), None)) else f"_{{args.gpt_variant}}"

    # Only the request-building config_dir changes -- ModuleConfigRegistry(args.config_dir)
    # below (criteria/weights/thresholds) always uses the real directory unchanged, since
    # those never vary by dashboard. For a module with no registered variants (e.g.
    # open_gpt) this is a no-op: variant_config_dir == args.config_dir, byte-identical to
    # before this flag existed.
    variant_config_dir = args.config_dir
    if MODULE_VARIANTS:
        default_variant = next(iter(MODULE_VARIANTS))
        print(
            f"[{{MODULE}}] --gpt-variant: {{args.gpt_variant!r}} "
            f"(available: {{sorted(MODULE_VARIANTS)!r}}, default: {{default_variant!r}})"
        )
    if variant_overrides:
        real_yaml_path = Path(args.config_dir) / f"{{MODULE}}.yaml"
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: reading base config from {{real_yaml_path}}")
        module_yaml = yaml.safe_load(real_yaml_path.read_text())
        before_request_template = dict(module_yaml.get("service", {{}}).get("request_template", {{}}))
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: base request_template = {{before_request_template!r}}")
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: applying field overrides = {{variant_overrides!r}}")
        module_yaml.setdefault("service", {{}}).setdefault("request_template", {{}}).update(variant_overrides)
        after_request_template = module_yaml["service"]["request_template"]
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: patched request_template = {{after_request_template!r}}")
        variant_dir = Path(tempfile.mkdtemp(prefix=f"{{MODULE}}_{{args.gpt_variant}}_"))
        variant_yaml_path = variant_dir / f"{{MODULE}}.yaml"
        variant_yaml_path.write_text(yaml.dump(module_yaml))
        atexit.register(shutil.rmtree, variant_dir, ignore_errors=True)
        variant_config_dir = str(variant_dir)
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: patched config written to {{variant_yaml_path}}")
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}}: request-building config_dir = {{variant_config_dir}} "
              f"(scoring config_dir stays {{args.config_dir}})")
    elif MODULE_VARIANTS:
        print(f"[{{MODULE}}] --gpt-variant {{args.gpt_variant!r}} is this module's default variant -- "
              f"no override needed, using {{args.config_dir}} as-is")

    if not Path(args.dataset).exists():
        print(
            f"[{{MODULE}}] dataset not found: {{args.dataset}}\\n"
            f"  Create it (input/golden per row, generated can be \\"\\") or pass --dataset to point elsewhere, "
            f"e.g. --dataset data/sample_dataset.jsonl for a quick smoke test.",
            file=sys.stderr,
        )
        return 2

    rows = [r for r in load_dataset(args.dataset) if r.module == MODULE]
    if not rows:
        print(f"[{{MODULE}}] no rows with module == {{MODULE!r}} found in {{args.dataset}}", file=sys.stderr)
        return 2

    out_path = Path(args.out or (Path(DEFAULT_OUTPUT_DIR) / f"{{MODULE}}{{out_variant_suffix}}_dataset.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # `state` is the single source of truth for every row's current record, keyed by
    # row_id -- loaded from disk on --resume (so a killed/crashed run's progress isn't
    # lost), then updated in memory as rows progress and rewritten to disk atomically
    # (see write_jsonl_atomic below) so each row's JSON entry is UPDATED in place rather
    # than duplicated by a fresh append. Each record carries a "status": "generated" /
    # "judged" / "failed_generate" / "failed_judge".
    state: dict[str, dict] = {{}}
    preloaded_rows: list[EvalRow] = []  # status "generated", not yet judged -- judge without regenerating.
    if args.resume:
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        state[d["row_id"]] = d

        before = len(rows)
        judged_ids = {{rid for rid, d in state.items() if d.get("status") == "judged"}}
        rows = [r for r in rows if r.row_id not in judged_ids]

        if not args.skip_generate:
            preloaded_rows = [
                EvalRow(
                    row_id=d["row_id"], module=d["module"], input=d["input"],
                    golden=d["golden"], generated=d["generated"], context=d.get("context"),
                )
                for r in rows if (d := state.get(r.row_id)) is not None and d.get("status") == "generated"
            ]
            preloaded_ids = {{r.row_id for r in preloaded_rows}}
            rows = [r for r in rows if r.row_id not in preloaded_ids]

        print(
            f"[{{MODULE}}] --resume: {{before - len(rows) - len(preloaded_rows)}}/{{before}} rows already judged "
            f"(skipping entirely), {{len(preloaded_rows)}} already generated in {{out_path}} "
            f"(judging only, not regenerating -- fed into the same live queue as fresh generation), "
            f"{{len(rows)}} remaining to generate + judge"
        )
        if not rows and not preloaded_rows:
            print(f"[{{MODULE}}] nothing left to do")
            return 0

    num_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    row_id_to_batch = {{r.row_id: (i // args.batch_size) + 1 for i, r in enumerate(preloaded_rows + rows)}}
    mode = (
        "batch-wise/sequential" if args.no_pipeline
        else "judging pre-filled rows" if args.skip_generate
        else f"streaming (row-wise, queue depth {{args.pipeline_depth}})"
    )
    if preloaded_rows:
        print(f"[{{MODULE}}] {{len(preloaded_rows)}} rows already generated -> merged into the live judge queue")
    print(
        f"[{{MODULE}}] {{len(rows)}} rows -> {{num_batches}} batch(es) of up to {{args.batch_size}}, {{mode}} "
        f"(skip_generate={{args.skip_generate}}, generate_workers={{args.generate_workers}}, judge={{args.judge!r}})"
    )

    judge_kwargs = {{}}
    if args.judge_model:
        judge_kwargs["model"] = args.judge_model
    if args.judge_timeout is not None:
        if args.judge != "nemotron":
            print(f"[{{MODULE}}] --judge-timeout is only supported by --judge nemotron, ignoring it", file=sys.stderr)
        else:
            judge_kwargs["timeout"] = args.judge_timeout
    judge = build_judge_provider(args.judge, **judge_kwargs)

    registry = default_registry(use_deepeval=args.use_deepeval)
    module_configs = ModuleConfigRegistry(args.config_dir)
    scheduler = ScorerScheduler(max_retries=args.judge_max_retries) if args.judge_max_retries is not None else None
    orchestrator = EvaluationOrchestrator(registry, module_configs, judge, scheduler=scheduler)
    store = ResultsStore(args.db)

    run_key = f"{{MODULE}}{{out_variant_suffix}}"  # separate regression-baseline history per --gpt-variant
    baseline_run_id = store.latest_run(run_key)
    run_id = store.start_run(run_key, baseline_run_id=baseline_run_id)

    # --- per-row log file: every row's status transitions, timestamped, append-only.
    # Also captures the scheduler's own retry/timeout warnings (evalsuite.scheduler
    # logs via the standard `logging` module under the "evalsuite" logger, which this
    # handler's parent-logger attachment picks up automatically -- no scheduler changes
    # needed) -- so "which row is being processed and what's its status" is answerable
    # by tailing this one file, not just scrollback.
    log_path = Path(args.log or (Path(DEFAULT_OUTPUT_DIR) / f"{{MODULE}}{{out_variant_suffix}}_run.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("evalsuite").addHandler(file_handler)
    logging.getLogger("evalsuite").setLevel(logging.INFO)
    row_log = logging.getLogger("evalsuite.run")
    if MODULE_VARIANTS:
        row_log.info(
            "gpt-variant=%s overrides=%s request_config_dir=%s scoring_config_dir=%s run_key=%s",
            args.gpt_variant, variant_overrides, variant_config_dir, args.config_dir, run_key,
        )
    row_log.info("=== starting run: %d row(s) to generate+judge, %d preloaded ===", len(rows), len(preloaded_rows))

    def write_jsonl_atomic(path, records):
        # Rewrites the WHOLE file from `records` (row_id -> record) -- so a row that's
        # already in it gets its entry UPDATED in place, not duplicated by another
        # append. Cheap at this scale (hundreds of rows): write a temp file, then an
        # atomic rename, so a crash mid-write never leaves a half-written file behind.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as f:
            for record in records.values():
                f.write(json.dumps(record) + "\\n")
        tmp_path.replace(path)

    # NOTE: on_row_generated (background generate thread) and on_row_judged (main
    # thread) now fire genuinely concurrently -- run_streaming's preload+generate
    # producer threads and the judge consumer all run in parallel. Without this
    # lock, two threads calling write_jsonl_atomic at the same moment race on the
    # SAME temp filename: whichever finishes its atomic rename second fails with
    # FileNotFoundError, since the first thread already moved the temp file away.
    # The lock also protects `state` itself from "dictionary changed size during
    # iteration" if one thread adds a row while another is mid-iteration
    # serializing it for the write.
    state_lock = threading.Lock()

    def update_state(row_id, record):
        with state_lock:
            state[row_id] = record
            write_jsonl_atomic(out_path, state)

    def update_state_many(records_by_id):
        with state_lock:
            state.update(records_by_id)
            write_jsonl_atomic(out_path, state)

    gen_failures_path = Path(DEFAULT_OUTPUT_DIR) / f"{{MODULE}}{{out_variant_suffix}}_generation_failures.jsonl"
    judge_failures_path = Path(DEFAULT_OUTPUT_DIR) / f"{{MODULE}}{{out_variant_suffix}}_judge_failures.jsonl"
    generation_failures = []  # (row, exception) -- appended from the generate thread in pipelined mode too
    judge_failures = []  # (row, exception) -- always appended from the main thread

    def on_generate_failure(row, exc):
        # NOTE: in pipelined/streaming mode (the default) this runs on the background
        # generation thread, not the main thread -- list.append and print are safe to
        # call from here under the GIL; update_state has its own lock (see above).
        generation_failures.append((row, exc))
        print(f"  [WARN] row {{row.row_id}} failed to generate, skipping it: {{exc}}", file=sys.stderr)
        row_log.warning("[%s] generation failed, skipped: %s\\n  Q: %s", row.row_id, exc, row.input)
        update_state(row.row_id, {{
            "status": "failed_generate", "batch": row_id_to_batch.get(row.row_id),
            "row_id": row.row_id, "module": row.module, "input": row.input,
            "golden": row.golden, "context": row.context, "error": str(exc),
        }})

    def on_judge_failure(row, exc):
        judge_failures.append((row, exc))
        print(f"  [WARN] row {{row.row_id}} failed to judge, skipping it: {{exc}}", file=sys.stderr)
        row_log.warning("[%s] judging failed, skipped: %s", row.row_id, exc)
        update_state(row.row_id, {{
            "status": "failed_judge", "batch": row_id_to_batch.get(row.row_id),
            "error": str(exc), **row.to_dict(),
        }})

    # --- row-wise callbacks: used by run_streaming (the default) and, for the
    # --resume "judge already-generated rows first" phase, always (there's no
    # generation happening in that phase either way, so row-wise progress is
    # strictly nicer there regardless of --no-pipeline). ---
    def on_row_generated(row):
        # Fires as soon as THIS row is generated, BEFORE it's handed to the judge --
        # store it now so it's safe on disk even if judging it later hangs, times
        # out, or the run is killed before reaching it.
        row_log.info("[%s] generated\\n  Q: %s\\n  A: %s", row.row_id, row.input, row.generated)
        if args.skip_generate:
            return
        update_state(row.row_id, {{"status": "generated", "batch": row_id_to_batch.get(row.row_id), **row.to_dict()}})

    def on_row_judged(result):
        store.save_results(run_id, [result])
        update_state(result.row.row_id, {{
            "status": "judged", "batch": row_id_to_batch.get(result.row.row_id), **result.to_dict(),
        }})
        status = "PASS" if result.passed else "FAIL"
        row_log.info("[%s] judged %s composite=%.3f", result.row.row_id, status, result.composite)
        print(f"  row {{result.row.row_id}}: {{status}}, composite {{result.composite:.3f}}")

    # --- batch-wise callbacks: only used by run_in_batches under --no-pipeline. ---
    def on_batch_generated(batch_index, batch_count, generated_batch):
        for r in generated_batch:
            row_log.info("[%s] generated (batch %d)\\n  Q: %s\\n  A: %s", r.row_id, batch_index, r.input, r.generated)
        if not args.skip_generate:
            update_state_many({{
                r.row_id: {{"status": "generated", "batch": batch_index, **r.to_dict()}} for r in generated_batch
            }})
        print(f"  batch {{batch_index}}/{{batch_count}}: {{len(generated_batch)}} rows generated -> queued for judging")

    def on_batch(batch_index, batch_count, generated_batch, batch_results):
        store.save_results(run_id, batch_results)

        for r in batch_results:
            row_log.info("[%s] judged %s composite=%.3f (batch %d)", r.row.row_id, "PASS" if r.passed else "FAIL", r.composite, batch_index)
        update_state_many({{
            r.row.row_id: {{"status": "judged", "batch": batch_index, **r.to_dict()}} for r in batch_results
        }})

        if not batch_results:
            # every row in this batch permanently failed judging (see on_judge_failure
            # above / output/{{MODULE}}_judge_failures.jsonl) -- nothing to average.
            print(f"  batch {{batch_index}}/{{batch_count}}: 0 rows judged (all failed, see judge failures file)")
            return
        batch_passed = sum(r.passed for r in batch_results)
        batch_avg = sum(r.composite for r in batch_results) / len(batch_results)
        print(
            f"  batch {{batch_index}}/{{batch_count}}: {{len(batch_results)}} rows judged, "
            f"{{batch_passed}}/{{len(batch_results)}} passed, avg composite {{batch_avg:.3f}}"
        )

    results = []
    if args.no_pipeline:
        # --no-pipeline: run_in_batches has no concept of a merged queue, so fall back
        # to the simpler two-phase approach -- judge the preloaded backlog first
        # (row-wise; there's no batch to wait on since no generation happens in this
        # phase), THEN generate+judge the rest, batch-wise.
        if preloaded_rows:
            print(f"[{{MODULE}}] judging {{len(preloaded_rows)}} previously-generated rows first...")
            results.extend(
                run_streaming(
                    preloaded_rows,
                    orchestrator,
                    skip_generate=True,
                    on_row_judged=on_row_judged,
                    on_judge_failure=on_judge_failure,
                )
            )
        if rows:
            results.extend(
                run_in_batches(
                    rows,
                    orchestrator,
                    config_dir=variant_config_dir,
                    batch_size=args.batch_size,
                    generate_max_workers=args.generate_workers,
                    skip_generate=args.skip_generate,
                    on_batch=on_batch,
                    on_batch_generated=on_batch_generated,
                    on_generate_failure=on_generate_failure,
                    on_judge_failure=on_judge_failure,
                )
            )
    elif rows or preloaded_rows:
        # the default: ONE run_streaming call merges `preloaded_rows` (a --resume
        # backlog that only needs judging) onto the SAME live judge queue as `rows`
        # (which still go through fresh generation) -- concurrently, not as a separate
        # blocking phase first, so the judge starts working through a resumed backlog
        # immediately while fresh generation proceeds at the same time.
        results.extend(
            run_streaming(
                rows,
                orchestrator,
                config_dir=variant_config_dir,
                batch_size=args.batch_size,
                generate_max_workers=args.generate_workers,
                skip_generate=args.skip_generate,
                queue_depth=args.pipeline_depth,
                preloaded_rows=preloaded_rows,
                on_row_generated=on_row_generated,
                on_row_judged=on_row_judged,
                on_generate_failure=on_generate_failure,
                on_judge_failure=on_judge_failure,
            )
        )

    def _partition_retryable(failures):
        retryable, permanent = [], []
        for row, exc in failures:
            (retryable if is_retryable_error(exc) else permanent).append((row, exc))
        return retryable, permanent

    for round_num in range(1, args.end_retries + 1):
        retryable_gen, permanent_gen = _partition_retryable(generation_failures)
        retryable_judge, permanent_judge = _partition_retryable(judge_failures)
        if not retryable_gen and not retryable_judge:
            break

        print(
            f"[{{MODULE}}] end-of-run retry {{round_num}}/{{args.end_retries}}: "
            f"{{len(retryable_gen)}} generation + {{len(retryable_judge)}} judge failure(s) look transient, retrying..."
        )
        generation_failures = permanent_gen
        judge_failures = permanent_judge

        # ONE call: retryable_judge rows (already generated) are merged onto the same
        # live queue as retryable_gen rows (need full regeneration) -- same concurrency
        # benefit as the main pass, not two sequential sub-phases.
        results.extend(
            run_streaming(
                [row for row, _ in retryable_gen],
                orchestrator,
                config_dir=variant_config_dir,
                batch_size=args.batch_size,
                generate_max_workers=args.generate_workers,
                queue_depth=args.pipeline_depth,
                preloaded_rows=[row for row, _ in retryable_judge],
                on_row_generated=on_row_generated,
                on_row_judged=on_row_judged,
                on_generate_failure=on_generate_failure,
                on_judge_failure=on_judge_failure,
            )
        )

    def write_failures(path, failures):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row, exc in failures:
                f.write(
                    json.dumps(
                        {{
                            "batch": row_id_to_batch.get(row.row_id),
                            "row_id": row.row_id,
                            "input": row.input,
                            "error": str(exc),
                        }}
                    )
                    + "\\n"
                )

    if generation_failures:
        write_failures(gen_failures_path, generation_failures)
    if judge_failures:
        write_failures(judge_failures_path, judge_failures)

    current_scores = store.avg_scores_for_run(run_id)
    baseline_scores = store.avg_scores_for_run(baseline_run_id)
    report = diff_against_baseline(current_scores, baseline_scores, run_id, baseline_run_id, args.regression_threshold)

    print(f"\\n=== {{MODULE}} ===")
    print(f"  output -> {{out_path}}")
    print(f"  row log -> {{log_path}}")
    print(report.render())

    if generation_failures:
        print(f"  {{len(generation_failures)}}/{{len(rows)}} rows failed to generate (skipped, not judged) -> {{gen_failures_path}}")
    if judge_failures:
        print(f"  {{len(judge_failures)}}/{{len(rows)}} rows failed to judge (skipped) -> {{judge_failures_path}}")

    failing_rows = [r for r in results if not r.passed]
    if failing_rows:
        print(f"  {{len(failing_rows)}}/{{len(results)}} rows failed a per-criterion threshold")
        for r in failing_rows:
            print(f"    row {{r.row.row_id}}: failed {{r.failed_criteria}}")

    row_log.info("=== run finished: %d judged, %d generation failures, %d judge failures ===",
                 len(results), len(generation_failures), len(judge_failures))
    logging.getLogger("evalsuite").removeHandler(file_handler)
    file_handler.close()
    store.close()

    if args.gate and (report.has_regression or failing_rows or generation_failures or judge_failures):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

for module, endpoints in MODULES.items():
    content = TEMPLATE.format(module=module, endpoints=endpoints)
    out_path = REPO_ROOT / "scripts" / f"run_{module}.py"
    out_path.write_text(content)
    out_path.chmod(0o755)
    print("wrote", out_path)
