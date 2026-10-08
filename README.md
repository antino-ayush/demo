# LLM Evaluation Framework

A working scaffold for the design in the "LLM Evaluation Framework — Design &
Architecture" doc: batch-scores a dataset of (golden, generated) pairs against
per-module criteria, using a pluggable LLM judge.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values — see "Configuration (.env)" below
python -m evalsuite.cli run \
  --dataset data/sample_dataset.jsonl \
  --config-dir configs/modules \
  --judge mock \
  --db eval_results.db
```

Runs fully offline with `--judge mock` — no API key needed. Useful for trying the
pipeline, CI, or local development. Swap in a real judge with `--judge claude`
(`pip install anthropic`, set `ANTHROPIC_API_KEY`), `--judge openai`
(`pip install openai`, set `OPENAI_API_KEY`), or `--judge nemotron` for a
self-hosted Nemotron judge called directly over HTTP (see below) — no extra
package needed, it uses `requests`.

### Generating `generated` answers first

`data/*.jsonl` files only need `input` + `golden` to start; `generated` can be
left as `""`. Run the model-under-test for a module's endpoint (configured in
`.env` + that module's `configs/modules/<module>.yaml`) to fill it in before
scoring:

```bash
python -m evalsuite.cli generate --dataset data/open_gpt_dataset.jsonl
python -m evalsuite.cli run --dataset output/open_gpt_dataset.jsonl \
  --config-dir configs/modules --judge nemotron --db eval_results.db
```

`generate` resolves each row's service endpoint from its `module` field (see
`evalsuite/generator.py`), calls it with `requests` (`Authorization: Bearer
<token>`), and writes the (`input`, `golden`, `generated`) rows to
`output/<same filename>` by default — `data/*.jsonl` golden sets are never
overwritten, so the same golden set can be regenerated against as many live
runs as you want. Pass `--out` to write somewhere else instead.

## Configuration (`.env`)

`cp .env.example .env` and fill in real values — `.env` is gitignored, never
commit real tokens/URLs. Two independent things are configured here:

- **The judge** (`evalsuite/judge_provider.py::NemotronJudgeProvider`) — one
  shared Nemotron deployment that scores every module: `NEMOTRON_BASE_URL`,
  `NEMOTRON_CHAT_PATH` (default `/v1/chat/completions`, OpenAI-compatible),
  `NEMOTRON_MODEL`, `NEMOTRON_API_KEY` (blank if the deployment needs no
  auth), `NEMOTRON_VERIFY_SSL`.
- **Each module's own service endpoint** (`evalsuite/generator.py`) — the
  model-under-test that `generate` calls to produce `generated` answers.
  `.env` only holds what's shared across every module: `SERVICE_BASE_URL`
  (the common host every module's service lives behind, e.g. a shared
  gateway) and `DEFAULT_SERVICE_TOKEN` (the shared bearer token). Everything
  module-specific lives in that module's `configs/modules/<module>.yaml`
  under an optional `service:` block: the path is just `endpoints[0]` from
  that same file (appended to `SERVICE_BASE_URL`), and `service.url` /
  `service.token` / `service.verify_ssl` / `service.timeout` /
  `service.offline_gpt_id` override the shared defaults for that one module
  when needed (e.g. `verify_ssl: false` for a self-signed internal cert, or
  `url:` for a module whose service isn't on the common host). See
  `configs/modules/gpts.yaml` / `open_gpt.yaml` for worked examples,
  including the `request_template` / `response_path` keys that let a module
  declare its own request/response JSON contract instead of the default
  Gemma-style one. Adding a new module under test means: one
  `configs/modules/<module>.yaml` (with its `endpoints:` and `service:`
  block) and one `data/<module>_dataset.jsonl` — no code change, and no
  `.env` change unless that module needs its own token/URL override.

Add `--use-deepeval` to swap faithfulness/completeness/conciseness/relevancy from
the hand-written G-Eval-style scorers to DeepEval-backed ones
(`pip install deepeval` first) — see `evalsuite/registry.py`.

Add `--gate` to make the run exit non-zero on a regression vs. the previous run for
a module, or on any row failing its per-criterion threshold — wire that into CI as
the release gate.

## Layout

| Path | Design doc component |
| --- | --- |
| `evalsuite/models.py` | `EvalRow`, `CriterionScore`, `ModuleConfig` |
| `evalsuite/dataset_loader.py` | Dataset Loader |
| `evalsuite/module_config.py` | Module Config Registry |
| `evalsuite/orchestrator.py` | Evaluation Orchestrator |
| `evalsuite/scheduler.py` | Scorer Scheduler |
| `evalsuite/scorer.py` | `Scorer` interface (Template Method / Strategy) |
| `evalsuite/scorers/geval_scorers.py` | hand-written G-Eval-style Scorers |
| `evalsuite/scorers/deepeval_scorers.py` | DeepEval-wrapped Scorers |
| `evalsuite/registry.py` | `ScorerRegistry` — binds criterion name → Scorer |
| `evalsuite/judge_provider.py` | `JudgeProvider` (Mock / Claude / OpenAI / Nemotron) + factory |
| `evalsuite/aggregator.py` | Score Aggregator |
| `evalsuite/results_store.py` | Results Store (SQLite) |
| `evalsuite/reporting.py` | Reporting / Diffing |
| `evalsuite/cli.py` | batch runner / CI-gate entry point, plus the `generate` subcommand |
| `evalsuite/generator.py` | Response Generator — calls each module's service endpoint (`requests`) to fill in `generated` |
| `evalsuite/env.py` | loads `.env` once (`python-dotenv`), read by `cli.py` / `generator.py` |
| `configs/modules/*.yaml` | one `ModuleConfig` per endpoint type, incl. the `endpoints:` list (endpoint → module mapping) |
| `data/sample_dataset.jsonl` | tiny example dataset — includes one deliberate hallucination row (`qa-2`) and one deliberate format violation (`ext-2`) |
| `data/<module>_dataset.jsonl` | one golden dataset per module (e.g. `data/open_gpt_dataset.jsonl` for `open_gpt`) — each module has its own dataset and its own service endpoint |
| `tests/` | pytest suite; runs fully offline via the mock judge |

## Running the tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -v
```

## How the hybrid (build vs. buy) actually works here

Every criterion is a `Scorer` behind one interface (`evalsuite/scorer.py`).
`evalsuite/registry.py::default_registry` is where each criterion name gets bound
to a concrete implementation — a hand-written G-Eval-style prompt
(`scorers/geval_scorers.py`) by default, or a DeepEval wrapper
(`scorers/deepeval_scorers.py`) when `use_deepeval=True`. Nothing else in the
pipeline (Orchestrator, Scheduler, Aggregator, Results Store) knows or cares which
kind of `Scorer` it's calling.

## Mapping endpoints to modules, and where scores map back

The endpoint → module mapping lives in `configs/modules/*.yaml`, in each config's
`endpoints:` list (e.g. `/api/v1/qa` → `rag_qa`) — edit those lists to your real API
routes or service names. `ModuleConfigRegistry.module_for_endpoint()` in
`evalsuite/module_config.py` resolves a raw endpoint name to its module by scanning
those lists; use it wherever you ingest real traffic into a dataset, so you tag each
captured row with the endpoint that produced it and resolve `module` once, instead
of hand-labeling every row. It raises if no config claims an endpoint, and refuses
to load (`ValueError`) if two configs both claim the same one.

Score mapping back to a module runs through `evalsuite/results_store.py`: every
`runs` row is scoped to exactly one module (`cli.py` groups the dataset by module
before starting a run), and every `scores`/`rows` row is tied to that `run_id` — so
`scores → rows → runs` is the join that gets you back to "this criterion's score,
for this row, for this module". `ResultsStore.avg_scores_for_run()` and
`reporting.diff_against_baseline()` are what turn that into the per-module report
printed by the CLI. To query it directly:

```sql
SELECT r.module, s.criterion, AVG(s.score)
FROM scores s JOIN runs r ON s.run_id = r.run_id
GROUP BY r.module, s.criterion;
```

## Notes / limitations

This is a scaffold meant to match the design doc's HLD/LLD and be genuinely
runnable, not a production system:

- The scheduler is a plain `ThreadPoolExecutor` — fine for hundreds of rows;
  swap for a real async queue at higher volume.
- The mock judge (`MockJudgeProvider`) is a crude token-overlap heuristic. It
  exists only to make the pipeline exercisable and testable offline — it is
  not a stand-in for real judgment quality.
- Chain-of-thought evaluation steps in `geval_scorers.py` are static per
  scorer class. The design doc's LLD suggests generating and caching these
  once per criterion via a judge call at startup instead — that hook is a
  natural next addition, not implemented here.
- `--gate`'s regression check compares averages within a single run; the doc's
  Improvements section (drift monitoring, human calibration, ensembling)
  describes further hardening not implemented in this scaffold.
