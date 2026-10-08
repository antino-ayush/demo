"""Batch pipeline: generate + judge a dataset in fixed-size batches instead
of generating every row before judging starts.

For a big dataset (hundreds of rows), `fill_dataset` (generator.py) calls
the model-under-test one row at a time, and `EvaluationOrchestrator.run`
(orchestrator.py) judges one row at a time (parallel only *within* a row,
across its criteria). This module adds a coarser, batch-level pipeline on
top of both, unchanged:

  - up to `generate_max_workers` HTTP calls to the model-under-test service
    are in flight at once within a batch (instead of one row at a time),
  - the judge starts scoring batch N as soon as batch N finishes
    generating, instead of waiting for every row in the dataset to be
    generated first,
  - progress and partial results are visible/persisted after every batch
    (via the `on_batch` callback), not just at the very end.

Three ways to run through a dataset, below:

  - `run_in_batches` -- strictly sequential: batch N+1's generation starts
    only after batch N has been judged. Simple, bounded, predictable.
  - `run_pipelined` -- overlapping at the BATCH level: a background thread
    keeps generating batch N+1 (and, depth permitting, further ahead) while
    the main thread is still judging batch N. Same batch-by-batch progress/
    flushing via `on_batch`, just less wall-clock time end to end.
  - `run_streaming` -- overlapping at the ROW level: generation still
    happens in batch-sized chunks, but each row is judged the instant it's
    generated, not batched with the rest of its chunk. Prefer this when the
    judge is the slow/unreliable side -- with `run_pipelined`, one stuck row
    anywhere in a batch delays every other row in that batch from even
    starting; here a stuck row only blocks itself.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .aggregator import RowResult
from .generator import DEFAULT_CONFIG_DIR, ResponseGenerator, ServiceConfig, ServiceContract
from .models import EvalRow
from .orchestrator import EvaluationOrchestrator

logger = logging.getLogger(__name__)

#: called after each batch finishes judging: (batch_index, batch_count, generated_rows, batch_results)
BatchCallback = Callable[[int, int, list[EvalRow], list[RowResult]], None]

#: called right after a batch finishes GENERATING, before it's judged:
#: (batch_index, batch_count, generated_rows). This is the hook for "store the
#: batch to disk the moment it's generated" -- in `run_pipelined` it fires
#: from the background generation thread, before that batch is handed to the
#: judge via the queue, so the generated answers are safely persisted even if
#: judging that batch later times out, crashes, or is never reached at all.
GeneratedBatchCallback = Callable[[int, int, list[EvalRow]], None]

#: called once per row that permanently failed to generate (retries exhausted
#: inside ResponseGenerator.generate_one, or any other exception from it):
#: (row, exception). The row is simply excluded from that batch's judging --
#: one flaky row (e.g. a backend timeout) shouldn't cost you every other row
#: in the dataset.
FailureCallback = Callable[[EvalRow, Exception], None]

#: called the instant ONE row finishes generating (run_streaming only, not
#: batched): (row). Fires before that row is handed to the judge.
RowCallback = Callable[[EvalRow], None]

#: called the instant ONE row finishes judging (run_streaming only): (result).
RowResultCallback = Callable[[RowResult], None]


def chunked(rows: list[EvalRow], size: int) -> Iterator[list[EvalRow]]:
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def generate_batch_concurrent(
    rows: list[EvalRow],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    max_workers: int = 8,
    on_failure: FailureCallback | None = None,
) -> list[EvalRow]:
    """Fills `generated` for every row in `rows` concurrently -- up to
    `max_workers` calls to each row's module service in flight at once --
    and returns the successfully-generated rows, in the same relative order
    as `rows`.

    A row whose generation call raises (retries already exhausted inside
    `generate_one`, e.g. a persistent backend timeout) is NOT retried again
    here and does NOT fail the batch -- it's dropped from the result and
    reported via `on_failure(row, exc)` if given, so a single bad row can't
    take down the whole run.

    One ServiceConfig/ServiceContract per distinct module among `rows` is
    resolved once (same caching `fill_dataset` does) and reused across the
    whole batch, so mixed-module batches only pay the YAML/env lookup once
    per module, not once per row.
    """
    configs: dict[str, ServiceConfig] = {}
    contracts: dict[str, ServiceContract] = {}
    for row in rows:
        if row.module not in configs:
            configs[row.module] = ServiceConfig.from_env(row.module, config_dir)
            contracts[row.module] = ServiceContract.from_module_config(row.module, config_dir)
    generator_for = {module: ResponseGenerator(configs[module], contracts[module]) for module in configs}

    generated: list[Optional[EvalRow]] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(generator_for[row.module].generate_one, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                generated[i] = future.result()
            except Exception as exc:
                logger.warning("row %s: generation failed, dropping it from this run: %s", rows[i].row_id, exc)
                if on_failure:
                    on_failure(rows[i], exc)
    return [r for r in generated if r is not None]


def run_in_batches(
    rows: list[EvalRow],
    orchestrator: EvaluationOrchestrator,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch_size: int = 50,
    generate_max_workers: int = 8,
    skip_generate: bool = False,
    on_batch: BatchCallback | None = None,
    on_batch_generated: GeneratedBatchCallback | None = None,
    on_generate_failure: FailureCallback | None = None,
    on_judge_failure: FailureCallback | None = None,
) -> list[RowResult]:
    """Generates and judges `rows` in fixed-size batches: each batch is
    generated concurrently (skipped if `skip_generate`, e.g. `rows` already
    has `generated` filled in), then immediately judged, before moving on to
    the next batch. `on_batch_generated`, if given, runs right after a batch
    finishes generating (before it's judged) -- store it to disk here so
    generated answers survive even if judging that batch fails. `on_batch`
    runs after each batch is judged -- typically used to flush the *scored*
    rows to disk and print progress, instead of waiting for the whole
    dataset to finish. `on_generate_failure` / `on_judge_failure`, if given,
    run once per row that permanently failed to generate / to score
    (respectively) -- such a row is excluded from the results, not counted
    as a failure of the whole batch/run."""
    batches = list(chunked(rows, batch_size))
    all_results: list[RowResult] = []
    for batch_index, batch in enumerate(batches, start=1):
        if skip_generate:
            generated_batch = batch
        else:
            logger.info("batch %d/%d: generating %d rows", batch_index, len(batches), len(batch))
            generated_batch = generate_batch_concurrent(
                batch, config_dir, generate_max_workers, on_failure=on_generate_failure
            )
        if not generated_batch:
            continue  # every row in this batch failed to generate -- nothing to judge
        if on_batch_generated:
            on_batch_generated(batch_index, len(batches), generated_batch)

        logger.info("batch %d/%d: judging %d rows", batch_index, len(batches), len(generated_batch))
        batch_results = orchestrator.run(generated_batch, on_failure=on_judge_failure)
        all_results.extend(batch_results)

        if on_batch:
            on_batch(batch_index, len(batches), generated_batch, batch_results)

    return all_results


def run_pipelined(
    rows: list[EvalRow],
    orchestrator: EvaluationOrchestrator,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch_size: int = 50,
    generate_max_workers: int = 8,
    skip_generate: bool = False,
    pipeline_depth: int = 1,
    on_batch: BatchCallback | None = None,
    on_batch_generated: GeneratedBatchCallback | None = None,
    on_generate_failure: FailureCallback | None = None,
    on_judge_failure: FailureCallback | None = None,
) -> list[RowResult]:
    """Like `run_in_batches`, but overlaps generation and judging across
    batches: a background thread generates batch N+1 (concurrently within
    that batch, same as `generate_batch_concurrent`) while the main thread
    is still judging batch N, instead of the two phases alternating in lock
    step. Net effect: total wall time is roughly
    `max(total generate time, total judge time)` instead of their sum.

    The flow per batch is exactly: generate it, hand it to `on_batch_generated`
    (store it), THEN enqueue it for the judge -- so a batch's generated
    answers are on disk before the judge ever sees them, and a judge that
    hangs, times out, or crashes on that batch never costs you the
    already-successful generation work.

    `pipeline_depth` bounds how many already-generated batches may be
    queued up waiting for the judge (1 = generate one batch ahead, which is
    enough to fully overlap generation with judging as long as judging a
    batch takes at least as long as generating the next one; raise it if
    generation is slower per-batch than judging and the judge would
    otherwise stall waiting).

    `on_batch` and `on_judge_failure` both run on the main thread (judging
    itself never leaves the main thread here) -- same contract as
    `run_in_batches`, so callers don't need to know which of the two
    functions produced a given call. `on_batch_generated` and
    `on_generate_failure` both run on the background generation thread
    instead, so keep them simple/side-effect light (e.g. print, append to a
    list, or write+flush a file -- all fine under the GIL; avoid anything
    that assumes it's only ever called from one thread at a time relative
    to `on_batch`/`on_judge_failure`).

    Falls back to plain sequential batches when `skip_generate` is set:
    there's no generation latency to hide behind judging, so the background
    thread/queue would add overhead for no benefit.
    """
    batches = list(chunked(rows, batch_size))
    if skip_generate:
        return run_in_batches(
            rows,
            orchestrator,
            config_dir=config_dir,
            batch_size=batch_size,
            skip_generate=True,
            on_batch=on_batch,
            on_batch_generated=on_batch_generated,
            on_judge_failure=on_judge_failure,
        )

    work_queue: queue.Queue = queue.Queue(maxsize=pipeline_depth)
    stop_event = threading.Event()

    def put_or_stop(item: tuple[str, object]) -> None:
        # blocks on a full queue, but re-checks stop_event periodically instead
        # of blocking forever -- otherwise, if the consumer dies (exception
        # while judging) while this queue is full, this thread would block on
        # put() forever and producer_thread.join() below would hang the process.
        while not stop_event.is_set():
            try:
                work_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def producer() -> None:
        try:
            for batch_index, batch in enumerate(batches, start=1):
                if stop_event.is_set():
                    return
                generated_batch = generate_batch_concurrent(
                    batch, config_dir, generate_max_workers, on_failure=on_generate_failure
                )
                if generated_batch:  # skip a batch that failed to generate entirely
                    if on_batch_generated:
                        # fires here, BEFORE the batch is queued for the judge -- so it's
                        # stored on disk even if the judge never gets to it (crash, kill,
                        # or a run stopped early) rather than only after judging succeeds.
                        on_batch_generated(batch_index, len(batches), generated_batch)
                    put_or_stop(("ok", (batch_index, generated_batch)))
        except Exception as exc:  # propagated to the consumer below, not swallowed
            put_or_stop(("error", exc))
            return
        put_or_stop(("done", None))

    producer_thread = threading.Thread(target=producer, name="batch-generate", daemon=True)
    producer_thread.start()

    all_results: list[RowResult] = []
    try:
        while True:
            status, payload = work_queue.get()
            if status == "done":
                break
            if status == "error":
                raise payload

            # batch_index travels with the payload (set by the producer, position-based
            # over the original `batches` list) rather than being a separate counter
            # incremented per dequeue here -- otherwise the two would silently drift
            # apart the moment any batch fails to generate entirely (skipped, never
            # queued), and on_batch_generated/on_batch would tag the same rows with
            # two different batch numbers.
            batch_index, generated_batch = payload
            logger.info("batch %d/%d: judging %d rows", batch_index, len(batches), len(generated_batch))
            batch_results = orchestrator.run(generated_batch, on_failure=on_judge_failure)
            all_results.extend(batch_results)

            if on_batch:
                on_batch(batch_index, len(batches), generated_batch, batch_results)
    except BaseException:
        stop_event.set()  # unblock the producer if it's mid put_or_stop, so join() below can't hang
        raise
    finally:
        producer_thread.join(timeout=10)

    return all_results


def run_streaming(
    rows: list[EvalRow],
    orchestrator: EvaluationOrchestrator,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch_size: int = 50,
    generate_max_workers: int = 8,
    skip_generate: bool = False,
    queue_depth: int = 20,
    preloaded_rows: list[EvalRow] | None = None,
    on_row_generated: RowCallback | None = None,
    on_row_judged: RowResultCallback | None = None,
    on_generate_failure: FailureCallback | None = None,
    on_judge_failure: FailureCallback | None = None,
) -> list[RowResult]:
    """Generation is still batched (rows are submitted to the model-under-test
    service in `batch_size`-sized chunks, up to `generate_max_workers` calls
    in flight within a chunk -- bounding both memory and how far ahead of the
    judge generation can run), but the hand-off to the judge is per ROW, not
    per batch: each row is judged the instant it finishes generating, instead
    of waiting for its whole chunk to complete first. This matters when
    judging is the slow/unreliable side (a flaky judge backend) -- with
    batch-level hand-off, one slow/stuck row anywhere in a chunk delays
    every other row in that chunk from even starting; here, a stuck row
    only blocks itself.

    `preloaded_rows`, if given, are rows that already have `generated` filled
    in (e.g. recovered from a previous run's output via --resume) -- they're
    pushed onto the SAME judge queue as `rows` (which still go through the
    full generate step), immediately and ahead of any newly-generated row, so
    the judge can start working through a resumed backlog from the first
    instant instead of waiting for it as a separate blocking phase before
    fresh generation even starts. `on_row_generated` does NOT fire for
    preloaded rows -- there's nothing new to store, they're already on disk.
    `on_generate_failure` never fires for them either, by construction.

    `on_row_generated(row)` fires as soon as that row is generated, before
    it's judged -- store it now so it survives even if judging it later
    hangs, times out, or the run is killed before reaching it (same idea as
    `on_batch_generated` in `run_in_batches`/`run_pipelined`, just per row).
    `on_row_judged(result)` fires as soon as that row is judged.
    `on_generate_failure`/`on_judge_failure` fire once per row that
    permanently failed generation/judging (same contract as elsewhere in
    this module) -- such a row is excluded from the results, not counted as
    a failure of the run.

    `on_row_generated` and `on_generate_failure` run on a background
    generation thread; `on_row_judged` and `on_judge_failure` run on the
    main thread (judging itself never leaves it here) -- keep both simple/
    side-effect light (print, append to a list, write+flush a file are all
    fine under the GIL).

    `skip_generate=True` skips generation entirely (every row -- from `rows`
    AND `preloaded_rows`, treated identically here -- already has `generated`
    filled in) and just judges them in order, one at a time, calling
    `on_row_judged` per row -- no background thread needed since there's no
    generation latency to overlap with judging.
    """
    if skip_generate:
        results: list[RowResult] = []
        for row in [*(preloaded_rows or []), *rows]:
            row_results = orchestrator.run([row], on_failure=on_judge_failure)
            if row_results:
                result = row_results[0]
                results.append(result)
                if on_row_judged:
                    on_row_judged(result)
        return results

    configs: dict[str, ServiceConfig] = {}
    contracts: dict[str, ServiceContract] = {}
    for row in rows:
        if row.module not in configs:
            configs[row.module] = ServiceConfig.from_env(row.module, config_dir)
            contracts[row.module] = ServiceContract.from_module_config(row.module, config_dir)
    generator_for = {module: ResponseGenerator(configs[module], contracts[module]) for module in configs}

    row_queue: queue.Queue = queue.Queue(maxsize=queue_depth)
    stop_event = threading.Event()

    def put_or_stop(item: tuple[str, object]) -> None:
        while not stop_event.is_set():
            try:
                row_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    # Two producer threads feed the SAME queue, running concurrently -- not one
    # producer that seeds preloaded_rows to completion before starting on rows.
    # With a bounded queue (queue_depth), a single sequential producer would
    # spend its entire time blocked feeding a large preloaded backlog through
    # the queue at the judge's pace, and never even START fresh generation
    # until that backlog was nearly drained -- defeating the point of merging
    # them (the judge working through a resumed backlog should never come at
    # the cost of fresh generation sitting idle). `_producers_remaining`
    # tracks how many of the two threads are still running; whichever one
    # finishes last sends the "done" sentinel, under a lock so exactly one
    # thread ever sends it.
    producers_remaining = [2]
    producers_lock = threading.Lock()

    def _producer_finished() -> None:
        with producers_lock:
            producers_remaining[0] -= 1
            if producers_remaining[0] == 0:
                put_or_stop(("done", None))

    def preload_producer() -> None:
        try:
            for row in (preloaded_rows or []):
                if stop_event.is_set():
                    return
                put_or_stop(("ok", row))
        except Exception as exc:  # propagated to the consumer below, not swallowed
            put_or_stop(("error", exc))
        finally:
            _producer_finished()

    def generate_producer() -> None:
        try:
            for chunk in chunked(rows, batch_size):
                if stop_event.is_set():
                    return
                with ThreadPoolExecutor(max_workers=generate_max_workers) as pool:
                    futures = {pool.submit(generator_for[row.module].generate_one, row): row for row in chunk}
                    for future in as_completed(futures):
                        if stop_event.is_set():
                            return
                        row = futures[future]
                        try:
                            generated_row = future.result()
                        except Exception as exc:
                            logger.warning(
                                "row %s: generation failed, dropping it from this run: %s", row.row_id, exc
                            )
                            if on_generate_failure:
                                on_generate_failure(row, exc)
                            continue
                        if on_row_generated:
                            # fires here, BEFORE this row is queued for the judge.
                            on_row_generated(generated_row)
                        put_or_stop(("ok", generated_row))
        except Exception as exc:  # propagated to the consumer below, not swallowed
            put_or_stop(("error", exc))
        finally:
            _producer_finished()

    preload_thread = threading.Thread(target=preload_producer, name="row-preload", daemon=True)
    generate_thread = threading.Thread(target=generate_producer, name="row-generate", daemon=True)
    preload_thread.start()
    generate_thread.start()

    results: list[RowResult] = []
    try:
        while True:
            status, payload = row_queue.get()
            if status == "done":
                break
            if status == "error":
                raise payload

            row = payload
            row_results = orchestrator.run([row], on_failure=on_judge_failure)
            if row_results:
                result = row_results[0]
                results.append(result)
                if on_row_judged:
                    on_row_judged(result)
    except BaseException:
        stop_event.set()  # unblock both producers if either is mid put_or_stop, so joins below can't hang
        raise
    finally:
        preload_thread.join(timeout=10)
        generate_thread.join(timeout=10)

    return results
