"""Scorer Scheduler: batches/rate-limits judge calls, retries fixable
failures, fails fast on non-retryable ones.

Kept deliberately simple (a ThreadPoolExecutor) -- fine for hundreds of
rows. Swap for asyncio or a real task queue at higher volume without
changing the Orchestrator or any Scorer.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .judge_provider import JudgeProvider
from .models import CriterionScore, EvalRow
from .scorer import Scorer

logger = logging.getLogger(__name__)

# NOTE: requests.exceptions.{Timeout,ConnectionError} are NOT subclasses of
# the builtin TimeoutError/ConnectionError (verified: ReadTimeout's MRO is
# Timeout -> RequestException -> OSError, never touching the builtins), so
# both forms are listed explicitly -- the builtins alone silently never
# matched a real HTTP read timeout from NemotronJudgeProvider (or any other
# requests-based JudgeProvider), which meant retries never actually fired
# for the single most common real-world judge failure.
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


class ScorerScheduler:
    def __init__(self, max_workers: int = 8, max_retries: int = 2, backoff_seconds: float = 1.0) -> None:
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def run(self, row: EvalRow, tasks: list[tuple[str, Scorer]], judge: JudgeProvider) -> list[CriterionScore]:
        results: dict[str, CriterionScore] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._score_with_retry, scorer, row, judge): criterion
                for criterion, scorer in tasks
            }
            for future in as_completed(futures):
                criterion = futures[future]
                results[criterion] = future.result()
        return [results[c] for c, _ in tasks]

    def _score_with_retry(self, scorer: Scorer, row: EvalRow, judge: JudgeProvider) -> CriterionScore:
        attempt = 0
        while True:
            try:
                return scorer.score(row, judge)
            except RETRYABLE_EXCEPTIONS as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                logger.warning(
                    "retrying %s on row %s after %s (attempt %d/%d)",
                    scorer.criterion_name, row.row_id, exc, attempt, self.max_retries,
                )
                time.sleep(self.backoff_seconds * attempt)
            except Exception:
                logger.exception(
                    "non-retryable failure scoring %s on row %s", scorer.criterion_name, row.row_id
                )
                raise
