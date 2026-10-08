"""Classifies exceptions from generation/judging as transient (worth one
more retry) vs. permanent (will fail identically no matter how many times
you retry). Reuses ResponseGenerator's and ScorerScheduler's own retry
judgments as the source of truth, rather than re-deriving a second opinion
that could drift from theirs.
"""
from __future__ import annotations

import requests

from .generator import ResponseGenerator
from .scheduler import RETRYABLE_EXCEPTIONS


def is_retryable_error(exc: Exception) -> bool:
    """True for a timeout, dropped connection, or HTTP 429/5xx -- the same
    exception types/status codes ResponseGenerator.generate_one and
    ScorerScheduler._score_with_retry already treat as transient. False for
    anything else, including a 4xx HTTPError (permanent by nature -- retrying
    422/401 wastes a call and will fail identically) and any exception type
    not specifically recognized (conservative default: don't assume an
    unknown failure is safe to retry blindly)."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in ResponseGenerator.RETRYABLE_STATUS_CODES
    return False
