from unittest.mock import MagicMock

import pytest
import requests

from evalsuite.errors import is_retryable_error


def _http_error(status_code=None):
    resp = None
    if status_code is not None:
        resp = MagicMock()
        resp.status_code = status_code
    return requests.exceptions.HTTPError("boom", response=resp)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (requests.exceptions.ReadTimeout("timed out"), True),
        (requests.exceptions.ConnectionError("dropped"), True),
        (TimeoutError("builtin timeout"), True),  # scheduler.py explicitly retries this form too
        (ConnectionError("builtin connection error"), True),  # same -- not just requests.exceptions.*
        (_http_error(429), True),
        (_http_error(500), True),
        (_http_error(503), True),
        (_http_error(422), False),  # permanent: bad request, will fail identically every retry
        (_http_error(401), False),  # permanent: auth failure
        (_http_error(None), False),  # no response object at all (e.g. connection-failure HTTPError)
        (ValueError("bad json from judge"), False),  # unrecognized type: conservative default
    ],
)
def test_is_retryable_error(exc, expected):
    assert is_retryable_error(exc) is expected
