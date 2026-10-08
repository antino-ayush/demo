"""Response Generator: calls each module's configured service endpoint over
HTTP (via `requests`) to fill in the `generated` field of a golden dataset,
turning (input, golden)-only rows into the (input, golden, generated) triples
`dataset_loader`/`EvalRow` expect. This is the model-under-test side of the
pipeline; `judge_provider.py` is the analogous swappable-provider pattern on
the scoring side.

.env holds only what's genuinely environment-specific and shared across every
module: SERVICE_BASE_URL (the common host every module's service lives
behind) and DEFAULT_SERVICE_TOKEN (the shared bearer token), alongside the
judge's NEMOTRON_* block. See .env.example.

Everything module-specific -- path, request/response contract, and any
per-module override of the token/TLS/timeout/offline_gpt_id -- lives in that
module's configs/modules/<module>.yaml, under an optional `service:` block.
Adding a module is a YAML change, never a code or .env change:

    module: varuna_gpt
    endpoints: [/varuna]        # also the path appended to SERVICE_BASE_URL
    service:
      url: https://other-host/varuna   # optional: full URL override, skips
                                        # SERVICE_BASE_URL + endpoints[0]
      token: ...                       # optional: overrides DEFAULT_SERVICE_TOKEN. A literal
                                        # string, or "{env:SOME_VAR}" to pull a distinct
                                        # module-specific secret out of .env instead of
                                        # hardcoding it in this (non-secret) yaml file.
      verify_ssl: false                # optional, default true
      timeout: 30                      # optional, default 60 (seconds)
      request_template:
        query: "{input}"
        session_id: "{uuid}"
      response_path: result.answer

`request_template` is rendered per-row: any string value containing
`{input}`, `{row_id}`, `{module}`, `{uuid}` (a fresh uuid4 per occurrence), or
`{env:SOME_VAR}` (read from the environment) has those placeholders
substituted; everything else in the template (numbers, bools, nested
dicts/lists) is passed through as-is. `response_path` is a dotted path (list
indices as plain integers, e.g. `choices.0.message.content`) into the JSON
response identifying the string field to use as the generated answer.

A module with no `request_template`/`response_path` falls back to the
original question/module/session_id request (with `offline_gpt_id` from
`service.offline_gpt_id`, default "") and the answer/response/output/...
response key scan -- the Gemma-style contract observed for open_gpt.

Some services stream the answer back instead of returning one JSON object,
so `response_path` doesn't apply -- set `service.response_stream_key` to the
per-chunk field name instead, and every chunk's value for that key is
concatenated in order to form the generated answer. Two wire shapes are
supported, auto-detected from the body:

  - Server-Sent Events (SSE) -- what offline_chatbot/local_llm actually
    sends: `event: <name>` lines followed by `data: <json>` lines, blocks
    separated by a blank line, e.g.:

        event: token
        data: {"status": "streaming", "data": "Hel"}

        event: token
        data: {"status": "streaming", "data": "lo"}

    Only the `data:` lines' JSON payloads are parsed; `event:` lines and
    blank lines are ignored.

  - Bare back-to-back JSON objects, no framing at all (no separator
    required, whitespace/newlines between them are fine), e.g.:

        {"status": "streaming", "data": "Hel"}{"status": "streaming", "data": "lo"}

Whichever shape is detected (by whether the body has any line starting with
`data:`) is used for the whole body.

A 429 response is retried automatically (honoring the response's
`Retry-After` or `RateLimit-Reset` header if present, else
`service.retry_backoff_seconds`, default 5s), up to `service.max_retries`
times (default 3) -- generating a whole dataset easily exceeds a per-minute
rate limit a single request never would.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

from .dataset_loader import load_dataset
from .models import EvalRow
from .request_schemas import REQUEST_SCHEMAS

DEFAULT_CONFIG_DIR = "configs/modules"


_PLACEHOLDER_RE = re.compile(r"\{(input|row_id|module|uuid|env:[A-Za-z0-9_]+)\}")


def _render_value(value: Any, row: EvalRow) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            token = match.group(1)
            if token == "input":
                return row.input
            if token == "row_id":
                return row.row_id
            if token == "module":
                return row.module
            if token == "uuid":
                return str(uuid.uuid4())
            return os.environ.get(token[len("env:"):], "")

        return _PLACEHOLDER_RE.sub(repl, value)
    if isinstance(value, dict):
        return {key: _render_value(val, row) for key, val in value.items()}
    if isinstance(value, list):
        return [_render_value(val, row) for val in value]
    return value


def _load_module_yaml(module: str, config_dir: str | Path) -> dict[str, Any]:
    path = Path(config_dir) / f"{module}.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _first_endpoint_path(module: str, config_dir: str | Path) -> str | None:
    endpoints = _load_module_yaml(module, config_dir).get("endpoints") or []
    return endpoints[0] if endpoints else None


def _extract_path(data: Any, path: str) -> str:
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"response_path {path!r}: no index {part!r} in list response") from exc
        elif isinstance(current, dict):
            if part not in current:
                raise ValueError(f"response_path {path!r}: no key {part!r} in {current!r}")
            current = current[part]
        else:
            raise ValueError(f"response_path {path!r}: hit non-container value before finishing the path")
    if not isinstance(current, str):
        raise ValueError(f"response_path {path!r} resolved to a {type(current).__name__}, expected a string")
    return current


_SSE_DATA_LINE_RE = re.compile(r"(?m)^data:")


def _concat_json_stream(text: str, key: str) -> str:
    """Parses `text` as a stream of JSON objects -- SSE (`data:` lines) or
    bare back-to-back objects, auto-detected -- and concatenates each
    object's `key` string value, in order. See the module docstring above
    for the two supported wire shapes."""
    if _SSE_DATA_LINE_RE.search(text):
        return _concat_sse_stream(text, key)
    return _concat_bare_json_stream(text, key)


def _concat_sse_stream(text: str, key: str) -> str:
    decoder = json.JSONDecoder()
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue  # e.g. "event: token", or the blank line between SSE blocks
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        obj, _ = decoder.raw_decode(payload)
        value = obj.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts)


def _concat_bare_json_stream(text: str, key: str) -> str:
    decoder = json.JSONDecoder()
    parts: list[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        value = obj.get(key)
        if isinstance(value, str):
            parts.append(value)
        idx = end
    return "".join(parts)


@dataclass
class ServiceContract:
    """Per-module request/response JSON shape, from that module's YAML.

    `request_template` / `response_path` of None mean "use the default
    Gemma-style contract" (see module docstring above). `response_stream_key`
    set means the response body is a stream of concatenated JSON objects,
    not a single JSON value -- see module docstring.
    """

    module: str
    request_template: dict[str, Any] | None = None
    response_path: str | None = None
    response_stream_key: str | None = None
    offline_gpt_id: str = ""

    @classmethod
    def from_module_config(cls, module: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> "ServiceContract":
        service = _load_module_yaml(module, config_dir).get("service") or {}
        return cls(
            module=module,
            request_template=service.get("request_template"),
            response_path=service.get("response_path"),
            response_stream_key=service.get("response_stream_key"),
            offline_gpt_id=service.get("offline_gpt_id") or "",
        )


@dataclass
class ServiceConfig:
    """Where + how to call the model-under-test service for one module."""

    module: str
    url: str
    token: str | None = None
    verify_ssl: bool = True
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0
    #: if set, no more than this many requests/minute are sent to `url`, shared
    #: across every concurrent worker and every batch within one process (see
    #: `_rate_limiter_for`) -- a proactive pace, not a reactive 429 retry.
    #: Some backends respond to sustained bursts past their own rate limit by
    #: temporarily rejecting the token/session outright (401) rather than just
    #: 429ing individual requests, so staying under the limit in the first
    #: place matters more than retrying after the fact once that happens.
    max_requests_per_minute: float | None = None

    @classmethod
    def from_env(cls, module: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> "ServiceConfig":
        service = _load_module_yaml(module, config_dir).get("service") or {}

        url = service.get("url")
        if not url:
            base_url = os.environ.get("SERVICE_BASE_URL")
            path = _first_endpoint_path(module, config_dir) if base_url else None
            if base_url and path:
                url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        if not url:
            raise KeyError(
                f"no service endpoint configured for module {module!r}; set SERVICE_BASE_URL in .env "
                f"(the module's path comes from its `endpoints:` entry), or set a full `service.url` "
                f"override in configs/modules/{module}.yaml (see .env.example)"
            )

        raw_token = service.get("token")
        env_token_match = re.fullmatch(r"\{env:([A-Za-z0-9_]+)\}", raw_token) if raw_token else None
        if env_token_match:
            token = os.environ.get(env_token_match.group(1)) or None
        else:
            token = raw_token or os.environ.get("DEFAULT_SERVICE_TOKEN") or None
        verify_ssl = bool(service.get("verify_ssl", True))
        timeout = float(service.get("timeout", 60))
        max_retries = int(service.get("max_retries", 3))
        retry_backoff_seconds = float(service.get("retry_backoff_seconds", 5.0))
        max_requests_per_minute = service.get("max_requests_per_minute")
        return cls(
            module=module,
            url=url,
            token=token,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_requests_per_minute=(
                float(max_requests_per_minute) if max_requests_per_minute is not None else None
            ),
        )


class _RateLimiter:
    """Paces calls to a fixed minimum interval apart, shared across every
    thread that acquires it. `acquire()` blocks the calling thread until
    it's safe to proceed, then reserves the next slot before releasing the
    lock -- so N concurrent callers naturally serialize into evenly spaced
    requests instead of bursting all at once."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed_at = now + self._min_interval


_rate_limiters: dict[str, _RateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def _rate_limiter_for(config: ServiceConfig) -> _RateLimiter | None:
    """One _RateLimiter per distinct service URL, created on first use and
    reused for the rest of the process -- so it stays in effect across every
    batch of a run (generate_batch_concurrent builds a fresh ResponseGenerator
    per batch, but the limiter itself outlives all of them), not just within
    one batch."""
    if not config.max_requests_per_minute:
        return None
    with _rate_limiters_lock:
        limiter = _rate_limiters.get(config.url)
        if limiter is None:
            limiter = _RateLimiter(60.0 / config.max_requests_per_minute)
            _rate_limiters[config.url] = limiter
        return limiter


def _retry_after_seconds(resp: requests.Response, default: float) -> float:
    """Reads how long to wait before retrying a 429 from the response's
    Retry-After header (RFC 9110) or, failing that, the RateLimit-Reset
    header (IETF ratelimit-headers draft, what this project's real
    deployments send) -- falling back to `default` if neither is present or
    parseable."""
    for header in ("Retry-After", "RateLimit-Reset"):
        value = resp.headers.get(header)
        if value is None:
            continue
        try:
            return max(float(value), 0.5)
        except ValueError:
            continue
    return default


class ResponseGenerator:
    """POSTs each EvalRow's `input` to its module's service endpoint and
    returns a new EvalRow with `generated` filled in from the response.

    The default request/response shape matches the offline_chatbot/local_llm
    contract observed for the Gemma general_knowledge endpoint: a
    question/module/session_id/... JSON body in, an "answer"-shaped JSON body
    out. A module whose service uses a different contract can subclass this
    and override `build_payload` / `parse_response`.
    """

    #: response keys tried, in order, when looking for the model's answer text
    ANSWER_KEYS = ("answer", "response", "output", "generated", "text")

    def __init__(self, config: ServiceConfig, contract: ServiceContract | None = None) -> None:
        self.config = config
        self.contract = contract

    def build_payload(self, row: EvalRow) -> dict[str, Any]:
        if self.contract and self.contract.request_template is not None:
            payload = _render_value(self.contract.request_template, row)
        else:
            offline_gpt_id = self.contract.offline_gpt_id if self.contract else ""
            payload = {
                "question": row.input,
                "module": "",
                "session_id": str(uuid.uuid4()),
                "offline_gpt_id": offline_gpt_id,
                "regenerate": False,
                "turn_id": str(uuid.uuid4()),
                "turn_index": 0,
                "generation_id": str(uuid.uuid4()),
            }

        # Validate against this module's typed contract (request_schemas.py),
        # when one has been confirmed for real -- catches a typo'd field name
        # or an enum value outside what's actually been observed on the wire
        # (e.g. a bad request_template edit) before it becomes a confusing
        # 4xx from the real service.
        schema_cls = REQUEST_SCHEMAS.get(row.module)
        if schema_cls is not None:
            try:
                schema_cls.from_payload(payload)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"row {row.row_id!r}: built request payload for module {row.module!r} doesn't "
                    f"match its request_schemas.{schema_cls.__name__} contract: {exc}"
                ) from exc

        return payload

    def parse_response(self, data: dict[str, Any]) -> str:
        if self.contract and self.contract.response_path:
            return _extract_path(data, self.contract.response_path)

        for key in self.ANSWER_KEYS:
            value = data.get(key)
            if isinstance(value, str):
                return value
        raise ValueError(f"could not find an answer field ({self.ANSWER_KEYS}) in service response: {data!r}")

    #: retried like a 429 (with backoff) instead of failing the row immediately --
    #: 5xx from this kind of backend is typically a transient overload/timeout on
    #: the model-serving side, not a permanent problem with the request.
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def generate_one(self, row: EvalRow) -> EvalRow:
        headers = {"content-type": "application/json"}
        if self.config.token:
            headers["authorization"] = f"Bearer {self.config.token}"

        payload = self.build_payload(row)
        rate_limiter = _rate_limiter_for(self.config)
        attempt = 0
        while True:
            if rate_limiter:
                rate_limiter.acquire()  # paces every attempt, not just the first
            try:
                resp = requests.post(
                    self.config.url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.config.max_retries:
                    raise requests.ConnectionError(
                        f"row {row.row_id!r}: {exc} -- sent payload: {json.dumps(payload)[:500]!r} "
                        f"(gave up after {attempt + 1} attempt(s))"
                    ) from exc
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
                attempt += 1
                continue

            if resp.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.config.max_retries:
                wait = (
                    _retry_after_seconds(resp, self.config.retry_backoff_seconds)
                    if resp.status_code == 429
                    else self.config.retry_backoff_seconds * (attempt + 1)
                )
                time.sleep(wait)
                attempt += 1
                continue
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                # The bare HTTPError swallows exactly what's needed to diagnose a
                # 4xx (e.g. 422 Unprocessable Entity) or a 5xx that outlasted
                # retries: which row, what was sent, and the server's own message
                # (e.g. `{"message": "timeout"}` -- a backend-side generation
                # timeout, not a malformed request).
                raise requests.HTTPError(
                    f"row {row.row_id!r}: {exc} -- sent payload: {json.dumps(payload)[:500]!r} "
                    f"-- response body: {resp.text[:1000]!r} (after {attempt + 1} attempt(s))",
                    response=resp,
                ) from exc
            break

        if self.contract and self.contract.response_stream_key:
            generated = _concat_json_stream(resp.text, self.contract.response_stream_key)
        else:
            generated = self.parse_response(resp.json())

        return EvalRow(
            row_id=row.row_id,
            module=row.module,
            input=row.input,
            golden=row.golden,
            generated=generated,
            context=row.context,
        )

    def generate_dataset(self, rows: list[EvalRow]) -> list[EvalRow]:
        return [self.generate_one(row) for row in rows]


DEFAULT_OUTPUT_DIR = "output"


def fill_dataset(
    path: str | Path,
    out_path: str | Path | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> list[EvalRow]:
    """Loads a golden dataset, calls each row's module service (one
    ServiceConfig + ServiceContract per module, built once and reused) to
    fill `generated`, and writes the result out as JSONL ready for
    `evalsuite.cli run`.

    Defaults to writing `output/<same filename>` -- golden datasets under
    `data/` stay input+golden-only so a rerun is always against the same
    golden set; pass `out_path` to write somewhere else instead (in place
    included, if that's genuinely what's wanted).
    """
    from .env import load_env

    load_env()

    rows = load_dataset(path)
    configs: dict[str, ServiceConfig] = {}
    contracts: dict[str, ServiceContract] = {}
    out_rows: list[EvalRow] = []
    for row in rows:
        if row.module not in configs:
            configs[row.module] = ServiceConfig.from_env(row.module, config_dir)
            contracts[row.module] = ServiceContract.from_module_config(row.module, config_dir)
        generator = ResponseGenerator(configs[row.module], contracts[row.module])
        out_rows.append(generator.generate_one(row))

    destination = Path(out_path) if out_path else Path(DEFAULT_OUTPUT_DIR) / Path(path).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    return out_rows
