"""JudgeProvider abstraction: the swappable LLM that actually judges.

Scorers never talk to a vendor SDK directly -- they call .judge() on
whatever JudgeProvider was wired in, and get back a structured response.
This is what makes "swap Claude for GPT-4 or a self-hosted model" a config
change instead of a code change (see the design doc's HLD/LLD and the
Self-hosted-vs-API-judge note).
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import requests


class JudgeProvider(ABC):
    """One judge model behind one interface."""

    name: str = "base"
    model_name: str = "unknown"

    @abstractmethod
    def judge(self, prompt: str, *, response_schema: dict[str, Any]) -> dict[str, Any]:
        """Send `prompt` to the judge LLM and return a dict matching response_schema.

        Implementations should use structured output / tool-calling rather
        than free-text parsing wherever the underlying API supports it.
        """
        raise NotImplementedError


class MockJudgeProvider(JudgeProvider):
    """Deterministic, offline judge for tests and local development.

    Scores token overlap between the golden and generated answer text found
    in the prompt. It is intentionally simple and NOT a substitute for a
    real judge -- it exists so the rest of the pipeline is runnable and
    testable without any API key or network access.
    """

    name = "mock"
    model_name = "mock-overlap-v1"

    def judge(self, prompt: str, *, response_schema: dict[str, Any]) -> dict[str, Any]:
        golden = _extract_field(prompt, "Golden answer")
        generated = _extract_field(prompt, "Generated answer")
        score = _token_overlap(golden, generated)
        return {
            "score": round(score * 10, 1),  # 0-10 scale, like the real judge prompt asks for
            "rationale": f"[mock judge] token overlap between golden and generated answer = {score:.2f}",
        }


class ClaudeJudgeProvider(JudgeProvider):
    """Judge backed by the Anthropic API. Requires `anthropic` and an API key."""

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "ClaudeJudgeProvider requires the 'anthropic' package: pip install anthropic"
            ) from exc
        self.model_name = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def judge(self, prompt: str, *, response_schema: dict[str, Any]) -> dict[str, Any]:
        tool = {
            "name": "submit_evaluation",
            "description": "Submit the evaluation score and rationale.",
            "input_schema": response_schema,
        }
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_evaluation"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input  # type: ignore[return-value]
        raise RuntimeError("judge did not return a tool_use block")


class OpenAIJudgeProvider(JudgeProvider):
    """Judge backed by the OpenAI API. Requires `openai` and an API key."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIJudgeProvider requires the 'openai' package: pip install openai"
            ) from exc
        self.model_name = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def judge(self, prompt: str, *, response_schema: dict[str, Any]) -> dict[str, Any]:
        tool = {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "description": "Submit the evaluation score and rationale.",
                "parameters": response_schema,
            },
        }
        resp = self._client.chat.completions.create(
            model=self.model_name,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
            messages=[{"role": "user", "content": prompt}],
        )
        call = resp.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)



class NemotronJudgeProvider(JudgeProvider):
    """Judge backed by a self-hosted Nemotron model, called directly over
    HTTP with `requests` -- no vendor SDK, since it's an internal deployment
    rather than a hosted API. Assumes an OpenAI-compatible chat-completions
    route (the vLLM / NIM default); set NEMOTRON_CHAT_PATH if yours differs.

    Config comes from .env (see .env.example): NEMOTRON_BASE_URL (required),
    NEMOTRON_CHAT_PATH, NEMOTRON_MODEL, NEMOTRON_API_KEY (optional -- omit if
    the deployment doesn't require auth), NEMOTRON_VERIFY_SSL, NEMOTRON_USE_TOOLS
    (optional, default "true"; set "false" for a vLLM deployment started
    without --tool-call-parser, which 400s on any tool_choice -- forces the
    JSON-in-plain-content path instead, via an explicit schema instruction
    appended to the prompt). A `<think>...</think>` reasoning block in the
    response, if present, is stripped before that JSON is parsed.
    """

    name = "nemotron"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        chat_path: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        verify_ssl: bool | None = None,
        use_tools: bool | None = None,
    ) -> None:
        self.model_name = model or os.environ.get("NEMOTRON_MODEL", "nemotron")
        self._base_url = (base_url or os.environ.get("NEMOTRON_BASE_URL", "")).rstrip("/")
        if not self._base_url:
            raise ValueError(
                "NEMOTRON_BASE_URL is not set; add it to .env (see .env.example) "
                "or pass base_url= explicitly"
            )
        self._chat_path = chat_path or os.environ.get("NEMOTRON_CHAT_PATH", "/v1/chat/completions")
        self._api_key = api_key if api_key is not None else (os.environ.get("NEMOTRON_API_KEY") or None)
        self._timeout = timeout
        if verify_ssl is not None:
            self._verify = verify_ssl
        else:
            env_verify = os.environ.get("NEMOTRON_VERIFY_SSL")
            self._verify = env_verify.strip().lower() not in ("false", "0", "no") if env_verify else True
        if use_tools is not None:
            self._use_tools = use_tools
        else:
            env_use_tools = os.environ.get("NEMOTRON_USE_TOOLS")
            self._use_tools = env_use_tools.strip().lower() not in ("false", "0", "no") if env_use_tools else True

    def judge(self, prompt: str, *, response_schema: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        if self._use_tools:
            tool = {
                "type": "function",
                "function": {
                    "name": "submit_evaluation",
                    "description": "Submit the evaluation score and rationale.",
                    "parameters": response_schema,
                },
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "submit_evaluation"}},
                "temperature": 0,
            }
        else:
            content = (
                f"{prompt}\n\nRespond with ONLY a single JSON object (no markdown fences, no text "
                f"before or after it) matching this schema: {json.dumps(response_schema)}"
            )
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
            }

        resp = requests.post(
            f"{self._base_url}{self._chat_path}",
            json=payload,
            headers=headers,
            timeout=self._timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return json.loads(tool_calls[0]["function"]["arguments"])

        # Some OpenAI-compatible servers ignore tool_choice / don't support
        # tool calling at all -- fall back to JSON embedded in plain content.
        return _parse_json_object(message.get("content") or "")


def build_judge_provider(name: str, **kwargs: Any) -> JudgeProvider:
    """Factory: turns a config string ("mock" | "claude" | "openai") into a
    JudgeProvider. This is the Adapter/Factory pattern from the design
    doc's HLD/LLD -- swapping the judge model is a config change here,
    never a code change anywhere else in the pipeline.
    """
    providers = {
        "mock": MockJudgeProvider,
        "claude": ClaudeJudgeProvider,
        "openai": OpenAIJudgeProvider,
        "nemotron": NemotronJudgeProvider,
    }
    try:
        cls = providers[name]
    except KeyError as exc:
        raise ValueError(f"unknown judge provider {name!r}; choose from {sorted(providers)}") from exc
    return cls(**kwargs)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Pulls the first {...} JSON object out of free-form text -- used when a
    judge endpoint returns its answer as plain content instead of a proper
    tool call. Strips a `<think>...</think>` reasoning block first (present
    on reasoning models), then scans for a `{` and lets json.JSONDecoder
    find its true matching end -- unlike a greedy `\\{.*\\}` regex, this
    doesn't get confused by stray braces earlier in the text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = start + 1
    raise ValueError(f"nemotron judge response did not contain a JSON object: {text!r}")


def _extract_field(prompt: str, label: str) -> str:
    marker = f"{label}:\n"
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = prompt.find("\n\n", start)
    return prompt[start: end if end != -1 else None].strip()


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _token_overlap(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens:
        return 1.0 if not b_tokens else 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)
