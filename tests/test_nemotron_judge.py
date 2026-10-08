import json
from unittest.mock import MagicMock, patch

import pytest

from evalsuite.judge_provider import NemotronJudgeProvider, _parse_json_object

SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "rationale": {"type": "string"}},
    "required": ["score", "rationale"],
}


def _mock_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_nemotron_judge_parses_tool_call(monkeypatch):
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://example.invalid:8000")
    provider = NemotronJudgeProvider()

    api_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_evaluation",
                                "arguments": json.dumps({"score": 9.0, "rationale": "close match"}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    with patch("evalsuite.judge_provider.requests.post", return_value=_mock_response(api_response)) as mock_post:
        result = provider.judge("some prompt", response_schema=SCHEMA)

    assert result == {"score": 9.0, "rationale": "close match"}
    call_url = mock_post.call_args.args[0]
    assert call_url == "http://example.invalid:8000/v1/chat/completions"


def test_nemotron_judge_falls_back_to_json_in_content(monkeypatch):
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://example.invalid:8000")
    provider = NemotronJudgeProvider()

    api_response = {
        "choices": [
            {"message": {"content": 'Here is my evaluation: {"score": 7.5, "rationale": "mostly right"} done.'}}
        ]
    }

    with patch("evalsuite.judge_provider.requests.post", return_value=_mock_response(api_response)):
        result = provider.judge("some prompt", response_schema=SCHEMA)

    assert result == {"score": 7.5, "rationale": "mostly right"}


def test_nemotron_judge_sends_bearer_token_when_configured(monkeypatch):
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://example.invalid:8000")
    monkeypatch.setenv("NEMOTRON_API_KEY", "secret-key")
    provider = NemotronJudgeProvider()

    api_response = {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]}
    with patch("evalsuite.judge_provider.requests.post", return_value=_mock_response(api_response)) as mock_post:
        provider.judge("prompt", response_schema=SCHEMA)

    assert mock_post.call_args.kwargs["headers"]["authorization"] == "Bearer secret-key"


def test_nemotron_judge_requires_base_url(monkeypatch):
    monkeypatch.delenv("NEMOTRON_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        NemotronJudgeProvider()


def test_nemotron_registered_in_factory():
    from evalsuite.judge_provider import build_judge_provider

    provider = build_judge_provider("nemotron", base_url="http://example.invalid:8000")
    assert isinstance(provider, NemotronJudgeProvider)


def test_nemotron_judge_use_tools_false_skips_tool_choice_and_appends_schema_hint(monkeypatch):
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://example.invalid:8000")
    provider = NemotronJudgeProvider(use_tools=False)

    api_response = {"choices": [{"message": {"content": '{"score": 8, "rationale": "good"}'}}]}
    with patch("evalsuite.judge_provider.requests.post", return_value=_mock_response(api_response)) as mock_post:
        result = provider.judge("some prompt", response_schema=SCHEMA)

    assert result == {"score": 8, "rationale": "good"}
    payload = mock_post.call_args.kwargs["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "some prompt" in payload["messages"][0]["content"]
    assert json.dumps(SCHEMA) in payload["messages"][0]["content"]


def test_nemotron_judge_use_tools_false_reads_from_env(monkeypatch):
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://example.invalid:8000")
    monkeypatch.setenv("NEMOTRON_USE_TOOLS", "false")
    provider = NemotronJudgeProvider()

    api_response = {"choices": [{"message": {"content": '{"score": 8, "rationale": "good"}'}}]}
    with patch("evalsuite.judge_provider.requests.post", return_value=_mock_response(api_response)) as mock_post:
        provider.judge("some prompt", response_schema=SCHEMA)

    assert "tools" not in mock_post.call_args.kwargs["json"]


def test_parse_json_object_strips_reasoning_model_think_block():
    text = (
        "<think>\nLet me reason about this. Maybe {\"not\": \"it\"} is a distractor.\n</think>\n\n"
        '{"score": 6, "rationale": "reasonable"}'
    )
    assert _parse_json_object(text) == {"score": 6, "rationale": "reasonable"}


def test_parse_json_object_raises_when_no_object_present():
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _parse_json_object("<think>just thinking, no answer</think>\nsorry, no JSON here")
