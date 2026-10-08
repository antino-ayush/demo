import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from evalsuite.generator import ResponseGenerator, ServiceConfig, ServiceContract, _retry_after_seconds, fill_dataset
from evalsuite.models import EvalRow


def _row(row_id="gk-1"):
    return EvalRow(
        row_id=row_id,
        module="general_knowledge",
        input="What is the capital of Australia?",
        golden="Canberra",
        generated="",
        context=None,
    )


def _mock_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _mock_stream_response(text):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.text = text
    return resp


def test_generate_one_fills_generated_from_service_response():
    config = ServiceConfig(
        module="general_knowledge",
        url="https://example.invalid/api/offline_chatbot/local_llm",
        token="tok123",
    )
    generator = ResponseGenerator(config)

    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"answer": "Canberra"})) as mock_post:
        result = generator.generate_one(_row())

    assert result.row_id == "gk-1"
    assert result.golden == "Canberra"
    assert result.generated == "Canberra"

    _, call_kwargs = mock_post.call_args
    assert call_kwargs["headers"]["authorization"] == "Bearer tok123"
    assert call_kwargs["json"]["question"] == "What is the capital of Australia?"
    assert call_kwargs["json"]["offline_gpt_id"] == ""
    assert call_kwargs["verify"] is True


def test_default_payload_uses_offline_gpt_id_from_contract():
    config = ServiceConfig(module="general_knowledge", url="https://example.invalid/x")
    contract = ServiceContract(module="general_knowledge", offline_gpt_id="abc123")
    generator = ResponseGenerator(config, contract)

    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"answer": "Canberra"})) as mock_post:
        generator.generate_one(_row())

    _, call_kwargs = mock_post.call_args
    assert call_kwargs["json"]["offline_gpt_id"] == "abc123"


def test_generate_one_raises_if_no_known_answer_key():
    config = ServiceConfig(module="general_knowledge", url="https://example.invalid/x")
    generator = ResponseGenerator(config)

    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"unexpected": "field"})):
        try:
            generator.generate_one(_row())
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unrecognized response shape")


def test_service_config_from_env_falls_back_to_default_token(monkeypatch, tmp_path):
    (tmp_path / "general_knowledge.yaml").write_text(
        "module: general_knowledge\nendpoints: [/x]\ncriteria: [c]\nweights: {c: 1.0}\n"
    )
    monkeypatch.setenv("SERVICE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("DEFAULT_SERVICE_TOKEN", "shared-token")

    config = ServiceConfig.from_env("general_knowledge", tmp_path)

    assert config.url == "https://example.invalid/x"
    assert config.token == "shared-token"
    assert config.verify_ssl is True


def test_service_config_from_env_yaml_overrides_win_over_shared_defaults(monkeypatch, tmp_path):
    (tmp_path / "rag_qa.yaml").write_text(
        "module: rag_qa\nendpoints: [/rag]\ncriteria: [c]\nweights: {c: 1.0}\n"
        "service:\n  token: rag-only-token\n  verify_ssl: false\n  timeout: 15\n"
    )
    monkeypatch.setenv("SERVICE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("DEFAULT_SERVICE_TOKEN", "shared-token")

    config = ServiceConfig.from_env("rag_qa", tmp_path)

    assert config.token == "rag-only-token"
    assert config.verify_ssl is False
    assert config.timeout == 15


def test_service_config_from_env_missing_url_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVICE_BASE_URL", raising=False)
    try:
        ServiceConfig.from_env("summarization", tmp_path)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError when neither SERVICE_BASE_URL nor a service.url override is set")


def test_service_config_from_env_resolves_via_base_url_and_module_endpoint(monkeypatch, tmp_path):
    (tmp_path / "gpts.yaml").write_text(
        "module: gpts\nendpoints: [/varuna]\ncriteria: [correctness]\nweights: {correctness: 1.0}\n"
    )
    monkeypatch.setenv("SERVICE_BASE_URL", "https://gateway.internal/")

    config = ServiceConfig.from_env("gpts", tmp_path)

    assert config.url == "https://gateway.internal/varuna"


def test_service_config_from_env_yaml_url_override_wins_over_base_url(monkeypatch, tmp_path):
    (tmp_path / "general_knowledge.yaml").write_text(
        "module: general_knowledge\nendpoints: [/api/offline_chatbot/local_llm]\n"
        "criteria: [correctness]\nweights: {correctness: 1.0}\n"
        "service:\n  url: https://gemma.other-host/api/offline_chatbot/local_llm\n"
    )
    monkeypatch.setenv("SERVICE_BASE_URL", "https://gateway.internal")

    config = ServiceConfig.from_env("general_knowledge", tmp_path)

    assert config.url == "https://gemma.other-host/api/offline_chatbot/local_llm"


def test_custom_contract_renders_request_template_and_parses_nested_response():
    config = ServiceConfig(module="gpts", url="https://example.invalid/varuna")
    contract = ServiceContract(
        module="gpts",
        request_template={"prompt": "{input}", "meta": {"row": "{row_id}"}},
        response_path="choices.0.message.content",
    )
    generator = ResponseGenerator(config, contract)

    response = {"choices": [{"message": {"content": "Canberra"}}]}
    with patch("evalsuite.generator.requests.post", return_value=_mock_response(response)) as mock_post:
        result = generator.generate_one(_row())

    assert result.generated == "Canberra"
    _, call_kwargs = mock_post.call_args
    assert call_kwargs["json"] == {"prompt": "What is the capital of Australia?", "meta": {"row": "gk-1"}}


def test_custom_contract_response_path_missing_key_raises():
    config = ServiceConfig(module="open", url="https://example.invalid/open")
    contract = ServiceContract(module="open", request_template={"q": "{input}"}, response_path="answer")
    generator = ResponseGenerator(config, contract)

    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"unexpected": "field"})):
        with pytest.raises(ValueError, match="response_path"):
            generator.generate_one(_row())


def test_service_contract_from_module_config_reads_service_block(tmp_path):
    (tmp_path / "gpts.yaml").write_text(
        "module: gpts\n"
        "criteria: [correctness]\n"
        "weights: {correctness: 1.0}\n"
        "service:\n"
        "  request_template:\n"
        "    prompt: '{input}'\n"
        "  response_path: answer\n"
        "  offline_gpt_id: abc123\n"
    )
    contract = ServiceContract.from_module_config("gpts", tmp_path)
    assert contract.request_template == {"prompt": "{input}"}
    assert contract.response_path == "answer"
    assert contract.offline_gpt_id == "abc123"


def test_service_contract_from_module_config_defaults_when_no_service_block(tmp_path):
    (tmp_path / "general_knowledge.yaml").write_text(
        "module: general_knowledge\ncriteria: [correctness]\nweights: {correctness: 1.0}\n"
    )
    contract = ServiceContract.from_module_config("general_knowledge", tmp_path)
    assert contract.request_template is None
    assert contract.response_path is None
    assert contract.offline_gpt_id == ""


def test_response_stream_key_concatenates_chunks_from_a_live_style_body():
    config = ServiceConfig(module="open_gpt", url="https://example.invalid/x")
    contract = ServiceContract(module="open_gpt", response_stream_key="data")
    generator = ResponseGenerator(config, contract)

    body = (
        '{"status": "streaming", "data": "Hel"}'
        '{"status": "streaming", "data": "lo"}'
        '{"status": "streaming", "data": "!"}'
        '{"status": "complete", "data": "", "turn_id": "x"}'
    )
    with patch("evalsuite.generator.requests.post", return_value=_mock_stream_response(body)):
        result = generator.generate_one(_row())

    assert result.generated == "Hello!"


def test_response_stream_key_ignores_chunks_without_the_key():
    config = ServiceConfig(module="open_gpt", url="https://example.invalid/x")
    contract = ServiceContract(module="open_gpt", response_stream_key="data")
    generator = ResponseGenerator(config, contract)

    body = '{"status": "heartbeat"}{"status": "streaming", "data": "ok"}'
    with patch("evalsuite.generator.requests.post", return_value=_mock_stream_response(body)):
        result = generator.generate_one(_row())

    assert result.generated == "ok"


def test_service_contract_from_module_config_reads_response_stream_key(tmp_path):
    (tmp_path / "open_gpt.yaml").write_text(
        "module: open_gpt\ncriteria: [correctness]\nweights: {correctness: 1.0}\n"
        "service:\n  response_stream_key: data\n"
    )
    contract = ServiceContract.from_module_config("open_gpt", tmp_path)
    assert contract.response_stream_key == "data"


def test_generate_one_retries_after_429_honoring_retry_after_header():
    config = ServiceConfig(module="open_gpt", url="https://example.invalid/x", max_retries=3)
    generator = ResponseGenerator(config)

    throttled = MagicMock()
    throttled.status_code = 429
    throttled.headers = {"Retry-After": "2"}

    ok = _mock_response({"answer": "Canberra"})
    ok.status_code = 200

    with patch("evalsuite.generator.requests.post", side_effect=[throttled, ok]) as mock_post, \
            patch("evalsuite.generator.time.sleep") as mock_sleep:
        result = generator.generate_one(_row())

    assert result.generated == "Canberra"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    throttled.raise_for_status.assert_not_called()


def test_generate_one_gives_up_after_max_retries_on_429():
    config = ServiceConfig(module="open_gpt", url="https://example.invalid/x", max_retries=2, retry_backoff_seconds=0)
    generator = ResponseGenerator(config)

    throttled = MagicMock()
    throttled.status_code = 429
    throttled.headers = {}
    throttled.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")

    with patch("evalsuite.generator.requests.post", return_value=throttled) as mock_post, \
            patch("evalsuite.generator.time.sleep") as mock_sleep:
        with pytest.raises(requests.exceptions.HTTPError):
            generator.generate_one(_row())

    assert mock_post.call_count == 3  # initial attempt + 2 retries
    assert mock_sleep.call_count == 2


def test_retry_after_seconds_prefers_retry_after_over_ratelimit_reset():
    resp = MagicMock()
    resp.headers = {"Retry-After": "2", "RateLimit-Reset": "50"}
    assert _retry_after_seconds(resp, default=5.0) == 2.0


def test_retry_after_seconds_falls_back_to_ratelimit_reset_header():
    resp = MagicMock()
    resp.headers = {"RateLimit-Reset": "42"}
    assert _retry_after_seconds(resp, default=5.0) == 42.0


def test_retry_after_seconds_falls_back_to_default_when_headers_missing():
    resp = MagicMock()
    resp.headers = {}
    assert _retry_after_seconds(resp, default=5.0) == 5.0


def test_fill_dataset_defaults_to_output_dir_and_leaves_source_untouched(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVICE_BASE_URL", "https://example.invalid")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "open_gpt.yaml").write_text(
        "module: open_gpt\nendpoints: [/x]\ncriteria: [correctness]\nweights: {correctness: 1.0}\n"
    )

    dataset_path = tmp_path / "data" / "open_gpt_dataset.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text(
        json.dumps(
            {"row_id": "r1", "module": "open_gpt", "input": "hi", "golden": "hi back", "generated": "", "context": None}
        )
        + "\n"
    )

    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"answer": "hi back"})):
        fill_dataset(dataset_path, config_dir=config_dir)

    output_path = tmp_path / "output" / "open_gpt_dataset.jsonl"
    assert output_path.exists()
    written = json.loads(output_path.read_text().splitlines()[0])
    assert written["generated"] == "hi back"

    source = json.loads(dataset_path.read_text().splitlines()[0])
    assert source["generated"] == ""


def test_fill_dataset_respects_explicit_out_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SERVICE_BASE_URL", "https://example.invalid")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "open_gpt.yaml").write_text(
        "module: open_gpt\nendpoints: [/x]\ncriteria: [correctness]\nweights: {correctness: 1.0}\n"
    )

    dataset_path = tmp_path / "open_gpt_dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {"row_id": "r1", "module": "open_gpt", "input": "hi", "golden": "hi back", "generated": "", "context": None}
        )
        + "\n"
    )

    custom_out = tmp_path / "somewhere" / "else.jsonl"
    with patch("evalsuite.generator.requests.post", return_value=_mock_response({"answer": "hi back"})):
        fill_dataset(dataset_path, custom_out, config_dir=config_dir)

    assert custom_out.exists()
    assert not (tmp_path / "output").exists()


def test_service_config_from_env_reads_retry_settings_from_yaml(monkeypatch, tmp_path):
    (tmp_path / "open_gpt.yaml").write_text(
        "module: open_gpt\nendpoints: [/x]\ncriteria: [c]\nweights: {c: 1.0}\n"
        "service:\n  max_retries: 5\n  retry_backoff_seconds: 10\n"
    )
    monkeypatch.setenv("SERVICE_BASE_URL", "https://example.invalid")

    config = ServiceConfig.from_env("open_gpt", tmp_path)

    assert config.max_retries == 5
    assert config.retry_backoff_seconds == 10
