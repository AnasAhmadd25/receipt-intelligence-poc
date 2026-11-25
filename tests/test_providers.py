import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.providers.base import ProviderError, parse_receipt
from app.providers.cloud import GeminiProvider, OllamaProvider, NvidiaProvider, ModalProvider


def test_fenced_arabic_json_and_decimal_validation():
    parsed = parse_receipt('```json\n{"merchant_name":"متجر","total":"١٢٫٥٠"}\n```')
    assert parsed.total == 12.5
    assert parsed.merchant_name == "متجر"
    with pytest.raises(ValueError):
        parse_receipt('{"total":"NaN"}')


async def test_ollama_exact_one_repair_and_usage(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-secret")
    calls = []
    async def post(client, url, **kwargs):
        calls.append(kwargs["json"])
        data = {"message": {"content": "broken" if len(calls) == 1 else '{"total":"12.50"}'}, "prompt_eval_count": 11, "eval_count": 7}
        return httpx.Response(200, json=data, request=httpx.Request("POST", url))
    with patch.object(httpx.AsyncClient, "post", post):
        result = await OllamaProvider().extract(b"", "image/png", ocr={"text": "total 12.50", "lines": []})
    assert len(calls) == 2
    assert "format" not in calls[0]
    assert result["metadata"]["json_valid"] is False
    assert result["metadata"]["repairs"] == 1
    assert result["metadata"]["input_tokens"] == 22
    assert result["metadata"]["output_tokens"] == 14
    assert result["receipt"]["raw_text"] == "total 12.50"


async def test_failed_repair_retains_attempts_and_usage(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    monkeypatch.setenv("NVIDIA_MODEL", "configured-model")
    async def post(*args, **kwargs):
        return httpx.Response(200, json={"choices": [{"message": {"content": "invalid"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 4}}, request=httpx.Request("POST", "https://test.example"))
    with patch.object(httpx.AsyncClient, "post", post), pytest.raises(ProviderError) as failure:
        await NvidiaProvider().extract(b"", "image/png", ocr={"text": "x", "lines": []})
    assert len(failure.value.metadata["attempts"]) == 2
    assert failure.value.metadata["output_tokens"] == 8


def test_gemini_thinking_count_not_added_twice():
    usage = GeminiProvider.usage({"prompt_token_count": 20, "candidates_token_count": 10, "thoughts_token_count": 4, "total_token_count": 34})
    assert usage == {"input_tokens": 20, "output_tokens": 14, "thinking_tokens": 4, "candidate_tokens": 10, "usage_reported": True}


async def test_gemini_native_schema_repair_and_token_details(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    calls = []
    class FakeModels:
        async def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="invalid" if len(calls) == 1 else '{"total":"2.50"}', usage_metadata=SimpleNamespace(model_dump=lambda **_: {"prompt_token_count": 20, "candidates_token_count": 10, "thoughts_token_count": 4}))
    class FakeAio:
        models = FakeModels()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    def fake_client(**kwargs):
        assert kwargs["http_options"].retry_options.attempts == 1
        return SimpleNamespace(aio=FakeAio())
    with patch("google.genai.Client", fake_client):
        result = await GeminiProvider().extract(b"example", "image/png")
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["config"].response_json_schema["properties"]["total"]
    assert result["metadata"]["output_tokens"] == 28
    assert result["metadata"]["thinking_tokens"] == 8
    assert result["metadata"]["repairs"] == 1


async def test_transport_error_preserves_safe_metadata_without_secret(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-secret")
    async def broken_post(*args, **kwargs):
        raise httpx.ConnectError("https://example.com/?secret=test-secret")
    with patch.object(httpx.AsyncClient, "post", broken_post), pytest.raises(ProviderError) as error:
        await OllamaProvider().extract(b"", "image/png", ocr={"text": "x"})
    assert "test-secret" not in json.dumps(error.value.metadata)
    assert error.value.metadata["usage_complete"] is False
    assert error.value.metadata["attempts"][0]["error"] == "ConnectError"


async def test_modal_invalid_receipt_retains_server_resource_evidence(monkeypatch):
    monkeypatch.setenv("MODAL_ENDPOINT", "https://modal.example/extract")
    monkeypatch.setenv("MODAL_API_KEY", "test-secret")
    async def post(client, url, **kwargs):
        return httpx.Response(200, json={"receipt": None, "raw_output": "not json", "metadata": {"gpu_peak_memory_mb": 1200, "initialization_seconds": 12}}, request=httpx.Request("POST", url))
    with patch.object(httpx.AsyncClient, "post", post), pytest.raises(ProviderError) as failure:
        await ModalProvider().extract(b"", "image/png")
    assert failure.value.metadata["gpu_peak_memory_mb"] == 1200
    assert failure.value.metadata["raw_output"] == "not json"
    assert failure.value.metadata["json_valid"] is False


async def test_repair_identifies_invalid_monetary_tax_without_echoing_input(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    monkeypatch.setenv("NVIDIA_MODEL", "configured-model")
    calls = []
    async def post(client, url, **kwargs):
        calls.append(json.loads(json.dumps(kwargs["json"])))
        raw = '{"tax":"5%"}' if len(calls) == 1 else '{"tax":null}'
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}], "usage": {"prompt_tokens": 5, "completion_tokens": 4}}, request=httpx.Request("POST", url))
    with patch.object(httpx.AsyncClient, "post", post):
        result = await NvidiaProvider().extract(b"", "image/png", ocr={"text": "tax 5%"})
    repair_prompt = calls[1]["messages"][-1]["content"]
    assert "tax" in repair_prompt and "value_error" in repair_prompt
    assert "percentage" in repair_prompt and "null" in repair_prompt
    assert result["receipt"]["tax"] is None
    assert result["metadata"]["template_revision"]


async def test_gemini_repair_supplies_literal_version_diagnostics(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    calls = []
    class Models:
        async def generate_content(self, **kwargs):
            calls.append({**kwargs, "contents": list(kwargs["contents"])})
            raw = '{"schema_version":"1.0.0","total":"10"}' if len(calls) == 1 else '{"schema_version":"1.0","total":"10"}'
            return SimpleNamespace(text=raw, usage_metadata=SimpleNamespace(model_dump=lambda **_: {"prompt_token_count": 20, "candidates_token_count": 10, "thoughts_token_count": 0}))
    class Aio:
        models = Models()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    with patch("google.genai.Client", return_value=SimpleNamespace(aio=Aio())):
        result = await GeminiProvider(model="gemini-3.1-flash-lite").extract(b"example", "image/png")
    repair = calls[1]["contents"][-1].text
    assert "schema_version" in repair and "literal_error" in repair
    assert "1.0" in repair and "exactly" in repair
    assert len(calls) == 2 and result["receipt"]["schema_version"] == "1.0"
    assert result["metadata"]["repair_revision"]
