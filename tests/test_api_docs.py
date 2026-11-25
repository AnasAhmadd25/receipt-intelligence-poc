"""Shared API schemas, authentication and image-specific capability boundaries."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


def png():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4)).save(stream, "PNG")
    return stream.getvalue()


def test_openapi_exposes_upload_json_responses_and_authorization(monkeypatch):
    monkeypatch.delenv("ALLOWED_PROVIDERS", raising=False)
    with TestClient(main.app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        schema = client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["APIKey"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    for path in ("/v1/extract", "/v1/batch"):
        operation = schema["paths"][path]["post"]
        assert {"APIKey": []} in operation["security"]
        assert {"BearerAuth": []} in operation["security"]
        content = operation["requestBody"]["content"]
        assert set(content) == {"application/json", "multipart/form-data"}
        assert "properties" in content["multipart/form-data"]["schema"]
        assert "schema" in operation["responses"]["200"]["content"]["application/json"]
        for code in ("401", "413", "422", "502", "504"):
            assert operation["responses"][code]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert "security" not in schema["paths"]["/health"]["get"]
    assert "security" not in schema["paths"]["/ready"]["get"]
    receipt = schema["components"]["schemas"]["Receipt"]
    assert "raw_text" in receipt["properties"]
    total = receipt["properties"]["total"]
    assert {part.get("type") for part in total["anyOf"]} == {"string", "null"}


def test_openapi_refs_resolve_and_provider_enum_matches_image(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("ALLOWED_PROVIDERS", "gemini")
    with TestClient(main.app) as client:
        schema = client.get("/openapi.json").json()
        health = client.get("/health").json()
    assert health["default_provider"] == "gemini"
    assert health["allowed_providers"] == ["gemini"]
    properties = schema["components"]["schemas"]["ExtractionInput"]["properties"]
    assert properties["provider"]["enum"] == ["gemini"]
    assert properties["provider"]["default"] == "gemini"
    upload = schema["paths"]["/v1/extract"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert upload["properties"]["provider"]["enum"] == ["gemini"]
    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = schema
                assert node["$ref"].startswith("#/")
                for key in node["$ref"][2:].split("/"):
                    target = target[key]
            for value in node.values():
                resolve(value)
        elif isinstance(node, list):
            for value in node:
                resolve(value)
    resolve(schema)


def test_disabled_provider_never_downloads_or_constructs_provider(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("ALLOWED_PROVIDERS", "ocr_nvidia")
    with TestClient(main.app) as client, patch("app.main.get_provider") as factory, patch("app.main.fetch_image_url", new_callable=AsyncMock) as download:
        response = client.post("/v1/extract", json={"provider": "gemini", "image_url": "https://example.org/x.png"})
        assert response.status_code == 422
        factory.assert_not_called()
        download.assert_not_called()


@pytest.mark.parametrize("value", ["", "gemini", "ocr_nvidia,unknown"])
def test_invalid_provider_configuration_fails_startup(monkeypatch, value):
    monkeypatch.setenv("ALLOWED_PROVIDERS", value)
    with pytest.raises(RuntimeError, match="ALLOWED_PROVIDERS"):
        with TestClient(main.app):
            pass


def test_bearer_auth_and_safe_batch_partial_failure(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("ALLOWED_PROVIDERS", "ocr_nvidia")
    provider = MagicMock()
    provider.extract = AsyncMock(return_value={"receipt": {"total": "4.20"}, "metadata": {"provider": "ocr_nvidia"}})
    with TestClient(main.app) as client, patch("app.main.get_provider", return_value=provider), patch("app.main.fetch_image_url", AsyncMock(return_value=png())):
        invalid = client.post("/v1/batch", headers={"Authorization": "Bearer wrong"}, json={"items": []})
        assert invalid.status_code == 401
        result = client.post("/v1/batch", headers={"Authorization": "Bearer test-key"}, json={"items": [
            {"provider": "ocr_nvidia", "image_url": "https://example.org/a.png"},
            {"provider": "gemini", "image_url": "https://example.org/b.png"},
        ]})
    assert result.status_code == 200
    items = result.json()["results"]
    assert [item["index"] for item in items] == [0, 1]
    assert items[0]["result"]["receipt"]["total"] == "4.20"
    assert items[1]["status_code"] == 422
    provider.extract.assert_awaited_once()


def test_request_size_limit_remains_before_body_parsing(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(main.app) as client:
        result = client.post("/v1/extract", headers={"Content-Length": str(main.MAX_REQUEST_BYTES + 1)}, content=b"x")
    assert result.status_code == 413


def test_selfhosted_warmup_and_readiness_use_local_state(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_PROVIDER", "selfhosted")
    monkeypatch.setenv("ALLOWED_PROVIDERS", "selfhosted")
    monkeypatch.setenv("SELFHOSTED_WARMUP", "true")
    provider = MagicMock()
    provider.readiness.return_value = {"ready": True, "provider": "selfhosted", "gpu_name": "test-gpu"}
    with patch("app.main.get_provider", return_value=provider), TestClient(main.app) as client:
        provider.warmup.assert_called_once_with()
        result = client.get("/ready")
        assert result.status_code == 200
        assert result.json()["gpu_name"] == "test-gpu"
        provider.readiness.return_value = {"ready": False, "provider": "selfhosted"}
        assert client.get("/ready").status_code == 503


def test_hosted_readiness_does_not_call_paid_provider(monkeypatch):
    monkeypatch.delenv("ALLOWED_PROVIDERS", raising=False)
    with patch("app.main.get_provider") as factory, TestClient(main.app) as client:
        result = client.get("/ready")
    assert result.status_code == 200
    assert "configuration only" in result.json()["scope"]
    factory.assert_not_called()
