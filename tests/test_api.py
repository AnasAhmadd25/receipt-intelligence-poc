import io
import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 4)).save(output, "PNG")
    return output.getvalue()


def test_api_key_and_upload_contract(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")
    class Provider:
        async def extract(self, *args, **kwargs):
            return {"receipt": {"total": "2.50"}, "metadata": {"provider": "gemini"}}
    with TestClient(app) as client, patch("app.main.get_provider", return_value=Provider()):
        assert client.get("/health").status_code == 200
        assert client.post("/v1/extract", files={"file": ("a.png", image_bytes(), "image/png")}).status_code == 401
        result = client.post("/v1/extract", headers={"X-API-Key": "test-api-key"}, data={"provider": "gemini", "merchant_id": "shop1"}, files={"file": ("a.png", image_bytes(), "image/png")})
        assert result.status_code == 200
        assert result.json()["receipt"]["total"] == "2.50"


def test_batch_limits_and_unknown_provider(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(app) as client:
        assert client.post("/v1/batch", json={"items": [{}] * 21}).status_code == 422
        result = client.post("/v1/extract", data={"provider": "unknown"}, files={"file": ("a.png", image_bytes(), "image/png")})
        assert result.status_code == 422


def test_public_deployment_requires_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    import pytest
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(app):
            pass


def test_uploaded_batch_preserves_order_and_bounds_inference(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    active = 0
    peak = 0
    class Provider:
        async def extract(self, *args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"receipt": {}, "metadata": {}}
    with TestClient(app) as client, patch("app.main.get_provider", return_value=Provider()):
        result = client.post("/v1/batch", data={"provider": "gemini", "concurrency": "2"}, files=[("files", (f"{i}.png", image_bytes(), "image/png")) for i in range(5)])
    assert result.status_code == 200
    assert [item["index"] for item in result.json()["results"]] == list(range(5))
    assert peak == 2
