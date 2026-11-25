import json
from types import SimpleNamespace
from unittest.mock import patch

from app.providers.cloud import GeminiTextProvider


async def test_ocr_text_stage_sends_no_image_and_preserves_observed_transcript(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    calls = []
    class Models:
        async def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text='{"schema_version":"1.0","total":"12.50","raw_text":"shortened"}',
                usage_metadata=SimpleNamespace(model_dump=lambda **_: {"prompt_token_count": 30, "candidates_token_count": 10, "thoughts_token_count": 0}))
    class Aio:
        models = Models()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    observed = "الحليب 12.50\nTOTAL 12.50"
    with patch("google.genai.Client", return_value=SimpleNamespace(aio=Aio())):
        result = await GeminiTextProvider().extract(b"private-image-must-not-be-sent", "image/jpeg",
            ocr={"text": observed, "lines": [], "engine": "test_ocr"})
    assert len(calls) == 1
    assert all(part.inline_data is None and part.file_data is None for part in calls[0]["contents"])
    assert json.loads(calls[0]["contents"][0].text.split("Observed OCR data:\n", 1)[1])["ocr_text"] == observed
    assert result["receipt"]["raw_text"] == observed
    assert result["metadata"]["input_mode"] == "ocr_text_only"
    assert result["metadata"]["model"] == "gemini-3.1-flash-lite"
