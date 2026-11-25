"""Shared strict parsing, usage bookkeeping, and provider errors."""
import json
import re
from typing import Protocol

from app.schema import Receipt
from app.validation import validate_receipt


class ProviderError(RuntimeError):
    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class Provider(Protocol):
    async def extract(self, image_bytes: bytes, mime_type: str, *, ocr: dict | None = None, merchant_id: str | None = None) -> dict: ...


def parse_receipt(raw: str) -> Receipt:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    return validate_receipt(Receipt.model_validate(json.loads(text)))


def metadata(provider: str, model: str, attempts: list[dict], *, json_valid: bool = False, repairs: int = 0) -> dict:
    return {
        "provider": provider,
        "model": model,
        "attempts": attempts,
        "json_valid": json_valid,
        "repairs": repairs,
        "input_tokens": sum(a.get("input_tokens", 0) or 0 for a in attempts),
        "output_tokens": sum(a.get("output_tokens", 0) or 0 for a in attempts),
        "thinking_tokens": sum(a.get("thinking_tokens", 0) or 0 for a in attempts),
        "usage_complete": all(a.get("usage_reported", False) for a in attempts) if attempts else False,
        "api_latency_seconds": sum(a.get("latency_seconds", 0) for a in attempts),
        "latency_seconds": sum(a.get("latency_seconds", 0) for a in attempts),
    }
