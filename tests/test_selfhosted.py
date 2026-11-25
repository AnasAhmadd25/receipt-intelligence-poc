"""Offline checks for the local GPU protocol; these never download/load weights."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import sys
import threading
import time
from types import SimpleNamespace

from PIL import Image
import pytest

from app.providers.base import ProviderError
from app.providers import selfhosted as module


def image_bytes(size=(20, 30)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


class FakeRuntime:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.lock = threading.Lock()
        self.requests = 0
        self.details = {"device": "cuda", "gpu_name": "FAKE TEST GPU", "dtype": "torch.float16"}
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda: None, empty_cache=lambda: None,
            max_memory_allocated=lambda: 1234, max_memory_reserved=lambda: 2345))

    def generate(self, messages):
        self.calls.append(list(messages))
        raw = next(self.responses)
        if isinstance(raw, Exception):
            raise raw
        return raw, {"input_tokens": 11, "output_tokens": 7, "thinking_tokens": 0, "usage_reported": True}


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(module, "_runtime", None)
    monkeypatch.setattr(module, "_load_error", None)


def test_preprocess_caps_pixels_without_enlarging_small_images():
    picture, original = module._prepare_image(image_bytes((2000, 1000)))
    assert original == [2000, 1000]
    assert picture.width * picture.height <= 1_000_000
    assert list(module._prepare_image(image_bytes())[0].size) == [20, 30]
    with pytest.raises(ProviderError, match="valid receipt image"):
        module._prepare_image(b"invalid")


async def test_initial_valid_receipt_has_local_usage_and_revision(monkeypatch):
    runtime = FakeRuntime(['{"merchant_name":"متجر","total":"١٢٫٥٠","raw_text":"متجر ١٢٫٥٠"}'])
    monkeypatch.setattr(module, "_get_runtime", lambda: runtime)
    result = await module.SelfHostedProvider().extract(image_bytes(), "image/png", merchant_id="do not invent this", ocr={"text": "not visual evidence"})
    assert result["receipt"]["total"] == "12.50"
    assert result["receipt"]["raw_text"] == "متجر ١٢٫٥٠"
    value = result["metadata"]
    assert value["provider"] == "selfhosted" and value["hosted_inference"] is False
    assert value["revision"] == module.MODEL_REVISION
    assert value["json_valid"] and value["final_valid"] and value["repairs"] == 0
    assert value["input_tokens"] == 11 and value["output_tokens"] == 7
    assert value["gpu_peak_allocated_bytes"] == 1234
    assert len(runtime.calls) == 1
    prompt = runtime.calls[0][0]["content"][1]["text"]
    assert "raw_text" in prompt and "untrusted" in prompt
    assert "not visual evidence" not in prompt and "do not invent this" not in prompt


async def test_one_repair_retains_first_visual_transcript_and_both_attempts(monkeypatch):
    first = json.dumps({"schema_version": "1.0.0", "tax": "5%", "raw_text": "متجر TAX 5%"}, ensure_ascii=False)
    runtime = FakeRuntime([first, '{"tax":null,"total":"10.00","raw_text":"truncated"}'])
    monkeypatch.setattr(module, "_get_runtime", lambda: runtime)
    result = await module.SelfHostedProvider().extract(image_bytes(), "image/png")
    assert result["receipt"]["raw_text"] == "متجر TAX 5%"
    assert result["receipt"]["tax"] is None
    value = result["metadata"]
    assert value["repairs"] == 1 and value["json_valid"] is False and value["final_valid"] is True
    assert value["input_tokens"] == 22 and value["output_tokens"] == 14
    assert value["attempts"][0]["raw_response"] == first
    assert value["attempts"][1]["status"] == "ok"
    prompt = runtime.calls[1][-1]["content"]
    assert "schema_version" in prompt and "tax" in prompt
    assert "percentage" in prompt and "Preserve" in prompt and "Do not add facts" in prompt


async def test_failed_repair_is_bounded_and_retains_invalid_output(monkeypatch):
    runtime = FakeRuntime(["bad JSON", '{"total":"NaN"}', '{"total":"3"}'])
    monkeypatch.setattr(module, "_get_runtime", lambda: runtime)
    with pytest.raises(ProviderError, match="after one repair") as error:
        await module.SelfHostedProvider().extract(image_bytes(), "image/png")
    assert len(runtime.calls) == 2
    assert error.value.metadata["repairs"] == 1
    assert error.value.metadata["attempts"][0]["raw_response"] == "bad JSON"
    assert error.value.metadata["attempts"][1]["raw_response"] == '{"total":"NaN"}'


async def test_generation_error_never_calls_a_hosted_fallback_or_retries(monkeypatch):
    runtime = FakeRuntime([RuntimeError("sensitive model-cache path")])
    monkeypatch.setattr(module, "_get_runtime", lambda: runtime)
    with pytest.raises(ProviderError, match="inference failed") as error:
        await module.SelfHostedProvider().extract(image_bytes(), "image/png")
    assert len(runtime.calls) == 1
    assert error.value.metadata["attempts"][0]["error"] == "RuntimeError"
    assert "sensitive" not in json.dumps(error.value.metadata)


async def test_parallel_requests_serialize_model_generation(monkeypatch):
    runtime = FakeRuntime(['{"total":"1"}'] * 3)
    original = runtime.generate
    active = 0
    peak = 0

    def slow_generate(messages):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(.02)
        result = original(messages)
        active -= 1
        return result

    runtime.generate = slow_generate
    monkeypatch.setattr(module, "_get_runtime", lambda: runtime)
    await asyncio.gather(*(module.SelfHostedProvider().extract(image_bytes(), "image/png") for _ in range(3)))
    assert peak == 1 and runtime.requests == 3


def test_lazy_singleton_and_readiness_do_not_load_on_health(monkeypatch):
    loads = []
    value = FakeRuntime([])

    def load():
        loads.append(1)
        time.sleep(.02)
        return value

    monkeypatch.setattr(module, "_Runtime", load)
    assert module.readiness()["ready"] is False
    assert not loads
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: module._get_runtime(), range(8)))
    assert all(result is value for result in results)
    assert len(loads) == 1
    assert module.warmup()["ready"] is True
    assert module.readiness()["device"] == "cuda"


def test_loader_pins_revision_fp16_sdpa_and_disables_remote_code(monkeypatch):
    calls = []

    def pretrained(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace(eval=lambda: SimpleNamespace(device="cuda"))

    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                           get_device_name=lambda: "T4", get_device_capability=lambda: (7, 5))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, float16="FP16", __version__="2.9.1"))
    factory = SimpleNamespace(from_pretrained=pretrained)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=factory, Qwen3VLForConditionalGeneration=factory, __version__="5.24.1"))
    value = module._Runtime()
    assert len(calls) == 2
    assert all(model == module.MODEL_ID and options["revision"] == module.MODEL_REVISION and options["trust_remote_code"] is False for model, options in calls)
    assert calls[1][1]["dtype"] == "FP16"
    assert calls[1][1]["attn_implementation"] == "sdpa" and calls[1][1]["device_map"] == "cuda"
    assert value.details["gpu_name"] == "T4"


def test_no_cuda_fails_before_model_download_and_remains_unready(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not download a model without CUDA")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    factory = SimpleNamespace(from_pretrained=forbidden)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=factory, Qwen3VLForConditionalGeneration=factory))
    with pytest.raises(ProviderError, match="requires a CUDA GPU"):
        module.warmup()
    assert module.readiness()["ready"] is False
