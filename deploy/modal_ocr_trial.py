"""Development-only PaddleOCR-VL-1.6 OCR experiment; no structured extraction.

The Transformers interface recognizes elements. Whole receipts here are an
explicit trial, not the official layout detection / page parsing pipeline.
Run using the original environment read-only, from this separate copy:
PYTHONDONTWRITEBYTECODE=1 ../receipt-intelligence-poc/.venv/bin/modal run deploy/modal_ocr_trial.py
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import threading
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
REVISION = "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd"
APP_NAME = "receipt-ocr-vl16-rd"
GPU = os.getenv("OCR_TRIAL_GPU", "L4")
if GPU not in ("L4", "T4"):
    raise ValueError("OCR_TRIAL_GPU must be L4 or T4")
OUT = ROOT / "results/ocr_vl16_dev"
MAX_PIXELS = 1280 * 28 * 28
PROMPTS = {"ocr": "OCR:", "table": "Table Recognition:"}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "transformers==5.24.1",
                 "accelerate==1.12.0", "pillow==12.0.0", "psutil==7.1.3",
                 "fastapi==0.121.3", "pydantic==2.12.4", "sentencepiece==0.2.1",
                 "einops==0.8.1")
    .env({"HF_HOME": "/model-cache", "TOKENIZERS_PARALLELISM": "false", "OCR_TRIAL_GPU": GPU})
)
cache = modal.Volume.from_name("receipt-ocr-vl16-cache", create_if_missing=True)
app = modal.App(APP_NAME)
api_secret = modal.Secret.from_dict({"RECEIPT_API_KEY": os.environ.get("MODAL_API_KEY", "")})


class Resources:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.start_cpu = self.process.cpu_times()
        self.started = time.monotonic()
        self.baseline = self.process.memory_info().rss
        self.rss = [self.baseline]
        self.gpu = []
        self.done = threading.Event()

    def start(self):
        def sample():
            import subprocess
            tick = 0
            while not self.done.wait(0.2):
                try:
                    self.rss.append(self.process.memory_info().rss)
                    tick += 1
                    if tick % 5 == 0:
                        values = subprocess.check_output([
                            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], text=True, timeout=1)
                        self.gpu.append([float(x.strip()) for x in values.strip().split(",")])
                except Exception:
                    pass
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join(timeout=2)
        cpu = self.process.cpu_times()
        wall = time.monotonic() - self.started
        core_s = cpu.user + cpu.system - self.start_cpu.user - self.start_cpu.system
        return {"wall_seconds": wall, "cpu_core_seconds": core_s,
                "cpu_percent_mean": 100 * core_s / wall,
                "baseline_ram_bytes": self.baseline, "peak_ram_bytes": max(self.rss),
                "gpu_utilization_percent_mean": sum(v[0] for v in self.gpu) / len(self.gpu) if self.gpu else None,
                "gpu_memory_observed_peak_mib": max((v[1] for v in self.gpu), default=None)}


@app.cls(image=image, gpu=GPU, cpu=(2, 2), memory=(8192, 8192),
         volumes={"/model-cache": cache}, secrets=[api_secret],
         max_containers=1, min_containers=0, scaledown_window=2,
         timeout=180, startup_timeout=300, region="us")
class OCRTrial:
    @modal.enter()
    def load(self):
        self.started = time.monotonic()
        self.container_id = os.getenv("MODAL_TASK_ID")
        profile = Resources().start()
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # Official example uses BF16. T4 is an explicitly separate FP16
        # compatibility trial because compute capability7.5 lacks BF16.
        self.dtype = torch.float16 if GPU == "T4" else torch.bfloat16
        print("Starting pinned OCR model", MODEL_ID, REVISION, GPU, str(self.dtype), flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=self.dtype,
            attn_implementation="sdpa", device_map="cuda").eval()
        torch.cuda.synchronize()
        self.initialization = profile.stop()
        self.initialization_seconds = time.monotonic() - self.started
        self.gpu_name = torch.cuda.get_device_name()
        self.capability = list(torch.cuda.get_device_capability())
        self.versions = {"torch": str(torch.__version__), "transformers": str(transformers.__version__)}
        self.requests = 0
        self.lock = threading.Lock()
        cache.commit()
        print("OCR model ready", self.gpu_name, self.initialization_seconds, flush=True)

    def infer(self, image_bytes: bytes, task: str = "ocr"):
        import torch
        from PIL import Image
        if task not in PROMPTS:
            raise ValueError("Supported tasks: ocr, table")
        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("Image exceeds 10 MiB")
        if time.monotonic() - self.started > 2200:
            raise RuntimeError("Experiment container lifecycle budget exceeded")
        picture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if picture.width * picture.height > 40_000_000:
            raise ValueError("Image exceeds 40 MP")
        with self.lock:
            profile = Resources().start()
            torch.cuda.reset_peak_memory_stats()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": picture}, {"type": "text", "text": PROMPTS[task]}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", images_kwargs={"size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": MAX_PIXELS}}).to(self.model.device)
            generated = None
            error = None
            try:
                with torch.inference_mode():
                    generated = self.model.generate(**inputs, max_new_tokens=2048,
                                                    do_sample=False, max_time=150)
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)[:2000]
            torch.cuda.synchronize()
            n_input = int(inputs["input_ids"].shape[-1])
            output = generated[0][n_input:] if generated is not None else []
            raw = self.processor.decode(output, skip_special_tokens=True) if generated is not None else ""
            resource = profile.stop()
            resource.update({"allocated_gpu": GPU, "allocated_cpu_physical_cores": 2,
                             "allocated_ram_mib": 8192,
                             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                             "container_uptime_seconds": time.monotonic() - self.started,
                             "region_price_multiplier": 1.15})
            value = {"text": raw, "status": "error" if error else "ok", "error": error,
                     "metadata": {"model": MODEL_ID, "revision": REVISION,
                        "container_id": self.container_id, "prompt": PROMPTS[task], "task": task,
                        "dtype": str(self.dtype), "attention_implementation": "sdpa",
                        "compatibility_trial": GPU == "T4",
                        "gpu_name": self.gpu_name, "gpu_compute_capability": self.capability,
                        "runtime_versions": self.versions, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "original_image_size": list(picture.size), "max_processor_pixels": MAX_PIXELS,
                        "input_tokens": n_input, "output_tokens": len(output),
                        "output_limit_reached": len(output) >= 2048,
                        "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
                        "cold_first_request": self.requests == 0, "initialization_seconds": self.initialization_seconds,
                        "initialization_resource": self.initialization, "resource": resource,
                        "scope_note": "Element OCR model on whole receipt; official page parsing pipeline not used",
                        "measurement_note": "Observed process resources and container uptime; not provider billed duration"}}
            self.requests += 1
            return json.loads(json.dumps(value))

    @modal.method()
    def recognize(self, image_bytes: bytes, task: str = "ocr"):
        return self.infer(image_bytes, task)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        web = FastAPI(docs_url=None, redoc_url=None)

        @web.post("/ocr")
        async def ocr(request: Request):
            import asyncio
            expected = os.getenv("RECEIPT_API_KEY", "")
            if not expected or not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + expected):
                raise HTTPException(401, "Unauthorized")
            body = await request.body()
            if len(body) > 15 * 1024 * 1024:
                raise HTTPException(413, "Payload too large")
            payload = json.loads(body)
            return await asyncio.to_thread(self.infer, base64.b64decode(payload["image_b64"], validate=True), payload.get("task", "ocr"))
        return web


def save_result(entry, result, client_wall):
    from benchmark.ocr_benchmark import coverage_metrics
    truth = json.loads((ROOT / entry["annotation"]).read_text())
    result.update({"id": entry["id"], "dataset": entry["dataset"],
                   "language": entry["language"], "client_wall_seconds": client_wall,
                   "metrics": coverage_metrics(result.get("text", ""), truth["ocr_text"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (entry["id"] + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"id": entry["id"], "status": result["status"],
                      "seconds": client_wall, "metrics": result["metrics"]}), flush=True)


@app.local_entrypoint()
def run(endpoint: str = ""):
    entries = json.loads((ROOT / "data/dev_manifest.json").read_text())["development"]
    assert len(entries) == 8 and sum(e["dataset"] == "AraReceipt" for e in entries) == 5
    service = OCRTrial()
    started = time.monotonic()
    for entry in entries:
        if time.monotonic() - started > 2600:
            raise RuntimeError("Experiment wall budget exceeded")
        data = (ROOT / entry["image"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        before = time.monotonic()
        if endpoint:
            import httpx
            with httpx.Client(timeout=480, follow_redirects=True, trust_env=False) as client:
                response = client.post(endpoint.rstrip("/") + "/ocr",
                    headers={"Authorization": "Bearer " + os.environ["MODAL_API_KEY"]},
                    json={"image_b64": base64.b64encode(data).decode(), "task": "ocr"})
                response.raise_for_status()
                result = response.json()
        else:
            result = service.recognize.remote(data)
        save_result(entry, result, time.monotonic() - before)
