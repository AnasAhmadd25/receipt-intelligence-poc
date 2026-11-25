"""Small, scale-to-zero document VLM service. No external inference APIs.

Deploy: .venv/bin/modal deploy deploy/modal_app.py
Dev experiment: .venv/bin/modal run deploy/modal_app.py --manifest data/dev_manifest.json
All model overrides and images stay in the independently created POC directory.
"""
import base64
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

MODEL_ID = os.environ.get("MODAL_MODEL_ID", "zai-org/GLM-OCR")
GPU = os.environ.get("MODAL_GPU", "T4")
CPU = float(os.environ.get("MODAL_CPU", "2"))
MEMORY_MIB = int(os.environ.get("MODAL_MEMORY_MIB", "8192"))
MAX_BYTES = 10 * 1024 * 1024
REVISIONS = {
    "zai-org/GLM-OCR": "2e85a62840ccac27daa451df36c736c4636b8628",
    "tencent/HunyuanOCR": "47644ecc4fc854efa4f505155158831f36773ee4",
    "Qwen/Qwen3-VL-2B-Instruct": "89644892e4d85e24eaac8bacfd4f463576704203",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1", "torchvision==0.24.1", "transformers==5.24.1",
        "accelerate==1.12.0", "pillow==12.0.0", "psutil==7.1.3",
        "fastapi==0.121.3", "pydantic==2.12.4", "python-multipart==0.0.20",
        "sentencepiece==0.2.1", "einops==0.8.1",
    )
    .env({"HF_HOME": "/model-cache", "TOKENIZERS_PARALLELISM": "false",
          "MODAL_MODEL_ID": MODEL_ID, "MODAL_GPU": GPU,
          "MODAL_CPU": str(CPU), "MODAL_MEMORY_MIB": str(MEMORY_MIB)})
)
cache = modal.Volume.from_name("receipt-intelligence-model-cache", create_if_missing=True)
if MODEL_ID == "Qwen/Qwen3-1.7B":
    image = (image.apt_install("libgomp1")
             .pip_install("paddlepaddle==3.3.1", "paddleocr==3.7.0", "paddlex==3.7.2")
             .env({"PADDLE_PDX_CACHE_HOME": "/model-cache/paddle", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}))
image = image.add_local_python_source("app")
app = modal.App(os.getenv("MODAL_APP_NAME", "receipt-intelligence-poc"))
api_secret = modal.Secret.from_dict({"RECEIPT_API_KEY": os.environ.get("MODAL_API_KEY", "")})


class SampleResources:
    """Process and system samples, not allocated/billed resource measurements."""
    def __init__(self):
        import psutil
        import resource
        self.psutil = psutil
        self.rusage = resource
        self.child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.process = psutil.Process()
        self.start_cpu = self.process.cpu_times()
        self.started = time.monotonic()
        self.baseline = self.process.memory_info().rss
        self.rss = [self.baseline]
        self.cpu = []
        self.gpu = []
        self.done = threading.Event()

    def start(self):
        def sample():
            import subprocess
            self.process.cpu_percent(None)
            while not self.done.wait(0.1):
                children = self.process.children(recursive=True)
                self.rss.append(self.process.memory_info().rss + sum(
                    p.memory_info().rss for p in children if p.is_running()))
                self.cpu.append(self.process.cpu_percent(None))
                if len(self.cpu) % 10 == 0:
                    try:
                        values = subprocess.check_output([
                            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], timeout=1, text=True)
                        self.gpu.append([float(x.strip()) for x in values.strip().split(",")])
                    except Exception:
                        pass
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.rss.append(self.process.memory_info().rss)
        cpu = self.process.cpu_times()
        child_cpu = self.rusage.getrusage(self.rusage.RUSAGE_CHILDREN)
        core_seconds = (cpu.user + cpu.system - self.start_cpu.user - self.start_cpu.system
                        + child_cpu.ru_utime + child_cpu.ru_stime
                        - self.child_cpu_start.ru_utime - self.child_cpu_start.ru_stime)
        wall = time.monotonic() - self.started
        return {
            "wall_seconds": wall,
            "cpu_core_seconds": core_seconds,
            "cpu_percent_mean": 100 * core_seconds / wall,
            "cpu_percent_peak": max(self.cpu, default=None),
            "baseline_ram_bytes": self.baseline,
            "peak_ram_bytes": max(self.rss),
            "gpu_utilization_percent_mean": sum(v[0] for v in self.gpu) / len(self.gpu) if self.gpu else None,
            "gpu_memory_observed_peak_mib": max((v[1] for v in self.gpu), default=None),
        }


@app.cls(image=image, gpu=GPU if GPU.lower() != "cpu" else None,
         cpu=(CPU, CPU), memory=(MEMORY_MIB, MEMORY_MIB), volumes={"/model-cache": cache},
         secrets=[api_secret], max_containers=1, min_containers=0,
         scaledown_window=2, timeout=180, startup_timeout=300,
         region="us")
class ReceiptService:
    model_id: str = modal.parameter(default=MODEL_ID)

    @modal.enter()
    def load(self):
        import uuid
        self.container_id = os.getenv("MODAL_TASK_ID") or str(uuid.uuid4())
        self.container_started = time.monotonic()
        profiler = SampleResources().start()
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer
        self.text_mode = self.model_id == "Qwen/Qwen3-1.7B"
        self.requests = 0
        self.lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # T4 is compute capability 7.5: use FP16/eager, never BF16 or FlashAttention-2.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        cls_name = {"zai-org/GLM-OCR": "GlmOcrForConditionalGeneration",
                    "tencent/HunyuanOCR": "HunYuanVLForConditionalGeneration",
                    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3VLForConditionalGeneration",
                    "Qwen/Qwen3-1.7B": "Qwen3ForCausalLM"}[self.model_id]
        revision = REVISIONS[self.model_id]
        factory = AutoTokenizer if self.text_mode else AutoProcessor
        self.processor = factory.from_pretrained(self.model_id, revision=revision)
        self.model = getattr(transformers, cls_name).from_pretrained(
            self.model_id, revision=revision, dtype=self.dtype,
            attn_implementation="sdpa", device_map=self.device,
        ).eval()
        if self.device == "cuda":
            torch.cuda.synchronize()
        cache.commit()
        self.initialization = profiler.stop()
        self.initialization_seconds = time.monotonic() - self.container_started
        self.gpu_name = torch.cuda.get_device_name() if self.device == "cuda" else None
        self.compute_capability = list(torch.cuda.get_device_capability()) if self.device == "cuda" else None
        self.versions = {"torch": str(torch.__version__), "transformers": str(transformers.__version__)}

    def infer(self, image_bytes: bytes, *, prompt: str | None = None, merchant_id: str | None = None):
        from PIL import Image
        import torch
        from app.schema import Receipt
        if len(image_bytes) > MAX_BYTES:
            raise ValueError("Image exceeds 10 MiB")
        picture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if picture.width * picture.height > 40_000_000:
            raise ValueError("Image exceeds 40 megapixels")
        original_size = list(picture.size)
        if picture.width * picture.height > 1_000_000:
            factor = (1_000_000 / (picture.width * picture.height)) ** .5
            picture = picture.resize((int(picture.width * factor), int(picture.height * factor)), Image.Resampling.LANCZOS)
        # No held-out test-specific prompts. Template is fixed before evaluation.
        shape = Receipt().model_dump(mode="json")
        shape["line_items"] = [{"raw_description": None, "quantity": None, "unit_price": None, "line_total": None}]
        instruction = prompt or (
            "Extract only information actually visible in this receipt. Preserve Arabic and original product text. "
            "Do not infer currency or invent unreadable values. Use null for unavailable fields. "
            "Return only one JSON object in this exact shape (monetary values as decimal strings):\n"
            + json.dumps(shape, ensure_ascii=False)
        )
        if prompt is None and self.model_id == "zai-org/GLM-OCR":
            # Model-card information-extraction template, rather than an arbitrary chat task.
            instruction = "请按下列JSON格式输出图中信息:\n" + json.dumps(shape, ensure_ascii=False) + "\nMissing fields must be null. Do not invent information. Preserve original Arabic text."
        with self.lock:
            cold = self.requests == 0
            profiler = SampleResources().start()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            observed = None
            if self.text_mode:
                from app.providers.ocr import run_ocr
                observed = run_ocr(image_bytes, engine="paddle", language=os.getenv("OCR_LANGUAGE", "ar"))
                messages = [{"role": "user", "content": instruction + "\nReceipt OCR and layout:\n" + json.dumps(observed, ensure_ascii=False)}]
                inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=True, return_tensors="pt").to(self.model.device)
            else:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": picture}, {"type": "text", "text": instruction}]}]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(self.model.device)
            generated = None
            inference_error = None
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs, max_new_tokens=2048, do_sample=False,
                        repetition_penalty=1.08, max_time=150,
                    )
            except Exception as exc:
                inference_error = type(exc).__name__ + ": " + str(exc)
            if self.device == "cuda":
                torch.cuda.synchronize()
            raw = (self.processor.batch_decode(generated[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
                   if generated is not None else "")
            resource = profiler.stop()
            resource.update({
                "gpu_name": self.gpu_name, "gpu_compute_capability": self.compute_capability,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
                "allocated_cpu_physical_cores": CPU, "allocated_ram_mib": MEMORY_MIB,
                "allocated_gpu": GPU, "container_uptime_seconds": time.monotonic() - self.container_started,
                "region_selection": "us", "region_price_multiplier": 1.15,
            })
            metadata = {"model": self.model_id, "revision": REVISIONS[self.model_id],
                        "container_id": self.container_id,
                        "runtime_versions": self.versions, "dtype": str(self.dtype),
                        "attention_implementation": "sdpa", "original_image_size": original_size,
                        "inference_image_size": list(picture.size), "image_pixel_cap": 1_000_000,
                        "latency_seconds": resource["wall_seconds"], "cold_first_request": cold,
                        "initialization_seconds": self.initialization_seconds,
                        "initialization_resource": self.initialization, "resource": resource,
                        "input_tokens": inputs["input_ids"].shape[-1],
                        "output_tokens": generated.shape[-1] - inputs["input_ids"].shape[-1] if generated is not None else 0,
                        "fallback_used": self.text_mode, "json_valid": False,
                        "billed_duration_note": "Container uptime is observed, not a Modal invoice; includes init, excludes provider scheduling before Python start and terminal idle."}
            self.requests += 1
            if observed is not None:
                metadata["ocr"] = {"engine": observed["engine"], "latency_seconds": observed["latency_seconds"], "model": observed["model"]}
            if inference_error:
                metadata["inference_error"] = inference_error
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                return {"receipt": None, "metadata": metadata, "raw_output": raw}
        # Keep invalid output in the envelope; callers must record json_valid=False.
        try:
            value = raw.strip()
            if value.startswith("```"):
                value = value.split("\n", 1)[1].rsplit("```", 1)[0]
            receipt = Receipt.model_validate_json(value)
            if observed is not None:
                receipt.raw_text = observed["text"]
            metadata["json_valid"] = True
            return {"receipt": receipt.model_dump(mode="json"), "metadata": metadata, "raw_output": raw}
        except Exception as exc:
            metadata["validation_error"] = str(exc)[:2000]
            return {"receipt": None, "metadata": metadata, "raw_output": raw}

    @modal.method()
    def extract(self, image_bytes: bytes, prompt: str | None = None):
        # Transport only JSON primitives; TorchVersion is a str subclass that
        # otherwise makes a CPU-only caller import torch while unpickling.
        return json.loads(json.dumps(self.infer(image_bytes, prompt=prompt)))

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import JSONResponse
        web = FastAPI(title="Receipt Intelligence POC", docs_url=None, redoc_url=None)

        @web.middleware("http")
        async def protect(request: Request, call_next):
            expected = os.environ.get("RECEIPT_API_KEY", "")
            supplied = request.headers.get("authorization", "")
            if not expected or not secrets.compare_digest(supplied, "Bearer " + expected):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if int(request.headers.get("content-length", "0")) > MAX_BYTES * 2:
                return JSONResponse({"detail": "Payload too large"}, status_code=413)
            return await call_next(request)

        @web.get("/health")
        def health():
            return {"status": "ok", "model": self.model_id, "device": self.device}

        @web.post("/v1/extract")
        async def extract(request: Request):
            import asyncio
            body = await request.body()
            if len(body) > MAX_BYTES * 2:
                raise HTTPException(413, "Payload too large")
            try:
                payload = json.loads(body)
                data = base64.b64decode(payload["image_b64"], validate=True)
                return await asyncio.to_thread(self.infer, data, merchant_id=payload.get("merchant_id"))
            except (KeyError, ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc

        @web.post("/v1/batch")
        async def batch(request: Request):
            import asyncio
            body = await request.body()
            if len(body) > MAX_BYTES * 2:
                raise HTTPException(413, "Payload too large")
            try:
                payload = json.loads(body)
                items = payload["images"]
                if not isinstance(items, list) or not 1 <= len(items) <= 20:
                    raise ValueError("Batch requires 1 to 20 images")
                data = [base64.b64decode(item["image_b64"], validate=True) for item in items]
                if any(len(item) > MAX_BYTES for item in data):
                    raise ValueError("Image exceeds 10 MiB")
                # Serialize model use within a single container; this is a synchronous POC batch.
                results = []
                for item in data:
                    results.append(await asyncio.to_thread(self.infer, item))
                return {"results": results}
            except (KeyError, ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
        return web


cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("tesseract-ocr", "tesseract-ocr-ara", "tesseract-ocr-eng")
             .pip_install("pytesseract==0.3.13", "pillow==12.0.0", "psutil==7.1.3"))


@app.function(image=cpu_image, cpu=(1, 1), memory=(4096, 4096),
              max_containers=1, scaledown_window=2, timeout=300)
def linux_cpu_probe(images: list[bytes], concurrency: int = 1):
    """Linux x86 OCR validation with 2vCPU-equivalent CPU and 4GiB RAM."""
    import concurrent.futures
    import platform
    import pytesseract
    from PIL import Image
    if concurrency not in (1, 2, 4):
        raise ValueError("concurrency must be 1, 2 or 4")
    os.environ["OMP_THREAD_LIMIT"] = "1"
    def ocr(data):
        start = time.monotonic()
        text = pytesseract.image_to_string(Image.open(io.BytesIO(data)), lang="ara+eng", config="--psm 6")
        return {"text": text, "seconds": time.monotonic() - start}
    profile = SampleResources().start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(ocr, images))
    resource = profile.stop()
    resource.update({"cpu_model": Path("/proc/cpuinfo").read_text().split("model name",1)[-1].split("\n",1)[0].strip(" :\t"),
                     "allocated_cpu_physical_cores": 1, "allocated_vcpu_equivalent": 2,
                     "allocated_ram_mib": 4096, "concurrency": concurrency,
                     "platform": platform.platform(), "receipts": len(rows),
                     "receipts_per_minute": len(rows) * 60 / resource["wall_seconds"]})
    return {"rows": rows, "resource": resource, "tesseract_version": str(pytesseract.get_tesseract_version())}


paddle_image = (cpu_image.apt_install("libgl1", "libglib2.0-0", "libgomp1")
                .pip_install("paddlepaddle==3.3.1", "paddleocr==3.7.0", "paddlex==3.7.2")
                .env({"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                      "PADDLE_PDX_CACHE_HOME": "/model-cache/paddle"})
                .add_local_python_source("app"))


@app.function(image=paddle_image, cpu=(1, 1), memory=(8192, 8192),
              volumes={"/model-cache": cache}, max_containers=1, scaledown_window=2, timeout=300)
def linux_paddle_probe(images: list[bytes], languages: list[str], concurrency: int = 1):
    """Same selected Paddle pipeline on Linux, preserving its serialization lock."""
    import concurrent.futures
    import platform
    from app.providers.ocr import run_ocr
    if concurrency not in (1, 2, 4):
        raise ValueError("concurrency must be 1, 2 or 4")
    profile = SampleResources().start()
    def ocr(pair):
        data, language = pair
        return run_ocr(data, engine="paddle", language=language)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(ocr, zip(images, languages)))
    resource = profile.stop()
    resource.update({"allocated_cpu_physical_cores": 1, "allocated_vcpu_equivalent": 2,
                     "allocated_ram_mib": 8192, "concurrency": concurrency,
                     "platform": platform.platform(), "receipts": len(rows),
                     "receipts_per_minute": len(rows) * 60 / resource["wall_seconds"],
                     "container_id": os.getenv("MODAL_TASK_ID"),
                     "note": "Paddle engine cache uses a process lock; higher request concurrency does not imply simultaneous inference."})
    cache.commit()
    return {"rows": rows, "resource": resource}


@app.local_entrypoint()
def dev(manifest: str, model: str = MODEL_ID, limit: int = 4, repeats: int = 1):
    """Bounded dev-only experiment; one container reused across all calls."""
    path = Path(manifest)
    entries = json.loads(path.read_text())
    if isinstance(entries, dict):
        entries = entries.get("development", entries.get("receipts", entries.get("samples", entries.get("items", []))))
    entries = [r for r in entries if r.get("split", "development") in ("development", "dev")]
    if not entries:
        raise ValueError("Model selection requires development entries")
    # Fixed balanced dev selection: Arabic and Latin-script receipts, no test samples.
    groups = {}
    for entry in entries:
        groups.setdefault(entry.get("dataset", "unknown"), []).append(entry)
    balanced = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                balanced.append(group.pop(0))
    entries = balanced
    service = ReceiptService(model_id=model)
    rows = []
    for entry in entries[:limit]:
        filename = entry.get("image_path", entry.get("image"))
        image_path = Path(filename)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        for repeat in range(repeats):
            start = time.monotonic()
            result = service.extract.remote(image_path.read_bytes())
            result.update({"sample_id": entry.get("id", entry.get("sample_id")), "repeat": repeat,
                           "client_wall_seconds": time.monotonic() - start})
            rows.append(result)
            dest = ROOT / "results" / ("modal_dev_" + model.split("/")[-1] + ".json")
            dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            print({"sample_id": result["sample_id"], "valid": result["metadata"]["json_valid"],
                   "seconds": round(result["client_wall_seconds"], 3)})
