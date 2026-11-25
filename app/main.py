"""Minimal synchronous receipt API. Run with uvicorn app.main:app."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import os
from typing import Literal
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from app.providers.base import ProviderError
from app.providers.cloud import get_provider
from app.security import InputError, MAX_IMAGE_BYTES, fetch_image_url, validate_image

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
ProviderName = Literal["ocr_ollama", "ocr_nvidia", "gemini", "modal_hybrid", "ocr_openrouter"]
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ocr_nvidia")
MAX_REQUEST_BYTES = 12 * 1024 * 1024


class ExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_url: str = Field(min_length=1, max_length=4096)
    provider: ProviderName = DEFAULT_PROVIDER
    merchant_id: str | None = Field(default=None, max_length=200)


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ExtractionInput] = Field(min_length=1, max_length=20)
    concurrency: int = Field(default=2, ge=1, le=4)


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Request exceeds 12 MiB; each image is limited to 10 MiB")
            return message
        lengths = [v for k, v in scope.get("headers", []) if k == b"content-length"]
        if lengths and (not lengths[0].isdigit() or int(lengths[0]) > MAX_REQUEST_BYTES):
            return await JSONResponse(status_code=413, content={"detail": "Request exceeds size limit"})(scope, receive, send)
        await self.app(scope, bounded_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("PUBLIC_DEPLOYMENT", "").lower() in {"1", "true", "yes"} and not os.getenv("API_KEY", "").strip():
        raise RuntimeError("Public deployment requires API_KEY")
    app.state.inference_slots = asyncio.Semaphore(max(1, min(4, int(os.getenv("MAX_IN_FLIGHT", "4")))))
    yield


app = FastAPI(title="Receipt Intelligence POC", version="0.1.0", lifespan=lifespan)
app.add_middleware(BodyLimit)


async def authenticate(request: Request):
    expected = os.getenv("API_KEY", "")
    if expected:
        supplied = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "Valid API key required")


@app.get("/health")
async def health():
    return {"status": "ok", "schema_version": "1.0"}


def validate_model(model, value):
    try:
        return model.model_validate(value)
    except ValidationError:
        # Validation errors can embed signed image URLs, so expose a summary only.
        raise HTTPException(422, "Invalid request fields, provider, batch size, or concurrency") from None


async def input_bytes(file: UploadFile | None, image_url: str | None):
    if (file is not None) == bool(image_url):
        raise HTTPException(422, "Provide exactly one file or image_url")
    try:
        if file is not None:
            try:
                data = await file.read(MAX_IMAGE_BYTES + 1)
            finally:
                await file.close()
        else:
            data = await asyncio.wait_for(fetch_image_url(image_url), timeout=30)
        return await asyncio.to_thread(validate_image, data)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from None
    except asyncio.TimeoutError:
        raise HTTPException(422, "Image download exceeded time limit") from None


async def perform(request: Request, *, provider: str, merchant_id: str | None, file: UploadFile | None = None, image_url: str | None = None):
    try:
        implementation = get_provider(provider)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    # The same cap applies across concurrent HTTP requests and all batch items.
    async with request.app.state.inference_slots:
        image, mime = await input_bytes(file, image_url)
        try:
            return await asyncio.wait_for(implementation.extract(image, mime, merchant_id=merchant_id), timeout=300)
        except ProviderError as exc:
            raise HTTPException(502, {"error": str(exc), "metadata": exc.metadata}) from None
        except asyncio.TimeoutError:
            raise HTTPException(504, "Receipt extraction exceeded time limit") from None


@app.post("/v1/extract", dependencies=[Depends(authenticate)])
async def extract(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0] == "application/json":
        try:
            payload = validate_model(ExtractionInput, await request.json())
        except ValueError:
            raise HTTPException(422, "Invalid JSON body") from None
        return await perform(request, provider=payload.provider, merchant_id=payload.merchant_id, image_url=payload.image_url)
    async with request.form(max_files=1, max_fields=4, max_part_size=MAX_IMAGE_BYTES) as form:
        file = form.get("file")
        if file is not None and not isinstance(file, UploadFile):
            raise HTTPException(422, "file must be an uploaded image")
        provider = form.get("provider", DEFAULT_PROVIDER)
        merchant_id = form.get("merchant_id")
        image_url = form.get("image_url")
        validate_model(ExtractionInput, {"image_url": image_url or "https://upload.invalid", "provider": provider, "merchant_id": merchant_id})
        return await perform(request, provider=provider, merchant_id=merchant_id, file=file, image_url=image_url)


@app.post("/v1/batch", dependencies=[Depends(authenticate)])
async def batch(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0] == "application/json":
        try:
            payload = validate_model(BatchInput, await request.json())
        except ValueError:
            raise HTTPException(422, "Invalid JSON body") from None
        jobs = [{"provider": item.provider, "merchant_id": item.merchant_id, "image_url": item.image_url} for item in payload.items]
        return await execute_batch(request, jobs, payload.concurrency)
    async with request.form(max_files=20, max_fields=4, max_part_size=MAX_IMAGE_BYTES) as form:
        files = form.getlist("files")
        if not files or len(files) > 20 or not all(isinstance(file, UploadFile) for file in files):
            raise HTTPException(422, "Upload 1–20 images in files")
        provider, merchant_id = form.get("provider", DEFAULT_PROVIDER), form.get("merchant_id")
        try:
            concurrency = int(form.get("concurrency", 2))
        except (ValueError, TypeError):
            raise HTTPException(422, "concurrency must be 1–4") from None
        validate_model(BatchInput, {"items": [{"image_url": "https://upload.invalid", "provider": provider, "merchant_id": merchant_id}], "concurrency": concurrency})
        return await execute_batch(request, [{"file": file, "provider": provider, "merchant_id": merchant_id} for file in files], concurrency)


async def execute_batch(request: Request, jobs: list[dict], concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    async def run(index, job):
        async with semaphore:
            try:
                return {"index": index, "result": await perform(request, **job)}
            except HTTPException as exc:
                return {"index": index, "error": exc.detail, "status_code": exc.status_code}
    return {"results": await asyncio.gather(*(run(index, job) for index, job in enumerate(jobs)))}
