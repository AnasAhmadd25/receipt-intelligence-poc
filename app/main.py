"""Minimal synchronous receipt API. Run with uvicorn app.main:app."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import os
from typing import Any, Literal, get_args
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from app.providers.base import ProviderError
from app.providers.cloud import get_provider
from app.schema import Receipt
from app.security import InputError, MAX_IMAGE_BYTES, fetch_image_url, validate_image

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
ProviderName = Literal["ocr_ollama", "ocr_nvidia", "gemini", "modal_hybrid", "ocr_openrouter", "ocr_gemini", "selfhosted"]
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


class ExtractionResponse(BaseModel):
    receipt: Receipt
    metadata: dict[str, Any] = Field(description="Provider/model, usage, timing and validation evidence; fields vary by provider")


class ProviderFailure(BaseModel):
    error: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str | ProviderFailure


class BatchSuccess(BaseModel):
    index: int = Field(ge=0)
    result: ExtractionResponse


class BatchFailure(BaseModel):
    index: int = Field(ge=0)
    error: str | ProviderFailure
    status_code: int = Field(ge=400, le=599)


class BatchResponse(BaseModel):
    results: list[BatchSuccess | BatchFailure]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    schema_version: Literal["1.0"]
    default_provider: str
    allowed_providers: list[str]


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ready: bool
    provider: str | None = None
    default_provider: str | None = None
    scope: str | None = None


def allowed_providers() -> list[str]:
    configured = os.getenv("ALLOWED_PROVIDERS")
    names = list(get_args(ProviderName)) if configured is None else list(dict.fromkeys(name.strip() for name in configured.split(",") if name.strip()))
    if not names or any(name not in get_args(ProviderName) for name in names):
        raise RuntimeError("ALLOWED_PROVIDERS must contain known, comma-separated provider names")
    if DEFAULT_PROVIDER not in names:
        raise RuntimeError("DEFAULT_PROVIDER must be included in ALLOWED_PROVIDERS")
    return names


ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Malformed multipart request", 401: "Missing or invalid API key",
        413: "Request exceeds 12 MiB", 422: "Invalid fields, unsupported provider or unsafe image input",
        502: "Provider/OCR failure with safe diagnostic metadata", 504: "Inference exceeded 300 seconds",
    }.items()
}


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
    app.state.allowed_providers = allowed_providers()
    app.openapi_schema = None
    app.state.inference_slots = asyncio.Semaphore(max(1, min(4, int(os.getenv("MAX_IN_FLIGHT", "4")))))
    app.state.local_provider = get_provider("selfhosted") if DEFAULT_PROVIDER == "selfhosted" else None
    if app.state.local_provider is not None and os.getenv("SELFHOSTED_WARMUP", "").lower() in {"1", "true", "yes"}:
        await asyncio.to_thread(app.state.local_provider.warmup)
    yield


app = FastAPI(title="Receipt Intelligence API", version="0.2.0", lifespan=lifespan,
              description="Shared receipt contract for CPU OCR + hosted LLM, Gemini, and a local GPU model. Set either X-API-Key or Bearer authentication using Authorize. Health is liveness only.")
app.add_middleware(BodyLimit)

api_key_header = APIKeyHeader(name="X-API-Key", scheme_name="APIKey", auto_error=False)
http_bearer = HTTPBearer(scheme_name="BearerAuth", auto_error=False)


async def authenticate(request: Request, api_key: str | None = Security(api_key_header),
                       bearer: HTTPAuthorizationCredentials | None = Security(http_bearer)):
    expected = os.getenv("API_KEY", "")
    if expected:
        supplied = bearer.credentials if bearer else api_key or ""
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "Valid API key required")


@app.get("/health", response_model=HealthResponse, summary="Process liveness and enabled providers")
async def health(request: Request):
    return {"status": "ok", "schema_version": "1.0", "default_provider": DEFAULT_PROVIDER,
            "allowed_providers": request.app.state.allowed_providers}


@app.get("/ready", response_model=ReadinessResponse, summary="Model readiness without making a hosted API call",
         responses={200: {"description": "Configured or warmed; hosted credentials are not tested"},
                    503: {"model": ReadinessResponse, "description": "Local model is not loaded yet"}})
async def ready(request: Request):
    provider = request.app.state.local_provider
    if provider is None:
        return {"ready": True, "default_provider": DEFAULT_PROVIDER,
                "scope": "configuration only; hosted credentials and OCR weights are not tested"}
    status = provider.readiness()
    return JSONResponse(status_code=200 if status.get("ready") else 503, content=status)


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
    if provider not in request.app.state.allowed_providers:
        raise HTTPException(422, "Provider is not enabled on this service")
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


@app.post("/v1/extract", dependencies=[Depends(authenticate)], response_model=ExtractionResponse,
          responses=ERROR_RESPONSES, summary="Extract one receipt from an image upload or secure URL")
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


@app.post("/v1/batch", dependencies=[Depends(authenticate)], response_model=BatchResponse,
          responses=ERROR_RESPONSES, summary="Extract 1–20 receipts, preserving input order",
          description="Accepts JSON URL items or repeated multipart files. HTTP 200 may contain per-item failures; inspect every result. The 12 MiB total request limit still applies.")
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


def custom_openapi():
    if app.openapi_schema is not None:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in (ExtractionInput, BatchInput):
        definition = model.model_json_schema(ref_template="#/components/schemas/{model}")
        components.update(definition.pop("$defs", {}))
        components[model.__name__] = definition
    names = allowed_providers()
    provider_property = components["ExtractionInput"]["properties"]["provider"]
    provider_property.update(enum=names, default=DEFAULT_PROVIDER)
    upload_properties = {
        "provider": {"type": "string", "enum": names, "default": DEFAULT_PROVIDER},
        "merchant_id": {"type": "string", "maxLength": 200},
    }
    for path, model, is_batch in (("/v1/extract", "ExtractionInput", False), ("/v1/batch", "BatchInput", True)):
        properties = dict(upload_properties)
        if is_batch:
            properties.update(files={"type": "array", "minItems": 1, "maxItems": 20,
                                     "items": {"type": "string", "format": "binary"}},
                              concurrency={"type": "integer", "minimum": 1, "maximum": 4, "default": 2})
        else:
            properties.update(file={"type": "string", "format": "binary"},
                              image_url={"type": "string", "minLength": 1, "maxLength": 4096})
        multipart = {"type": "object", "properties": properties,
                     "description": "12 MiB total request; 10 MiB per image. PNG, JPEG, WebP only."}
        if is_batch:
            multipart["required"] = ["files"]
        else:
            multipart["oneOf"] = [{"required": ["file"], "not": {"required": ["image_url"]}},
                                  {"required": ["image_url"], "not": {"required": ["file"]}}]
        schema["paths"][path]["post"]["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{model}"}},
                        "multipart/form-data": {"schema": multipart}},
        }
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
