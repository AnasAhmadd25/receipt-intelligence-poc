# Receipt Intelligence API

All three services expose the same paths, request fields and receipt schema. Containers bind to localhost by default; use an authenticated TLS reverse proxy when exposing a service remotely.

| Service | Port | Default provider | Processing |
| --- | --- | --- | --- |
| A: CPU OCR + hosted text LLM | 8001 | `ocr_nvidia` | PaddleOCR CPU reads the image; a configured hosted model structures the text |
| B: Gemini | 8002 | `gemini` | Gemini reads and structures the original image directly |
| C: Local GPU model | 8003 | `selfhosted` | Qwen3-VL runs inside the container on an NVIDIA GPU |

Service A enables `ocr_nvidia`, `ocr_openrouter`, `ocr_ollama`, and `ocr_gemini`. Service B accepts only `gemini`; service C accepts only `selfhosted`. `ALLOWED_PROVIDERS` is enforced before image downloads or provider creation. An unknown/empty allow-list, or a default outside the allow-list, stops startup. Change A's default with `OCR_DEFAULT_PROVIDER`; the default remains `ocr_nvidia` unless explicitly configured otherwise. Model availability, quota and pricing are account dependent; no provider is guaranteed free.

## Start containers

From this directory, create a local `.env` with `API_KEY` and the credentials for the services you need. `.env` stays outside image build contexts. Compose passes only explicitly listed variables into containers. `API_KEY` is the key protecting these receipt APIs; it is separate from hosted provider credentials.

```dotenv
API_KEY=replace-with-a-long-random-secret
NVIDIA_API_KEY=your-hosted-key
NVIDIA_MODEL=an-exact-model-id-from-your-authenticated-inventory
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

```sh
docker compose up --build -d ocr gemini
# Linux host with an NVIDIA GPU, compatible driver and NVIDIA Container Toolkit:
docker compose --profile gpu up --build -d selfhosted
```

A uses Python 3.12, PaddlePaddle **3.3.1**, PaddleOCR **3.7.0**, and PaddleX **3.7.2** on `linux/amd64`. Its runtime sets `OCR_ENGINE=paddle`, Arabic as the default language hint, and `MAX_IN_FLIGHT=1`. Set `OCR_LANGUAGE=en` for English-only receipts. The Arabic and English recognizers are separate; this setting does not promise mixed-language recognition accuracy. Compose explicitly selects amd64; Apple Silicon runs A through emulation, with no native ARM compatibility or performance claim.

OCR weights are **not baked into the image**. Paddle downloads its selected detector/recognizer the first time that language is used; outbound access and sufficient initialization time are required. The `paddle-models` and `ocr-cache` volumes retain downloads. A cold first request can fail if downloads exceed the 300-second inference timeout; warm the cache with a controlled real image before measuring throughput. Do not treat process health as proof that OCR models downloaded successfully.

C is an optional `gpu` Compose profile. It requires a Linux NVIDIA GPU host, a compatible NVIDIA driver and NVIDIA Container Toolkit. Docker Desktop on an Apple Silicon Mac cannot provide the required CUDA GPU. `HF_HOME=/model-cache` retains its downloaded weights in `qwen-models`. `SELFHOSTED_WARMUP=true` loads the pinned Qwen model before startup completes; insufficient VRAM, missing drivers or inaccessible weights cause a startup failure. See the GPU Dockerfile and provider module for pinned runtime/model details. A GPU image build alone does not demonstrate inference works.

Optional A credentials: `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (the existing OpenRouter adapter requires an exact `:free` model), `OLLAMA_API_KEY` and `OLLAMA_MODEL`, or `GEMINI_API_KEY` and `OCR_GEMINI_MODEL` (default `gemini-3.1-flash-lite` for text only). B defaults to `gemini-3.5-flash-lite`; override `GEMINI_MODEL` to compare other models. NVIDIA requires both its key and a model ID. Hosted credentials are tested only by a real extraction request.

## Interactive documentation and authentication

Use `http://localhost:8001/docs`, `http://localhost:8002/docs`, or `http://localhost:8003/docs` for Swagger UI. `/redoc` is the read-only reference and `/openapi.json` is the machine-readable specification, including both multipart and JSON bodies, receipt responses, errors and enabled provider names.

Click **Authorize** in Swagger and set either `APIKey` to the raw API key, or `BearerAuth` to the raw key. A request needs either `X-API-Key: KEY` or `Authorization: Bearer KEY`. If both headers are supplied, Bearer takes precedence. Health, readiness and documentation paths are public; extraction and batch require the key whenever `API_KEY` is set. Every container sets `PUBLIC_DEPLOYMENT=true` and refuses startup if the key is blank.

## Endpoints

| Method/path | Contract |
| --- | --- |
| `GET /health` | Process liveness, schema version, default and enabled providers; never calls inference |
| `GET /ready` | A/B: configuration-level readiness only. C: loaded local model/runtime state; 503 until loaded |
| `POST /v1/extract` | One image upload or one secure public image URL |
| `POST /v1/batch` | 1–20 uploads or URL items, preserving input order |
| `GET /docs`, `/redoc`, `/openapi.json` | Shared interactive/reference documentation and schemas |

### One upload

```sh
curl --fail-with-body http://localhost:8001/v1/extract \
  -H "X-API-Key: $API_KEY" \
  -F 'file=@/absolute/path/receipt.png' \
  -F 'merchant_id=shop-42'
```

`provider` is optional and defaults to the service's configured default. `merchant_id` is optional, up to 200 characters; it provides context and is never evidence for a merchant name. Upload field `file` must be a PNG, JPEG or WebP image. Do not include `image_url` with a file.

### One secure URL

```sh
curl --fail-with-body http://localhost:8002/v1/extract \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://images.example.org/receipt.png","merchant_id":"shop-42"}'
```

JSON body: required `image_url` (1–4096 characters), optional `provider`, optional `merchant_id`. Extra JSON fields are rejected. Multipart also accepts `image_url` instead of `file`.

URLs must use public HTTPS on port 443 without user credentials or fragments. Every resolved IP must be public; connections pin a validated address with the original TLS hostname. Redirects, private/reserved addresses, non-200 responses, and compressed transfer encodings are rejected. URL downloads have a 30-second overall limit. Signed URLs are not echoed in validation errors.

### Batch URLs

```sh
curl --fail-with-body http://localhost:8002/v1/batch \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"items":[{"image_url":"https://images.example.org/a.png"},{"image_url":"https://images.example.org/b.png"}],"concurrency":2}'
```

Each item has the same JSON fields as single extraction and can select an enabled provider. `items` must contain 1–20 entries. `concurrency` is 1–4, default 2. A service-wide `MAX_IN_FLIGHT` cap applies across all requests and batch items; A/C use 1 even when batch concurrency is larger.

### Batch uploads

```sh
curl --fail-with-body http://localhost:8001/v1/batch \
  -H "X-API-Key: $API_KEY" \
  -F 'files=@/absolute/path/a.png' -F 'files=@/absolute/path/b.png' \
  -F 'concurrency=2' -F 'merchant_id=shop-42'
```

Repeat `files` for 1–20 images. Optional `provider`, `merchant_id`, and `concurrency` apply to the whole uploaded batch. URL and upload items cannot be mixed in a single batch body.

## Responses and limits

A successful single extraction returns `{"receipt": {...}, "metadata": {...}}`. `receipt.schema_version` is `"1.0"`. The common receipt includes merchant name, receipt number, date/time, explicit currency, subtotal, discount, tax, total, line items, raw observed text, missing/uncertain fields and validation warnings. Monetary values serialize as decimal strings or `null`, never binary floating-point amounts. Line items include `raw_description`, `quantity`, `unit_price` and `line_total`. Missing values stay null rather than being invented. The OpenAPI `Receipt` schema lists every field.

Metadata varies by provider and includes observed model, token usage, latency, repair attempts and validation evidence. Local GPU responses also include measured runtime/GPU data when available. OCR errors or model output errors do not silently switch providers.

Batch returns HTTP 200 with `{"results":[...]}`. Every entry has its zero-based input `index` and either `result` with the single-extraction structure, or `error` plus `status_code`. **Inspect every entry:** a successful batch HTTP status does not imply all images succeeded. Successful entries remain available when another image fails.

```json
{"results":[{"index":0,"result":{"receipt":{"schema_version":"1.0","total":"12.50"},"metadata":{}}},{"index":1,"error":"Provider is not enabled on this service","status_code":422}]}
```

The receipt above is abbreviated; the actual success response includes all receipt fields.

| HTTP code | Meaning |
| --- | --- |
| 200 | Successful extraction, liveness/readiness, or batch envelope (possibly with per-image failures) |
| 400 | Malformed multipart request |
| 401 | Missing/invalid receipt API key |
| 413 | Total HTTP request exceeds 12 MiB, including multipart overhead |
| 422 | Invalid fields/provider/batch limits, unsafe URL, invalid image, or per-image size/pixel limit |
| 502 | Hosted/provider/OCR error; `detail` contains `error` and diagnostic `metadata` |
| 503 | Local model readiness check reports not loaded |
| 504 | An extraction exceeds 300 seconds after acquiring its inference slot |

Non-batch errors use `{"detail":"safe explanation"}` or `{"detail":{"error":"safe explanation","metadata":{...}}}`. Images are limited to 10 MiB encoded and 32 million decoded pixels each. The 12 MiB total request limit means twenty large uploads will not fit; use smaller batches or URL batches. Inference timeout does not bound time waiting for a shared inference slot or the whole batch. End-to-end batch latency depends on batch size, queueing and model speed.

Container `/health` checks prove the HTTP process is live. A/B `/ready` does not validate paid account credentials, provider availability or OCR model downloads. C warms its model at startup and `/ready` exposes its measured loaded state. Verify real extraction and every batch result before claiming a deployment works.
