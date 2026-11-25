# Receipt Intelligence — measured R&D POC

Compare receipt extraction architectures using the **same images, schema, ground truth and acceptance rules**. The aim is to find an affordable model that produces useful results, with enough evidence to estimate operating costs.

**Recommendation:** start with **Gemini 3.5 Flash-Lite** for quality. **Gemini 3.1 Flash-Lite** has the lowest measured paid inference cost per verified acceptable receipt. The tested free text models and self-hosted service produced fewer acceptable receipts.

This repository includes the code, **58 receipt images**, original and standardized annotations, raw model responses, usage records, resource profiles and reports. No API keys, local environments or GPU weights are included.

## Results at a glance

Measured on **10 fixed held-out images**, not all 50 prepared test images. One image has no scorable target fields, leaving **9 scorable receipts** for strict acceptance. Prices were checked on **25 November 2026**.

| Pipeline / model | Scalar accuracy | Line-item F1 | Strict passes | Mean end-to-end latency | Inference or compute cost for all 10 | Cost / 1,000 strict accepted |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash-Lite, direct image | **84.6%** (22/26) | **90.9%** | **6/9** | **5.59 s** | $0.02025 estimated | $3.38 estimated |
| Gemini 3.1 Flash-Lite, direct image | 73.1% (19/26) | 72.7% | 5/9 | 4.66 s | $0.01370 estimated | **$2.74 estimated** |
| PaddleOCR + NVIDIA Nemotron 3.5 Bolt E4B | 61.5% (16/26) | 27.3% | 2/9 | 19.03 s | Free developer trial; commercial price N/A | N/A commercial |
| PaddleOCR + OpenRouter free Nemotron Nano Omni | 65.4% (17/26) | 0% | 2/9 | 30.93 s | $0 within tested free quota | $0 trial inference; hosting extra |
| Modal T4 + Qwen3-VL-2B | 73.1% (19/26) | **90.9%** | 4/9 | 63.33 s | $0.14996 estimated | $37.49 estimated |
| PaddleOCR + Ollama Cloud Gemma 4 | N/A | N/A | Not evaluated | N/A | Inference blocked: HTTP 402 | N/A |

**Strict acceptance** requires a valid final receipt, every annotated scalar field correct, and grouped line-item F1 exactly 1 where grouping is annotated. Unannotated fields are excluded. A pass therefore certifies only covered annotations. Costs include spending on failed/invalid outputs and repairs, divided by verified passes.

These are **usage × list-price estimates, not verified invoices**. Token/GPU figures exclude API-proxy hosting and human review. Free quota is not an unlimited commercial price. The small sample does not establish production accuracy.

For a review-cost sensitivity calculation, the extra 3.5 model fee is about **$0.66 per 1,000 attempts**. On this sample it produced one more strict pass per ten attempts; an additional review costing more than about **$0.0066** would outweigh the token savings from 3.1. This is an illustrative scenario, not an assumed review wage.

## Architecture

All paths produce the same nullable, decimal-safe `Receipt` schema.

```mermaid
flowchart LR
    I[Receipt image] --> A[Local PP-OCRv5 CPU]
    A --> L[OCR text and bounding boxes]
    L --> N[NVIDIA hosted text model]
    L --> O[OpenRouter fixed free text model]
    L --> OL[Ollama Cloud text model]
    I --> G[Gemini Flash-Lite image extraction]
    I --> M[Modal T4: Qwen3-VL-2B]
    N --> V[Pydantic validation]
    O --> V
    OL --> V
    G --> V
    M --> V
    V --> R[Shared receipt JSON and evidence metadata]
    classDef selected fill:#dcfce7,stroke:#15803d,color:#14532d;
    class G selected;
```

The **API default remains NVIDIA**, as requested during the experiments. Select `provider=gemini` to use the recommended architecture. Ollama always uses the hosted API; no local Ollama inference model was downloaded.

The separately tested self-hosted fallback was **PaddleOCR + Qwen3-1.7B**. It produced 0/6 correct development scalar fields and was rejected. It is not silently substituted into the direct Qwen VLM results.

## Benchmark workflow

```mermaid
flowchart TD
    D[Public datasets and pinned revisions] --> P[Deterministic sampling and duplicate checks]
    P --> DEV[8 development receipts]
    P --> TEST[50 held-out test receipts saved]
    DEV --> S[Choose OCR, models and prompts]
    S --> F[Freeze configuration]
    TEST --> T[Fixed initial 10-image subset]
    F --> RUN[Run each available architecture]
    T --> RUN
    RUN --> ACC[Annotation-aware fields and one-to-one item matching]
    RUN --> RES[Latency, CPU, RAM, GPU and token usage]
    ACC --> COST[Cost per verified acceptable receipt]
    RES --> COST
    TEST -. later command .-> FULL[Full 50-image evaluation: not run]
```

The development/test split and initial test IDs are in [`data/manifest.json`](data/manifest.json). Predictions and OCR caches record image hashes and configuration fingerprints; use a new run name after changing a model or prompt.

## Data and example images

| Dataset | Held-out test | Development | What can be scored |
|---|---:|---:|---|
| AraReceipt | 30 | 5 | Arabic/mixed Arabic-English region labels; unambiguous scalar fields |
| CORD v2 | 20 | 3 | Indonesian/Latin-script receipt totals and grouped product fields |
| **Total unique images** | **50** | **8** | **58 unique file hashes and decoded-image hashes** |

AraReceipt does **not** provide authoritative product-row grouping. CORD is **not an English-only dataset** and lacks annotated merchant/date/currency fields. Full OCR transcripts and explicit known-absent fields are unavailable, so full CER and hallucination rates remain N/A. Do not interpret these results as a complete English-only or Arabic line-item validation.

<table>
<tr><th>Arabic example — AraReceipt</th><th>Latin-script example — CORD</th></tr>
<tr>
<td><img src="data/images/ara_0084.jpg" width="260" alt="Arabic receipt from AraReceipt"></td>
<td><img src="data/images/cord_test_0014.jpg" width="260" alt="CORD receipt with Latin-script product rows"></td>
</tr>
</table>

- [All images](data/images/)
- [Original annotations](data/original_annotations/)
- [Standardized ground truth](data/annotations/)
- [Dataset audit and coverage](reports/dataset_audit.md)
- [Attribution and dataset licenses](data/ATTRIBUTION.md): AraReceipt MIT; CORD CC BY 4.0. Those licenses and attribution remain applicable to bundled data.

Ground truth was converted from the actual dataset annotations. No LLM generated evaluation ground truth. CORU was researched but not combined with AraReceipt, avoiding their known ReceiptSense overlap.

## Repository map

```text
app/                  FastAPI, schema, validation, provider adapters, URL protection
benchmark/            Data preparation, metrics, profiling, evaluation and reports
deploy/               Modal ASGI service and CPU/development experiment drivers
configs/              Pricing, model research, weight hashes, dependency snapshot, Dockerfiles
data/                 Images, original labels, standardized labels and manifests
results/              Raw predictions, OCR caches, usage, profiles and CSVs
reports/              Measured comparisons, cost analysis and sizing recommendations
tests/                Schema, metrics, security, API, provider and reporting tests
.env.example          Configuration names and safe placeholders
```

## Quick start

Use Python 3.11 or newer; the measured local environment was native ARM64 Python 3.12.13 on an M1 Pro with 16 GB RAM.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[gemini,test]'
cp .env.example .env
```

For **Gemini only**, no OCR, PyTorch or GPU packages are needed. Set `DEFAULT_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`.

For the measured **PaddleOCR text-model paths**:

```bash
uv pip install -e '.[ocr]'
# Set OCR_ENGINE=paddle and OCR_LANGUAGE=ar or en in .env.
```

The benchmark passes corpus language hints to the Arabic/English recognizers. Automatic script routing is not validated. The Arabic model also contains Latin characters, but that does not substitute for testing English-only data.

For the lighter **Tesseract alternative**:

```bash
uv pip install pytesseract
# macOS:
brew install tesseract tesseract-lang
# Debian/Ubuntu alternative:
# sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng
tesseract --version
tesseract --list-langs
```

Set `OCR_ENGINE=tesseract`. The included `data/tessdata` contains Arabic/English language data; `TESSDATA_PREFIX` can point to a system language directory instead.

### Configure secrets without displaying them

```bash
python - <<'PY'
from getpass import getpass
from dotenv import set_key
for name in ('GEMINI_API_KEY', 'NVIDIA_API_KEY', 'OPENROUTER_API_KEY', 'OLLAMA_API_KEY'):
    value = getpass(f'{name} (Enter skips): ')
    if value:
        set_key('.env', name, value)
PY
chmod 600 .env
```

Do not commit `.env`. The published repository excludes credential values and local model caches.

Verified model identifiers used here:

```dotenv
DEFAULT_PROVIDER=ocr_nvidia
OCR_ENGINE=paddle
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
GEMINI_MODEL=gemini-3.5-flash-lite
OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
OLLAMA_MODEL=gemma4:31b
```

The listed NVIDIA Gemma 3 4B model returned account-specific HTTP 404, so the successful 8B Nemotron route was selected. Ollama model listing worked but inference required credits. The free OpenRouter Gemma 4 routes were upstream-rate-limited; the available fixed Nano route was tested. Multiple keys do not multiply OpenRouter's account quota. See [availability and pricing research](reports/openrouter_options.md).

## Reproduce the experiments

The data is already included. Preparing it again downloads pinned source samples and validates revisions:

```bash
python -m benchmark.prepare_data

# Development OCR comparison; add --concurrency 2 or 4 for resource trials.
python -m benchmark.ocr_benchmark --engine tesseract --concurrency 1
python -m benchmark.ocr_benchmark --engine paddle --concurrency 1

# Development-only model selection. Use a fresh run name.
python -m benchmark.run --providers ocr_nvidia gemini --split development --limit 8 --run-name my_dev

# Same fixed ten held-out images, after freezing choices.
python -m benchmark.run --providers ocr_nvidia gemini modal_hybrid --limit 10 --run-name my_test10

# Optional free-model comparison.
python -m benchmark.run --providers ocr_openrouter --limit 10 --run-name my_openrouter10

# Cheaper Gemini model, kept as a separate experiment.
GEMINI_MODEL=gemini-3.1-flash-lite python -m benchmark.run --providers gemini --limit 10 --run-name my_gemini31

# Full fifty-image evaluation later; this has not been run.
python -m benchmark.run --providers gemini --limit 50 --run-name my_test50
```

The initial paid experiment limit is **$5 cumulatively**. The spending ledger reserves conservative bounds, including unknown usage after server errors. A later command can require additional budget authorization even when its mean-cost projection is small; do not delete/reset the ledger to bypass it. Modal commands must also have a bounded resource reservation.

Every text provider uses the same saved OCR for a given image/configuration. Hosted text clients allow at most one JSON repair. Gemini uses native structured output plus Pydantic validation and at most one explicit retry/repair; all attempts remain recorded. Thinking tokens are already counted once in `output_tokens`.

### Existing runs

| Folder | Contents |
|---|---|
| `results/test10/` | NVIDIA, Gemini 3.5, OpenRouter and direct Modal comparison |
| `results/test10_gemini31/` | Cheaper Gemini 3.1, same test IDs |
| `results/test10_c2/`, `results/test10_c4/` | Gemini throughput trials with frozen prompts |
| `results/development_*/` | Development model trials and repair diagnostics |
| `results/ocr_development/` | Tesseract/Paddle development and concurrency profiles |
| `results/modal_dev_*.json` | Self-hosted candidate raw outputs and resource evidence |
| `results/modal_linux_*.json` | Actual x86 Linux CPU OCR trials |

## API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
# Estimated 4 GB Linux OCR machine: limit active requests.
# MAX_IN_FLIGHT=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | None | Status and schema version |
| `POST /v1/extract` | Multipart image or JSON `image_url`, provider, optional merchant ID | `{receipt, metadata}` |
| `POST /v1/batch` | Uploaded files or URL items, bounded concurrency | Ordered results with per-item failures |

Provider IDs: `ocr_nvidia`, `ocr_ollama`, `gemini`, `modal_hybrid`, `ocr_openrouter`.

```bash
curl --fail http://127.0.0.1:8000/v1/extract \
  -F provider=gemini -F file=@data/images/ara_0084.jpg

curl --fail http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"provider":"gemini","image_url":"https://your-image-host.example/receipt.jpg"}'

curl --fail http://127.0.0.1:8000/v1/batch \
  -F provider=gemini -F concurrency=2 \
  -F files=@data/images/ara_0084.jpg \
  -F files=@data/images/cord_test_0014.jpg
```

Images are limited to **10 MiB and 32 megapixels**; complete HTTP bodies are limited to 12 MiB. Batches accept 1–20 images, with concurrency 1–4 and a configurable global request cap. All 58 included images pass the API input policy.

For public hosting, set `PUBLIC_DEPLOYMENT=true` and a nonempty `API_KEY`. Supply `X-API-Key` or `Authorization: Bearer`. HTTPS URL fetching checks every DNS result, rejects private/reserved addresses and redirects, pins the connection to a validated IP, checks the original TLS hostname, and enforces transfer limits. The live upload route and DNS-pinned HTTPS fetching were exercised in addition to unit tests.

A returned JSON object is not proof of extraction accuracy. Inspect warnings and per-item batch failures; real-world acceptance needs a client-reviewed policy.

## Modal deployment

```bash
uv pip install -e '.[modal]'
modal profile current
# If authentication is unavailable: modal token new
```

Set `MODAL_API_KEY` locally, then deploy the selected direct model:

```bash
MODAL_MODEL_ID=Qwen/Qwen3-VL-2B-Instruct modal deploy deploy/modal_app.py
```

Set `MODAL_ENDPOINT` to the emitted service URL, with or without `/v1/extract`. The service has bearer protection, pinned model revisions, persistent weights, scale-to-zero, one maximum container, and a 2-second idle window.

The measured direct deployment used **T4, FP16/SDPA, 2 physical CPU cores and 8 GiB RAM**, with a 1-megapixel model input cap. Its approximately 7.6 GiB observed RSS suggests about **12 GiB RAM** for a further production trial with headroom. Native weights fit the T4; L4 and CPU-only VLM performance were not measured.

The direct GPU service uses base64 image JSON internally. The shared FastAPI layer provides normal uploads, URL fetching and interchangeable providers. Self-hosted inference calls **no paid external inference APIs**.

Development drivers and the separately tested fallback are available:

```bash
python -m deploy.run_modal_dev_http --model Qwen/Qwen3-VL-2B-Instruct --limit 4

MODAL_MODEL_ID=Qwen/Qwen3-1.7B MODAL_MEMORY_MIB=12288 \
  MODAL_APP_NAME=receipt-intelligence-hybrid-fallback modal deploy deploy/modal_app.py
```

The fallback deployment was stopped after its poor development result. The direct service was verified idle with zero running containers. Deploying or calling either endpoint can consume the configured experiment budget.

## Resource and cloud sizing findings

- **Local OCR:** Paddle recovered about 80% of annotated text tokens versus 44% for Tesseract, with higher RAM/latency. Partial transcription prevents a full CER claim.
- **Actual Linux OCR:** Paddle processed approximately 7.7–8.6 receipts/minute. Peak RSS was about 1.9 GiB at concurrency 1 and 3.2 GiB at concurrency 4. The model lock serialized inference, so extra queued requests raised memory without a parallel speedup.
- **CPU OCR server:** estimated 2 vCPU / 4 GiB at active concurrency 1; example prices $24/month DigitalOcean or €16.99/month Hetzner. Target-host performance remains estimated beyond the measured Modal Linux probe.
- **Gemini proxy:** estimated 1 vCPU / 2 GiB; example $12/month. Measured local client throughput was about 10.7, 25.4 and 39.4 receipts/minute at concurrency 1, 2 and 4. API capacity/retries still matter.
- **Self-hosting:** the measured warm variable compute cost exceeded Gemini's token price, with fewer strict acceptable results. No positive volume break-even appeared for these settings. Optimization or different models could change that conclusion and would require new measurements.

See [infrastructure recommendations](reports/infrastructure_recommendation.md) for plan comparisons, CPU/RAM headroom, capacity and price scope.

## Reports, CSVs and verification

```bash
python -m benchmark.report
python -m benchmark.acceptance --run-name test10
python -m benchmark.variant_report
python -m benchmark.modal_report
pytest -q
```

Run `variant_report` after `report` to add the cheaper-model/review-cost sensitivity section. Docker configurations are supplied for Gemini-only and Tesseract-only services; builds were not run because the local Docker daemon was stopped. Use `OCR_ENGINE=tesseract` with the Tesseract image.

| Deliverable | Purpose |
|---|---|
| [Model and cost comparison](reports/model_variants.md) | Paid/free models and strict acceptable-result costs |
| [Main benchmark](reports/benchmark.md) | Required architecture comparison and field denominators |
| [Acceptance report](reports/acceptance.md) | Per-receipt pass/fail decisions and coverage limits |
| [Cost analysis](reports/cost_analysis.md) | Usage, projections, arrival scenarios, review-cost sensitivity |
| [Infrastructure](reports/infrastructure_recommendation.md) | Measured resources and conditional machine sizing |
| [Modal experiments](reports/modal_experiments.md) | GLM, Hunyuan, Qwen and fallback development evidence |
| [Dataset audit](reports/dataset_audit.md) | Licensing, conversion policy, coverage and duplicate checks |
| [Sources](reports/sources.md) | Official dated model/cloud price references |
| [Benchmark CSV](results/benchmark.csv) | Per-receipt measured results |
| [Model variants CSV](results/model_variants.csv) | Same-image model comparison |
| [Cost projections CSV](results/cost_projections.csv) | Estimated costs for larger workloads |
| [Resource profiles CSV](results/resource_profiles.csv) | Local and remote process/GPU profiles |

Final verification: **61 tests passed**. Source and report files were scanned against configured credential values before publication. Actual invoices were not accessed, and unmeasured metrics remain N/A.
