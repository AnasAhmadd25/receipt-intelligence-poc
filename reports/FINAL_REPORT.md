# Receipt Intelligence: final R&D results

25 November 2026. Repository: https://github.com/AnasAhmadd25/receipt-intelligence-poc . This revision was developed in a separate local copy; the original checkout was preserved.

## Recommendation

Use **direct Gemini 3.5 Flash-Lite** for the highest measured quality. Direct **Gemini 3.1 Flash-Lite** has the lowest measured paid model cost per strictly acceptable receipt. Adding OCR before a cheap Gemini text call improved on the tested NVIDIA text model, but did not beat direct Gemini on acceptable-result cost or item extraction. NVIDIA remains service A's requested default; `ocr_gemini` provides a text stage with a published token price.

## Same-image results

| Approach | Correct scalar fields | Item micro F 1 | Strict passes | Cost for all ten attempts | Cost per 1, 000 accepted |
|---|---:|---:|---:|---:|---:|
| Direct Gemini 3.5 Flash-Lite | 22/26 (84.6%) | 90.9% | 6/9 | $0.020254 | $3.38 |
| Direct Gemini 3.1 Flash-Lite | 19/26 (73.1%) | 72.7% | 5/9 | $0.013695 | $2.74 |
| **PaddleOCR + Gemini 3.1 text-only, new** | **20/26 (76.9%)** | **27.3%** | **4/9** | **$0.016780 tokens** | **$4.20 tokens; OCR hosting extra** |
| PaddleOCR + NVIDIA Nemotron 3.5 Bolt E4B | 16/26 (61.5%) | 27.3% | 2/9 | Free developer endpoint | Commercial price unavailable |
| PaddleOCR + OpenRouter free Nemotron Nano Omni | 17/26 (65.4%) | 0% | 2/9 | $0 within tested quota | $0 trial inference; hosting extra |
| Original self-hosted Qwen3-VL-2B, Modal T4 | 19/26 (73.1%) | 90.9% | 4/9 | $0.149956 | $37.49 |
| Ollama Cloud Gemma 4 | Unmeasured | Unmeasured | Unmeasured | HTTP 402 inference | Unavailable |

Strict acceptance requires valid final JSON, every annotated scalar correct, and item F 1=1 wherever grouped items are annotated. One image has no scorable target fields, leaving nine strict-scorable receipts. Its cost still counts. Costs divide all attempted-input spending by verified passes, including unsuccessful model outputs and repairs. These are usage/list-price estimates, not invoices. Token fees exclude server rent, OCR CPU hosting and review.

The new text-only trial reported 26, 560 input and 6, 760 output tokens, complete usage, and 10/10 schema-valid final receipts. Arabic scalar accuracy was 12/16; CORD scalar accuracy was 8/10. Mean text-call wall time was 3.84s. Adding original cached OCR timing gives 13.01s, a reconstructed estimate rather than a new end-to-end measurement. Evidence: `results/ocr_structuring_test/paddle/` and `results/ocr_structuring_comparison.json`.

Original direct Gemini latency was 5.59s for 3.5 and 4.66s for 3.1. Original Qwen latency was 63.33s. The new Qwen container has additional prompt constraints and one bounded schema repair; its runtime smoke test does not replace the historical ten-image quality score.

## Stronger OCR and cheap structuring

| Development experiment | Correct scalar fields | Strict passes | Text token cost for eight |
|---|---:|---:|---:|
| Original PaddleOCR boxes + Gemini 3.1 | 14/20 | 4/8 | $0.009849 |
| Geometry-grouped rows + same model | 13/20 | 4/8 | $0.011928 |
| Partial human transcript control + same model | 14/20 | 4/8 | $0.007009 |

The transcript control used public annotated OCR text, never target extraction JSON. Transcripts are partial, so this is diagnostic, not an OCR system or an accuracy ceiling. Price separators and total/subtotal associations remained problems. Stronger recognition alone is not established as sufficient. Row grouping retained original boxes but added tokens without improving this development score; it stays disabled.

Initial parallel requests hit HTTP 429 limits and are retained separately. The completed comparison ran sequentially at 15-second intervals. An explicit schema-version instruction addressed the observed `1.0.0` versus `1.0` error. Failures are not silently scored as valid results.

**PaddleOCR-VL-1.6** was selected for the stronger trial: pinned 0.9B-class checkpoint, Apache 2.0, element-level Transformers OCR. The L4 attempt produced no worker/HTTP result before its bounded stop. A T4 FP16 compatibility attempt returned HTTP 500. **No validated OCR output or accuracy result was obtained**, so it was not promoted into an API image. This does not assess its official full page-parsing pipeline. Configuration, failures and cleanup evidence are in `results/ocr_vl16_dev/`.

DeepDoc is RAGFlow's document-analysis system; DeepSeek-OCR is a separate model family. Research also covered dots.mocr, Surya, MinerU and Docling. General document benchmarks do not establish Arabic receipt-photo accuracy. See [the source-linked research memo](../docs/OCR_RESEARCH.md) for licensing, hardware and Arabic evidence. Researched-only models have no fabricated results.

Earlier actual trials included GLM-OCR (1/6 development scalars), HunyuanOCR (invalid receipt schema), and PaddleOCR + self-hosted Qwen3-1.7B (0/6 development scalars). Direct Qwen3-VL-2B was retained as the self-hosted candidate. Free OpenRouter Gemma routes were rate-limited; Ollama listed models but inference required billing. Historical reports retain the raw evidence.

## Three built container images

| Local image tag | Architecture | Verification | Interactive docs |
|---|---|---|---|
| `receipt-intelligence:ocr-rd` | Linux amd64 | Real Paddle+NVIDIA upload 200; schema valid; 82.48s cold including weights, under Mac emulation | http://localhost:8001/docs |
| `receipt-intelligence:gemini-rd` | Linux arm64 | Real Gemini 3.5 upload 200; schema valid; 3.62s | http://localhost:8002/docs |
| `receipt-intelligence:selfhosted-rd` | Linux amd64 | Docker built; local packaging passed; same API/Dockerfile on Modal T4 returned real validated receipt in 56.83s | Hosted docs below; local 8003 needs NVIDIA CUDA |

All services expose uploads, secure HTTPS image URLs, batches, health, readiness, Swagger, ReDoc and OpenAPI with the same Receipt schema. Swagger supports API-key/Bearer authorization. A/B containers remain running locally. The third image needs an NVIDIA GPU; this Mac's Docker Desktop cannot provide CUDA. Local C packaging correctly returned 503 readiness when GPU warmup was explicitly disabled for that test, then the temporary container was removed.

Verified GPU API documentation:

https://anasahmadoff--receipt-intelligence-selfhosted-api-selfho-76fa77.us-east.modal.run/docs

Extraction uses the existing `MODAL_API_KEY`; documentation does not grant inference access. Missing authentication returned 401. Health/readiness/docs/OpenAPI returned 200, and readiness identified a real Tesla T4, FP16 and SDPA. The service has at most one worker and scales to zero. Cold requests can initialize the GPU model.

Images are built locally, **not uploaded to a container registry**. GitHub contains all three Dockerfiles and Compose, so another user can build/start with `docker compose up --build -d ocr gemini`; use `--profile gpu` for selfhosted on NVIDIA Linux. Weights download on first use and persist in volumes. Credentials stay outside build contexts. Modal ignores Docker USER/HEALTHCHECK/CMD and serves the common ASGI application, so remote inference is distinguished from local Docker packaging.

See [API documentation](../docs/API.md), `results/container_images.json` and `results/container_smoke_*.json` for exact image IDs, requests and limitations.

## Resources, budget and verification

- Original Linux Paddle measurements support starting near 2vCPU/4GiB with one active request; queued concurrency did not scale throughput proportionally. Validate sizing under real load.
- Direct Gemini needs no local OCR or GPU. Original Qwen observed about 4.46GiB GPU allocation and 7.6GiB process RSS; allow RAM headroom beyond the tight 8GiB experiment allocation.
- Cumulative budget ledger: **$4.090403 reserved or estimated against the original $5 cap**. This includes unused conservative reservations and incomplete-usage failures, not actual billed spending. It was not reset for this follow-up.
- Only ten of fifty prepared test images were evaluated. AraReceipt lacks grouped Arabic item truth. CORD is Indonesian/Latin, not English-only. Incomplete transcription and known-absence labels prevent full CER and hallucination-rate claims.

**83 tests passed**, covering schema, metrics, security, batches, OpenAPI, authentication, provider restrictions, GPU readiness and text-only model input. All 58 original images and annotations remain included. The original working copy was preserved; secrets are scanned before publication.
