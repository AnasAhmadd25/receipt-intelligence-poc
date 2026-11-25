# Receipt Intelligence benchmark

Verified on 2026-11-25. Prepared **50 held-out receipts plus 8 disjoint development receipts**; run `test10` selects **10 test receipts (6 Arabic, 4 CORD)**. Attempted and evaluated counts below show each provider's observed coverage. Unavailable requests with no model output have N/A model accuracy and JSON validity; generated invalid JSON counts as an evaluated failure. Service accuracy includes failed requests, and only providers with model output for the entire selection are ranked. Ground truth comes from dataset annotations, never model-generated labels.

The highest measured annotated scalar-field accuracy was **gemini-3.5-flash-lite (84.6%)**. This is a small POC result, not a production accuracy guarantee. Compare line-item accuracy separately below.

| Metric | OCR + Ollama Cloud | OCR + NVIDIA | Gemini Flash-Lite | Modal Hybrid |
|---|---|---|---|---|
| Selected model | gemma4:31b (blocked HTTP 402) | nvidia/nemotron-3.5-bolt-e4b | gemini-3.5-flash-lite | Qwen/Qwen3-VL-2B-Instruct |
| Attempted receipts | 0 | 10 | 10 | 10 |
| Evaluated model responses | 0 | 10 | 10 | 10 |
| Unavailable/no model output | 0 | 0 | 0 | 0 |
| Valid final receipts | 0 | 10 | 10 | 9 |
| Model field accuracy (scalar micro) | N/A | 61.5% | 84.6% | 73.1% |
| Service field accuracy (all attempts) | N/A | 61.5% | 84.6% | 73.1% |
| Arabic scalar accuracy | N/A | 75.0% | 87.5% | 81.2% |
| Non-Arabic scalar accuracy | N/A | 40.0% | 80.0% | 60.0% |
| Line-item F1 (CORD only) | N/A | 27.3% | 90.9% | 90.9% |
| JSON/schema validity, first attempt | N/A | 90.0% | 100.0% | 90.0% |
| JSON/schema validity, final | N/A | 100.0% | 100.0% | 90.0% |
| Average latency, serial end-to-end seconds | N/A | 19.026 | 5.591 | 63.327 |
| Median latency seconds | N/A | 18.792 | 6.130 | 56.998 |
| OCR average seconds | N/A | 9.173 | N/A | N/A |
| API/client average seconds | N/A | 9.853 | 5.591 | 63.327 |
| CPU core-seconds/receipt | N/A | 10.810 | 1.054 | 60.350 |
| Peak RAM MiB | N/A | 5,235.047 | 162.562 | 7,798.000 |
| GPU allocated peak MiB | N/A | N/A | N/A | 4,564.434 |
| Receipts/minute, serial measured stages | N/A | 3.154 | 10.732 | 0.947 |
| Usage-based cost for evaluated test set USD (estimate) | N/A | N/A | 0.02025 | 0.14996 |
| Input tokens incl repairs | N/A | 36857 | 12356 | 11388 |
| Output tokens incl reasoning/repairs | N/A | 8024 | 6619 | 5547 |
| Thinking tokens (subset of output) | N/A | 0 | 0 | 0 |
| JSON repairs | N/A | 1 | 0 | 0 |
| Projected inference cost for 50 USD (estimated) | N/A | N/A | 0.10127 | 0.74978 |
| Projected inference cost for 1000 USD (estimated) | N/A | N/A | 2.02543 | 14.99564 |
| Recommended cloud machine / observed Modal allocation | 2vCPU/4GiB Linux, OCR concurrency 1 (estimate) | 2vCPU/4GiB Linux, OCR concurrency 1 (estimate) | 1vCPU/2GiB CPU proxy (estimate) | T4 + 2.0 physical CPU + 8GiB (us) |
| Monthly infrastructure USD | 24 + tokens (DO example) | 24 + unknown API price | 12 + tokens (DO example) | Scale-zero variable; no fixed monthly GPU |

## Annotated field denominators

| Field | OCR + Ollama Cloud | OCR + NVIDIA | Gemini Flash-Lite | Modal Hybrid |
|---|---|---|---|---|
| merchant_name | N/A (0/0) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) |
| purchase_date | N/A (0/0) | 100.0% (3/3) | 100.0% (3/3) | 100.0% (3/3) |
| receipt_number | N/A (0/0) | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| total | N/A (0/0) | 75.0% (6/8) | 87.5% (7/8) | 75.0% (6/8) |
| subtotal | N/A (0/0) | 33.3% (2/6) | 83.3% (5/6) | 83.3% (5/6) |
| tax | N/A (0/0) | 20.0% (1/5) | 80.0% (4/5) | 60.0% (3/5) |
| discount | N/A (0/0) | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| line_items.raw_description | N/A (0/0) | 45.5% (5/11) | 100.0% (11/11) | 36.4% (4/11) |
| line_items.quantity | N/A (0/0) | 37.5% (3/8) | 87.5% (7/8) | 87.5% (7/8) |
| line_items.unit_price | N/A (0/0) | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) |
| line_items.line_total | N/A (0/0) | 36.4% (4/11) | 90.9% (10/11) | 90.9% (10/11) |

## Missing fields and hallucination limits

| Provider | Missing annotated-field rate | Count | Hallucination rate | Known-absent labels |
|---|---|---|---|---|
| ocr_ollama | N/A | 0/0 | N/A | 0/0 |
| ocr_nvidia | 11.5% | 3/26 | N/A | 0/0 |
| gemini | 0.0% | 0/26 | N/A | 0/0 |
| modal_hybrid | 3.8% | 1/26 | N/A | 0/0 |

Hallucination is **unscorable** where the dataset supplies no explicit known-absent fields. Unannotated predictions are retained as unscorable, never automatically called hallucinations. Wrong annotated values are accuracy errors. Raw predictions and per-receipt metrics remain in `results/test10/`.

## Interpretation and reproducibility

- AraReceipt is Arabic/mixed Arabic-English with region classes and no reliable product-row grouping. CORD is Indonesian/Latin-script with mixed English terms, not an English-only benchmark.
- Missing annotations are excluded. CORD merchant/date/currency are unscorable. Full-line F1 requires one-to-one matching, description similarity ≥0.8 and equality of each annotated numeric field; diagnostic field accuracy pairs descriptions one-to-one independently of amounts.
- Field accuracy is micro-averaged annotated scalar fields; the Arabic/non-Arabic columns have different coverage and are not equivalent difficulty.
- Text-provider latency adds the same cached measured OCR stage to live API latency. Cached OCR is shared byte-for-byte across Ollama/NVIDIA. OCR and API resource profiles are separate stages; reported RAM is their maximum, not an invented sum. API waiting CPU is not provider inference CPU.
- JSON validity is schema validity of the original response; final validity includes any permitted repair. All attempts and usage are retained.
- Costs are actual reported usage multiplied by official list rates, **not verified cash debits/invoices**. NVIDIA commercial rates are unavailable. Modal allocated-seconds estimates include observed container initialization, inference and 2s terminal idle; scheduler/pre-Python startup may be missing.
- CPU/RAM measurements are local Apple M1 Pro unless explicitly marked Modal Linux. GPU values follow recorded remote allocation metadata; missing GPU measurements remain N/A. Concurrency evidence is below.
- Dataset licenses/annotation conversion limitations: [dataset audit](dataset_audit.md). Provider/model/pricing references: [sources](sources.md).

## Development OCR/resource experiments

| CPU OCR | Concurrency | Completed | Annotated token recall | Numeric-token recall | Receipts/min | Peak MiB | Core-sec/receipt |
|---|---|---|---|---|---|---|---|
| tesseract | 1 | 8 | 43.9% | 49.3% | 32.254 | 414.652 | 1.804 |
| paddle | 1 | 8 | 80.4% | 72.9% | 8.764 | 5,129.250 | 6.588 |
| paddle | 2 | 8 | 80.4% | 72.9% | 8.696 | 5,464.266 | 6.629 |
| paddle | 4 | 8 | 80.4% | 72.9% | 8.607 | 5,321.969 | 6.698 |

These are partial-transcription recall proxies, not full CER. The selected PP-OCRv5 route uses the known language hint for Arabic vs English/Latin recognition. API deployments expose `OCR_LANGUAGE`; automatic script routing is not validated. Paddle adapter locks its model instances, so concurrency increases queueing; no independent multi-worker speedup is claimed. EasyOCR was unnecessary after a working higher-recall candidate was found.

## Live API-stage concurrency measurements

| Provider | Concurrency | Requests | Valid receipts | Batch seconds | Actual batch rpm | Local peak MiB | Local CPU% (100=one core) |
|---|---|---|---|---|---|---|---|
| ocr_nvidia | 1 | 10 | 10 | 98.563 | 6.087 | 2,417.016 | 19.014 |
| gemini | 1 | 10 | 10 | 55.991 | 10.716 | 162.562 | 18.817 |
| modal_hybrid | 1 | 10 | 9 | 633.315 | 0.947 | 97.250 | 21.457 |
| gemini | 2 | 10 | 10 | 23.618 | 25.404 | 167.797 | 17.276 |
| gemini | 4 | 10 | 10 | 15.216 | 39.433 | 215.562 | 24.019 |

Text-provider API-stage results exclude cached OCR; their serial full-pipeline latency above adds its measured OCR stage. Gemini has no OCR stage. Concurrency runs use the frozen ten-receipt set and are throughput trials, not prompt tuning. All retries remain in their run files; missing usage after server errors is unknown, not assumed free.

## Acceptance and operating decision

See [cost per acceptable receipt](acceptance.md) for the strict all-annotated-fields rule and its actual pass counts. The API default is NVIDIA as requested; the client can select Gemini without installing OCR. NVIDIA developer access is a temporary experimental price basis, not a known commercial price. Ollama is implemented but inference is blocked by account credits (HTTP402), so it has no held-out accuracy result. [Ollama pricing](https://ollama.com/pricing) currently lists pay-as-you-go; no usable free starter allowance was confirmed for this account.

Additional fixed-model trials and quality-adjusted pricing are in [model variants](model_variants.md). The separately tested1.7Bhybrid fallback produced0/6development scalar fields correct and was rejected; the held-out Modal column remains the direct2BVLM.
