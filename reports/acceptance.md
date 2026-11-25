# Cost per acceptable receipt

Run `test10`; split `test`; 10 fixed selected inputs per provider. Counts below use only saved predictions; missing providers remain unmeasured. These are explicit experimental acceptance rules, not a client-approved production SLA.

**Strict:** valid final Pydantic receipt with status `ok`, every annotated scalar correct, and line-item F1 exactly 1 when ground truth supports grouped items. At least one annotated scalar or product field is required. Unannotated fields remain unscorable; passing a receipt with only its total annotated does not certify its merchant, date, tax, or products.

**Critical-total sensitivity:** valid final receipt plus correct annotated receipt total. If total is not annotated, the rule is N/A. This deliberately weaker rule does not establish item or other-field quality. Unavailable providers without generated output are N/A; generated invalid output fails.

| Provider | Saved/selected | Strict passes/scorable | Critical-total passes/scorable | All-input inference cost, economic estimate | Cost/strict accepted | Cost/critical-total accepted |
|---|---|---|---|---|---|---|
| ocr_ollama | 0/10 | 0/0 | 0/0 | N/A | N/A | N/A |
| ocr_nvidia | 10/10 | 2/9 | 6/8 | N/A | N/A | N/A |
| gemini | 10/10 | 6/9 | 7/8 | $0.020254 | $0.003376 | $0.002893 |
| modal_hybrid | 10/10 | 4/9 | 6/8 | $0.149956 | $0.037489 | $0.024993 |
| ocr_openrouter | 10/10 | 2/9 | 6/8 | $0.000000 | $0.000000 | $0.000000 |

Cost per accepted receipt divides **all saved input inference cost, including generated failures and repairs**, by accepted outputs. It does not discard spending on rejected receipts. Unknown prices or usage produce N/A; zero accepted receipts means no acceptable output and no finite cost per accepted receipt. Raw attempt metadata supplies hosted token costs; the comparison aggregate can supply Modal lifecycle estimates when its model and counts match this run. Regenerate `python -m benchmark.report --run-name test10` before this command to keep lifecycle estimates current.

Gemini uses paid list-price inference even if the account has promotional credits; thoughts/retries are included once in recorded usage. NVIDIA developer inference is treated separately as an **estimated $0 experimental inference debit** under the tested free developer access; commercial inference price is unknown, so its economic cost per accepted receipt remains N/A. This does not make local OCR free: monthly infrastructure, measured OCR compute, operations and human review are excluded here and must be added for total operating cost. No invoice debit is asserted without billing evidence. See [cost analysis](cost_analysis.md).

Partial runs cannot establish a complete ten- or fifty-input acceptance result. Do not rank a provider on a smaller/easier saved subset. Full annotation coverage is also limited: AraReceipt has no reliable grouped product ground truth, whereas CORD has item annotations. The strict and total-only denominators therefore answer different questions. A new commercial acceptance threshold must be selected with the client before promoting these POC results.

| Provider | Strict cost qualification | Total-rule cost qualification |
|---|---|---|
| ocr_ollama | no acceptable receipts | no acceptable receipts |
| ocr_nvidia | unknown price, usage, or allocated lifecycle cost | unknown price, usage, or allocated lifecycle cost |
| gemini | all input inference cost divided by accepted receipts | all input inference cost divided by accepted receipts |
| modal_hybrid | all input inference cost divided by accepted receipts | all input inference cost divided by accepted receipts |
| ocr_openrouter | all input inference cost divided by accepted receipts | all input inference cost divided by accepted receipts |

## Per-receipt decisions

| Provider | Receipt | Strict | Critical total | Reason |
|---|---|---|---|---|
| ocr_nvidia | ara_0084 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| ocr_nvidia | ara_0018 | None | None | all relevant fields unannotated; no quality pass can be claimed |
| ocr_nvidia | ara_0050 | False | True | annotated scalar error |
| ocr_nvidia | ara_0033 | True | None | passes covered annotations only; unannotated fields remain unscorable |
| ocr_nvidia | ara_0022 | False | True | annotated scalar error |
| ocr_nvidia | ara_0064 | False | True | annotated scalar error |
| ocr_nvidia | cord_test_0014 | False | False | annotated scalar error; grouped line-item F1 below 1 |
| ocr_nvidia | cord_test_0027 | False | True | annotated scalar error |
| ocr_nvidia | cord_test_0065 | False | False | annotated scalar error; grouped line-item F1 below 1 |
| ocr_nvidia | cord_test_0002 | False | True | grouped line-item F1 below 1 |
| gemini | ara_0084 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| gemini | ara_0018 | None | None | all relevant fields unannotated; no quality pass can be claimed |
| gemini | ara_0050 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| gemini | ara_0033 | False | None | annotated scalar error |
| gemini | ara_0022 | False | True | annotated scalar error |
| gemini | ara_0064 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| gemini | cord_test_0014 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| gemini | cord_test_0027 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| gemini | cord_test_0065 | False | False | annotated scalar error; grouped line-item F1 below 1 |
| gemini | cord_test_0002 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| modal_hybrid | ara_0084 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| modal_hybrid | ara_0018 | None | None | all relevant fields unannotated; no quality pass can be claimed |
| modal_hybrid | ara_0050 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| modal_hybrid | ara_0033 | False | None | annotated scalar error |
| modal_hybrid | ara_0022 | False | True | annotated scalar error |
| modal_hybrid | ara_0064 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| modal_hybrid | cord_test_0014 | False | True | annotated scalar error |
| modal_hybrid | cord_test_0027 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| modal_hybrid | cord_test_0065 | False | False | annotated scalar error; grouped line-item F1 below 1 |
| modal_hybrid | cord_test_0002 | False | False | annotated scalar error |
| ocr_openrouter | ara_0084 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| ocr_openrouter | ara_0018 | None | None | all relevant fields unannotated; no quality pass can be claimed |
| ocr_openrouter | ara_0050 | True | True | passes covered annotations only; unannotated fields remain unscorable |
| ocr_openrouter | ara_0033 | False | None | annotated scalar error |
| ocr_openrouter | ara_0022 | False | True | annotated scalar error |
| ocr_openrouter | ara_0064 | False | True | annotated scalar error |
| ocr_openrouter | cord_test_0014 | False | False | invalid final receipt; annotated scalar error; grouped line-item F1 below 1 |
| ocr_openrouter | cord_test_0027 | False | True | grouped line-item F1 below 1 |
| ocr_openrouter | cord_test_0065 | False | False | annotated scalar error; grouped line-item F1 below 1 |
| ocr_openrouter | cord_test_0002 | False | True | grouped line-item F1 below 1 |
