# Receipt Intelligence experiment plan

User supplied the architectures, datasets, execution order, $5 spending limit and permission to make technical choices. Work is isolated in this new directory; existing projects are read only.

1. Inspect environment and reuse available credential integrations; record names/status only.
2. Prepare 50 deterministic held-out receipts plus 8 development receipts; record original annotations, coverage and SHA256. Initial 10 same receipts for all providers.
3. Shared Pydantic 1.0 schema with nullable fields, Decimal money and raw text. Metrics mask unannotated fields; item matching is one-to-one.
4. Evaluate CPU Tesseract ara+eng and PaddleOCR v5 Arabic/English on development. Preserve TSV/boxes, compare character/numeric accuracy and resource profiles. Freeze selected OCR/prompt before test.
5. Thin hosted Ollama/NVIDIA text clients and Gemini image SDK client; max one JSON repair for text clients. Same OCR cache for A variants.
6. Investigate GLM-OCR/HunyuanOCR Arabic support and T4 compatibility; run Modal development trials within reserved budget and deploy selected model with API key, cache and scale-zero.
7. Run held-out 10 across available providers; record explicit failures, usage, raw predictions and resource profiles. Profile concurrency 1/2/4 where practical.
8. Generate comparisons/costs/sizing from actual evidence. Unknown metrics N/A, estimated prices and throughput labeled, never call Apple performance measured x86 performance.

Verification focuses: no unannotated-field penalties; duplicate prediction cannot match two ground-truth items; decimal Arabic parsing; SSRF/private address blocking including DNS rebinding; all paid attempts and repairs counted. Test these before relevant code, then run integration probes and full suite.

Paid limit: $5 global. Reserve up to $2.50 Modal experiments after parent confirmation. Conservative request token caps and explicit usage ledger for APIs. Failures with unknown usage remain unknown, never free by assumption.
