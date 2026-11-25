# Dataset and scoring audit

Prepared 2026-11-25. This is an Arabic versus non-Arabic receipt benchmark. CORD is Indonesian with some English product names; these samples **do not establish English-only receipt performance**.

## Downloaded sample

| Source | Held-out test | Development | Fixed initial test | Language |
| --- | ---: | ---: | ---: | --- |
| AraReceipt | 30 | 5 | 6 | Arabic, mixed Arabic-English |
| CORD-v2 | 20 | 3 | 4 | Indonesian, Latin script, some English names |
| Total | 50 | 8 | 10 | |

All 58 image files were downloaded, decoded and verified with Pillow. There are 58 distinct SHA-256 file hashes and 58 distinct decoded RGB pixel hashes. Development and test IDs and image hashes are disjoint. This checks exact duplicate files/pixels; it does not prove there are no different photographs of the same physical receipt. Images total 50,444,508 bytes (about 48.1 MiB); dimensions range from 320–4,224 pixels wide and 624–5,632 pixels high.

`data/manifest.json` records paths, IDs, source split, row index, image SHA-256, seed, initial test IDs, and pinned source revisions. `data/dev_manifest.json` contains development records only. Random seed: `20261125`. A Python `random.Random(seed)` shuffles the complete 100 source-row indices in Ara train, CORD test, CORD validation order. Ara selects the first 30 for test, next 5 for development; CORD selects the first 20 test and first 3 validation rows. The first 6 Ara and first 4 CORD test IDs are frozen as the initial comparison:

`ara_0084`, `ara_0018`, `ara_0050`, `ara_0033`, `ara_0022`, `ara_0064`, `cord_test_0014`, `cord_test_0027`, `cord_test_0065`, `cord_test_0002`.

Download whole parquet shards was avoided. Ara row metadata was fetched once; selected CORD metadata was fetched through individual dataset-server rows. Only selected image assets were downloaded. Public cached CORD endpoints occasionally returned HTTP 500 or stalled; bounded retries completed all selected rows without substituting easier examples.

## Provenance and licenses

[AraReceipt official card](https://huggingface.co/datasets/IslamMesabah/AraReceipt) declares MIT and describes 100 manually reviewed Arabic receipts derived from ReceiptSense. Its annotations were produced using OCR and LLM pre-annotation followed by one human annotator correcting and approving regions. They are released human-reviewed labels, not predictions generated for this experiment. The single-annotator process has no independent agreement estimate. ReceiptSense was not added separately, avoiding a known derivative-source overlap risk. CORU was not needed.

AraReceipt revision: `d995859bd253ad37053faee8bce24b5fa1c265e2`. Credit: Islam Mesabah and the AraReceipt/GAIDA authors; original image provenance: ReceiptSense. The publisher's card and class mapping are retained under `data/sources/AraReceipt/`. The repository declares MIT in its card; it does not include a separate license file in the pinned file inventory.

[CORD-v2 official card](https://huggingface.co/datasets/naver-clova-ix/cord-v2) declares CC-BY-4.0. The [CORD authors' repository](https://github.com/clovaai/cord) identifies Indonesian receipts, provides group and text annotations, and says store/payment/other header classes were removed from the public release. Its class table identifies `menu.price` as line total and `menu.unitprice` as unit price. The release is a public sample, not the entire original collection.

CORD revision: `7f0115a4b758a71d6473b8d085751692da2fef98`. Credit: NAVER CLOVA, Seunghyun Park and the CORD authors, *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing* (2019). The card, Hub metadata, and official `LICENSE-CC-BY` are retained under `data/sources/CORD-v2/`. Changes in this POC: deterministic subsetting, annotation conversion, numeric normalization, and flattening annotated submenu modifier lines.

No model inference or newly generated OCR output was used as benchmark truth. Original published annotations are retained separately from the standardized schema for inspection.

## Observable held-out annotations

Counts below are the denominators available in the actual frozen test sample, after conservative conversion.

| Scalar field | AraReceipt / 30 | CORD-v2 / 20 |
| --- | ---: | ---: |
| merchant_name | 21 | 0 |
| receipt_number | 0 | 0 |
| purchase_date | 17 | 0 |
| purchase_time | 15 | 0 |
| currency | 0 | 0 |
| subtotal | 7 | 12 |
| discount | 0 | 0 |
| tax | 6 | 6 |
| total | 24 | 18 |

The 20 CORD test receipts contain 36 scored item lines: description 36, quantity 30, unit price 6, line total 36. AraReceipt has product-region labels but no authoritative item grouping or unambiguous unit-price-versus-line-total association; its item F1 is unscorable. Therefore **Arabic quantity/unit-price/item-F1 accuracy is not established** by this dataset. The 5 Arabic development receipts provide 14 scalar annotations; 3 CORD development receipts provide 6 scalar annotations and 5 item lines.

AraReceipt omits ambiguous or unparseable labeled values instead of choosing a convenient one: merchant 6 receipts, date 12, time 11, subtotal 14, tax 2, total 5. A field with no source annotation is also unscorable, so these omission counts do not exhaust all missing coverage. A single unambiguous normalized value is retained; duplicate identical labeled values collapse. Tax percentage labels are not treated as tax amounts. Fragmented store names are not reconstructed by guessing geometric relationships.

CORD `cord_test_0009` has cash/change but no annotated payable total; `cord_test_0037` labels its displayed amount `total_etc`, not `total_price`. Neither is relabeled by inference. Service charges and cash/change are preserved in original annotations but have no equivalent scored canonical field. `menu.sub` entries are retained as separate printed modifier lines; their original hierarchy remains available in original annotations. Unknown quantity and unit price are never derived from totals.

Monetary/quantity truth values are decimal strings, processed using `Decimal`, not binary floats. CORD's Indonesian comma/dot thousands separators are normalized to integers where appropriate (`12.000` → `12000`). AraReceipt accepts explicit numeric amounts and Arabic digits. Dates use explicit year-first or day-first parsing for Arabic receipts; invalid or multiple distinct dates/times are masked. This date policy is a dataset convention and should not be applied universally to other locales.

## Metric definitions and limits

- Scalar accuracy uses only `annotated_fields`; every field returns correct/total counts. Receipt-number, currency, held-out discount, and English-only performance have no available validation denominator here.
- Missing means an annotated present value was omitted by a prediction. Hallucinated means a value was predicted where truth **explicitly asserts known absence**. Unknown/unannotated fields are listed as unscorable, never hallucinated. This corpus does not assert absent fields, so scalar hallucination rate has denominator zero and must be reported unavailable, not zero percent.
- Item F1 uses maximum-cardinality one-to-one matching. Description similarity must be at least 0.8, with every other annotated item value matching after normalization. Duplicate predictions cannot receive repeated credit.
- Item-field accuracy and description diagnostics use a separate one-to-one maximum-weight Hungarian pairing by description, accepting similarity at least 0.5. Pairing does not use numeric correctness. Unknown per-item values are masked. Description field accuracy is normalized exact match; description similarity is the mean character-sequence ratio, with unpaired annotated descriptions contributing zero.
- Both sources provide partial labeled-region text, not complete receipt transcription. `ocr_text_complete` is false in every truth file. OCR comparisons must be labeled annotated-text coverage/recall; a whole-receipt OCR CER claim is unsupported.
- Public pretrained models may have seen these public datasets; model training contamination is not measured. Fifty receipts support a bounded POC comparison, not production-level population estimates or full multilingual claims.

Validation: 9 metric tests and 5 data tests pass. Tests cover unknown versus known-absent fields, Arabic digits, Decimal precision, one-to-one duplicate handling, quantity/price diagnostics independent of money correctness, conservative field conversion, modifier coverage, deterministic counts, file hashes, and split separation.
