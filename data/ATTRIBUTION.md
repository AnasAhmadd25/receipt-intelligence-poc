# Receipt sample attribution

AraReceipt: Islam Mesabah and AraReceipt/GAIDA authors; images derived from ReceiptSense. Publisher declares MIT. Official dataset: https://huggingface.co/datasets/IslamMesabah/AraReceipt . Card and class mapping retained in `sources/AraReceipt/`.

CORD-v2: NAVER CLOVA and CORD authors, including Seunghyun Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing* (2019). CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/ . Official dataset: https://huggingface.co/datasets/naver-clova-ix/cord-v2 ; authors' repository: https://github.com/clovaai/cord . License text retained in `sources/CORD-v2/LICENSE-CC-BY`.

Modifications: seeded sample selection, conversion of published annotations into a common receipt schema, monetary/date/time normalization, masking ambiguous values, and flattening annotated CORD modifier lines. Original annotations are retained. No model predictions were substituted for source labels.

See `../reports/dataset_audit.md` for revisions, split policy, coverage, limitations, and metric definitions. Do not label CORD as an English-only benchmark.

## OCR language data

The bundled `ara.traineddata` is from [tesseract-ocr/tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast); `eng.traineddata` is the existing Tesseract English language pack. Their distribution uses Apache License2.0; the license is retained in `data/tessdata/LICENSE`. Checksums are in `configs/ocr_weights.json`. GPU model weights and Paddle caches are not bundled.
