---
license: mit
language:
- ar
task_categories:
- object-detection
- image-to-text
tags:
- document-understanding
- key-information-extraction
- receipts
- ocr
- arabic
size_categories:
- n<1K
---
# AraReceipt

AraReceipt is a manually annotated dataset of **100 Arabic retail receipt images** labeled with **25 key-information classes** plus an `Ignore` category, in [WildReceipt](https://download.openmmlab.com/mmocr/data/wildreceipt.tar) style. It contains **4,609 annotated regions** (46.1 ± 21.6 per image), each with a box, a transcription, and a semantic class.

The dataset was built with [GAIDA](https://github.com/islammesabah/gaida), a human-in-the-loop annotation system that combines OCR-based region extraction, LLM-based semantic pre-annotation, and human validation in Label Studio. Producing AraReceipt required no changes to GAIDA's architecture — only OCR settings for Arabic and mixed-script text, prompt revisions for right-to-left writing, and language-specific field expressions. What's released here is the final result of that process: reviewed, human-corrected annotations only — no model predictions, scores, or annotation-session metadata.

Images are derived from ReceiptSense; AraReceipt is released under the MIT license, matching the terms of the source data.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("IslamMesabah/AraReceipt", split="train")
print(ds[0]["id"], len(ds[0]["annotations"]))
print(ds[0]["annotations"][0])
```

## Fields

Each row is one receipt image, named `images/0001.jpg` through `images/0100.jpg`.

| Field                 | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `image`             | The receipt image.                                           |
| `file_name`         | Path of the image inside the dataset.                        |
| `id`                | Sequential dataset id (1-100), matching the image file name. |
| `width`, `height` | Image size in pixels.                                        |
| `annotations`       | The annotated regions (see below).                           |

Each entry of `annotations` has:

| Field     | Description                                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `box`   | Quadrilateral as`[x1, y1, x2, y2, x3, y3, x4, y4]` in **absolute pixels**, corners ordered top-left, top-right, bottom-right, bottom-left. |
| `text`  | Transcription of the region.                                                                                                                       |
| `label` | Numeric class id; see`class_list.txt` for the id-to-name mapping (same 26 classes as WildReceipt: 25 key-information classes plus `Ignore`).   |

## Examples

Each region is boxed and labeled with its `label` class name, colored per class. These are the final, human-reviewed annotations.

<p>
  <img src="examples/example_0001.jpg" alt="AraReceipt example" width="280">
</p>

## Class distribution

| Class               | Regions | Class            | Regions |
| ------------------- | ------: | ---------------- | ------: |
| Others              |    1598 | Store_addr_value |      88 |
| Prod_price_value    |     466 | Date_key         |      61 |
| Prod_quantity_value |     461 | Tax_value        |      45 |
| Prod_item_value     |     402 | Tel_key          |      40 |
| Subtotal_value      |     225 | Store_addr_key   |      39 |
| Total_value         |     160 | Time_key         |      30 |
| Total_key           |     127 | Tax_key          |      21 |
| Date_value          |     113 | Tips_key         |      10 |
| Subtotal_key        |     110 | Tips_value       |       9 |
| Store_name_value    |     109 | Ignore           |       7 |
| Prod_price_key      |     101 | Store_name_key   |       1 |
| Time_value          |     100 |                  |         |
| Prod_item_key       |      99 |                  |         |
| Prod_quantity_key   |      97 |                  |         |
| Tel_value           |      90 |                  |         |

Slightly over half of the regions contain Arabic script and 43.6% are purely numeric (prices, quantities, totals), reflecting the mixed-script character of Arabic retail receipts. Key-type classes are markedly rarer than their value counterparts, because Arabic receipts often print values without field labels.

## Annotation process

One native Arabic speaker with professional English proficiency annotated all 100 images using GAIDA: PaddleOCR proposed regions and transcriptions, Llama 4 Maverick proposed a class for each region, and the annotator reviewed, corrected, and approved every region before export.

The end-to-end assisted annotation time averaged 5:51 per image (~10 hours in total).

## Limitations

The dataset was annotated by a single annotator without an independent second pass, so no inter-annotator agreement is available. Region boundaries originate from OCR detections and were corrected rather than drawn from scratch, which may bias them toward what the detector proposed. 

## Citation

```bibtex
@inproceedings{mesabah2026gaida,
  title     = {GAIDA: Generative AI for Human-in-the-Loop Document Annotation},
  author    = {Mesabah, Islam and Elhosary, Ali and Sandulu, Priyabanta and
               Obradovic, Darko and Vollmer, Sebastian J.},
  year      = {2026}
}
```
