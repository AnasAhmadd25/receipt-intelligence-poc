# Arabic receipt OCR research

Research date: **2026-11-25**. Sources below are developer repositories, official model cards, developer papers, and upstream runtime documentation. This is a selection memo, not a new receipt benchmark report.

## Decision and experiment status

The selected next experiments are **PaddleOCR-VL-1.6**, spatial row grouping of OCR boxes, and a **Gemini 3.1 text-model extraction pass** over the OCR output. Their purpose is to improve Arabic recognition and preserve item/quantity/price associations while keeping the language-model stage inexpensive. Research preceded testing. The completed follow-up is in [the final report](../reports/FINAL_REPORT.md): original Paddle boxes plus text-only Gemini reached20/26 test scalars and4/9 strict passes; row grouping did not improve development scores; the stronger VL runtime produced no validated OCR result. Research claims below do not substitute for those measurements. The exact Gemini endpoint and measured results belong in the experiment artifacts, not inferred from a product-family name.

Run the same receipt corpus through the alternatives and compare transcription, scalar fields, item rows, numeric associations, latency, and actual usage. General document benchmarks do not establish Arabic receipt-photo performance. An LLM cannot reliably recover characters that OCR omitted; spatial grouping can help associations but does not repair missing visual evidence.

All development for this investigation belongs in the new separate `receipt-intelligence-ocr-rd` copy. The original sibling project is outside this memo's editing scope.

## PaddleOCR choices

**PaddleOCR-VL-1.6** is the selected first trial: an advertised **0.9B** document VLM with Apache 2.0 licensing. PaddleOCR-VL's documented language coverage is **109 languages**, including the Arabic-script group; language coverage is not a receipt accuracy guarantee. The current card's Transformers example requires `transformers>=5.0.0` and uses the checkpoint `PaddlePaddle/PaddleOCR-VL-1.6` with `OCR:`, `Spotting:`, or task-specific prompts. The card expressly limits that example to **element recognition and text spotting**. It must not be described as the complete page-layout pipeline. The separately documented `PaddleOCRVL(pipeline_version="v1.6")` path provides page parsing. The sample uses BF16; a T4-compatible precision/backend needs validation.

Sources: [1.6 checkpoint and inference interface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), [official language appendix](https://ernie.baidu.com/blog/publication/PaddleOCR-VL_Technical_Report.pdf).

**PP-OCRv6 is not an automatic Arabic-recognizer upgrade.** The current official README advertises a unified recognizer covering 50 languages: Chinese variants, English, Japanese, and 46 Latin-script languages. Arabic is absent from that recognizer's supported-language table, while PP-OCRv5 lists `ar`. Therefore upgrading a package to v6 does not establish that Arabic recognition has improved. A newer detector could be evaluated separately with the existing Arabic recognizer.

Sources: [official README](https://github.com/PaddlePaddle/PaddleOCR), [version-specific supported-language table](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md), [developer's v6 announcement](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6).

## Other candidates, ranked for follow-up trials

| Priority | Candidate | Evidence and limits for these receipts |
|---|---|---|
| 1 after PaddleOCR-VL | **dots.mocr**, current checkpoint `dots-studio/dots.mocr`, approximately 3B | Successor to dots.ocr-1.5, renamed March 19, 2026. Official examples include scene-text spotting. The parsing interface emits boxes, categories, human reading order, Markdown text and HTML tables. Original dots.ocr reports multilingual training and a 126-language benchmark, but no Arabic-specific result was located in the inspected current cards/papers. Receipt performance remains unverified. |
| 2 | **MinerU**, current application workflow 4.2; relevant released VLM `opendatalab/MinerU2.5-Pro-2604-1.2B` | Potentially useful when layout/table handling dominates. Basic uses small models without a VLM; Standard adds a VLM; Advanced increases computation. Official ecosystem documentation lists an `arabic` language pack. This does not establish Arabic accuracy of the Pro VLM, whose model card is tagged Chinese/English. |
| 3, exploratory | **Surya 2.5**, `datalab-to/surya-ocr-2.5`, 689M | Small alternative, but developer evidence gives weak grounds to prioritize it for Arabic photos: the internal multilingual benchmark reports **45.4% Arabic pass rate**. This is neither character accuracy nor receipt accuracy. The README says photos/natural scenes are not its target. |

Sources: [dots.mocr repository](https://github.com/studio-dots-ai/dots.mocr), [dots.mocr checkpoint](https://huggingface.co/dots-studio/dots.mocr), [original dots.ocr paper](https://arxiv.org/html/2512.02498v1), [MinerU repository](https://github.com/opendatalab/MinerU), [MinerU Pro checkpoint](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B), [official MinerU Arabic pack](https://github.com/opendatalab/MinerU-Ecosystem/blob/main/skills/SKILL.md), [Surya README including Arabic results](https://github.com/datalab-to/surya).

Surya's package release **0.24.0** was published November 13, 2026. It returns structured blocks with HTML, boxes/polygons, reading order and confidence; tables are HTML. The 2.5 card distinguishes HTML tables from row/column geometry, for which it points to a Surya 2 checkpoint. Backends include vLLM, llama.cpp and opt-in MLX. Its advertised H100 throughput is not a T4 or single-receipt latency estimate. Sources: [release](https://github.com/datalab-to/surya/releases), [2.5 model card](https://huggingface.co/datalab-to/surya-ocr-2.5).

## DeepDoc and DeepSeek-OCR are different systems

**DeepDoc** is RAGFlow's document-analysis subsystem: OCR, layout recognition, table-structure recognition and parsing. It is more than a wrapper: the published implementation has ONNX detection and CTC recognition. The inspected **pinned v0.27.2** code loads `det.onnx`, `rec.onnx` and `ocr.res`; it sorts boxes top-to-bottom/left-to-right and exposes no language argument. The main-branch raw paths inspected during research returned 404, so these implementation observations are version-specific. No official Arabic receipt accuracy evidence was found.

A 2025 bot response saying RAGFlow lacked Arabic support is stale: a contributor closed the related issue with an affirmative reply on June 17, 2026. This supports a general Arabic-support claim, not the accuracy of a particular OCR backend. RAGFlow 1.0 was released November 6 with a major Go rewrite and unified parsing integration. Installing the complete RAG system is not evidence of stronger receipt OCR.

Sources: [DeepDoc introduction](https://github.com/infiniflow/ragflow/tree/main/deepdoc), [pinned OCR implementation](https://raw.githubusercontent.com/infiniflow/ragflow/v0.27.2/deepdoc/vision/ocr.py), [Arabic issue and contributor reply](https://github.com/infiniflow/ragflow/issues/9575), [RAGFlow 1.0](https://github.com/infiniflow/ragflow/releases/tag/v1.0.0).

**DeepSeek-OCR-2** is a separate 3B checkpoint, `deepseek-ai/DeepSeek-OCR-2`, with Apache 2.0 licensing. It supports free OCR and grounded Markdown conversion. Its headline paper evaluates Chinese/English OmniDocBench, not Arabic receipts; no Arabic-specific result was located. The official Transformers recipe uses BF16 and FlashAttention 2. Its dynamic crop/image path does not by itself guarantee suitability for receipt photos or low T4 latency. It is a possible later experiment, not the first cheap Arabic choice.

Sources: [checkpoint and inference recipe](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), [developer paper](https://arxiv.org/html/2601.20552v1), [license](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/main/LICENSE.txt).

**Docling** provides document structure, reading order, table handling, JSON/Markdown exports and pluggable OCR/VLM support. Selecting it without changing and testing the underlying recognizer would not demonstrate an Arabic-recognition improvement. Source: [official repository](https://github.com/docling-project/docling).

## Hardware boundaries

- **T4 precision and attention matter.** Standard FlashAttention 2 supports Ampere/Ada/Hopper; the upstream project points T4/Turing users to a separate implementation. BF16 requires newer supported architectures. Dots.mocr and DeepSeek's supplied Transformers recipes use both BF16 and FlashAttention 2. FP16 plus another attention implementation may work, but must be validated rather than presented as a supported drop-in recipe. [Upstream requirements](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).
- **Do not invent minimum VRAM.** The inspected dots/DeepSeek cards do not provide a reliable T4 minimum or guarantee. Parameter-memory arithmetic alone excludes activations, KV cache, image tokens and runtime overhead.
- **Surya 2.5** declares BF16 in its configuration but offers llama.cpp/MLX paths. Its actual precision, backend and long-image behavior still require measurement. [Configuration](https://huggingface.co/datalab-to/surya-ocr-2.5/blob/main/config.json).
- **MinerU Basic** can run on CPU. Its current runtime guide describes 8 GB VRAM as a planning starting point, not a guarantee, and recommends at least 16 GB system memory for higher-throughput deployment. CPU/ONNX and VLM-engine choices are independent. [Official runtime guide](https://opendatalab.github.io/MinerU/usage/tiers/).

## License distinctions

These are source summaries for implementation selection. Preserve the actual pinned license files with any redistributed model or application.

| Component | Source terms relevant to selection |
|---|---|
| PaddleOCR-VL-1.6 | Official checkpoint labeled Apache 2.0. [Model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6). |
| DeepDoc | Official ONNX model repository labeled Apache 2.0. [Model repository](https://huggingface.co/InfiniFlow/deepdoc). |
| DeepSeek-OCR-2 | Apache 2.0 code/checkpoint. [License](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/main/LICENSE.txt). |
| dots.mocr | MIT labeling plus a supplemental model agreement. It permits commercial/proprietary/SaaS use, says MIT prevails in conflicts, and adds responsible-use, privacy and copyright provisions. Avoid reducing this to an unrestricted-MIT claim. [Agreement](https://github.com/studio-dots-ai/dots.mocr/blob/main/dots.mocr%20LICENSE%20AGREEMENT). |
| Surya | Code Apache 2.0; weights modified OpenRAIL. Restrictions cover employer/affiliated entities above $5M previous-year revenue or $5M total funding except personal/research use, and competing products/services. The text also includes output attribution and share-alike clauses. [Actual weight license](https://raw.githubusercontent.com/datalab-to/surya/master/MODEL_LICENSE). |
| MinerU application | Current license is Apache 2.0 with added terms, replacing older AGPL descriptions. A separate commercial license is required above 100M MAU or $20M monthly revenue across affiliates; third-party online services must prominently attribute MinerU. [Application license](https://raw.githubusercontent.com/opendatalab/MinerU/master/LICENSE.md). |
| MinerU Pro checkpoint | Its model card separately labels the checkpoint Apache 2.0; distinguish model weights from the application license. [Checkpoint](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B). |

## Interpretation rules

Supported-language counts establish coverage claims, not accuracy. Internal multilingual pass rates, document edit distance and receipt field/item scores use different metrics and must not be compared numerically. Benchmark strengths in long PDFs, mathematics, charts or SVG reconstruction do not establish thermal-receipt-photo performance. Keep raw OCR, boxes, grouped rows, extraction responses, runtime and measured usage so every improvement can be traced to actual receipt evidence.
