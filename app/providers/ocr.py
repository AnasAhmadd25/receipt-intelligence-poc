"""CPU OCR adapters. Imports are lazy so Gemini-only installs need no OCR stack."""
import io
import os
from pathlib import Path
import threading
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
_paddle_engines = {}
_paddle_lock = threading.Lock()


def reconstruct_lines(words):
    groups = {}
    for word in words:
        if not str(word.get("text", "")).strip() or float(word.get("conf", -1)) < 0:
            continue
        key = tuple(word.get(x, 0) for x in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(word)
    rows = []
    for group in groups.values():
        ordered = sorted(group, key=lambda w: w["left"])
        # RTL-only rows retain Tesseract word order; mixed rows expose every box too.
        arabic_chars = sum("\u0600" <= c <= "\u06ff" for w in ordered for c in w["text"])
        if arabic_chars > sum(c.isalpha() and c.isascii() for w in ordered for c in w["text"]):
            ordered = sorted(group, key=lambda w: w["left"], reverse=True)
        rows.append({"text": " ".join(w["text"] for w in ordered),
                     "bbox": [min(w["left"] for w in group), min(w["top"] for w in group), max(w["left"] + w["width"] for w in group), max(w["top"] + w["height"] for w in group)],
                     "confidence": sum(float(w["conf"]) for w in group) / len(group),
                     "words": ordered})
    return sorted(rows, key=lambda row: (row["bbox"][1], row["bbox"][0]))


def run_ocr(image_bytes: bytes, engine: str = "tesseract", language: str | None = None) -> dict:
    started = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    initialization = 0.0
    if engine == "tesseract":
        import pytesseract
        # Limit each OCR child to one thread to make concurrency meaningful.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        tessdata = os.getenv("TESSDATA_PREFIX", str(ROOT / "data/tessdata"))
        config = f'--tessdata-dir "{tessdata}" --psm 6'
        data = pytesseract.image_to_data(image, lang="ara+eng", config=config, output_type=pytesseract.Output.DICT)
        words = [{k: data[k][i] for k in data} for i in range(len(data["text"]))]
        rows = reconstruct_lines(words)
        model = "tessdata_fast ara + installed eng, psm6"
    elif engine == "paddle":
        import numpy as np
        from paddleocr import PaddleOCR
        # Separate recognizers are necessary: the Arabic alphabet model lacks Latin coverage.
        language = language or os.getenv("OCR_LANGUAGE", "ar")
        language = "ar" if language.casefold().startswith("ar") else "en"
        with _paddle_lock:
            if language not in _paddle_engines:
                init = time.perf_counter()
                _paddle_engines[language] = PaddleOCR(
                    lang=language, ocr_version="PP-OCRv5", device="cpu",
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name=("arabic" if language == "ar" else "en") + "_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, cpu_threads=1, enable_mkldnn=False,
                )
                initialization = time.perf_counter() - init
            result = list(_paddle_engines[language].predict(np.array(image)))
        rows = []
        for page in result:
            for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_boxes"]):
                rows.append({"text": str(text), "bbox": [int(v) for v in box], "confidence": float(score) * 100})
        rows.sort(key=lambda row: (row["bbox"][1], row["bbox"][0]))
        model = ("arabic" if language == "ar" else "en") + "_PP-OCRv5_mobile_rec + PP-OCRv5_mobile_det"
    else:
        raise ValueError("Unknown OCR engine")
    return {"text": "\n".join(row["text"] for row in rows), "lines": rows, "engine": engine, "model": model,
            "initialization_seconds": initialization, "latency_seconds": time.perf_counter() - started,
            "language_hint": language, "image_size": list(image.size)}
