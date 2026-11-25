"""Development-only OCR comparison; annotations are partial, so CER is unscorable."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import statistics
import time

from dotenv import load_dotenv
from benchmark.metrics import normalize_text
from benchmark.profiler import Profile

ROOT = Path(__file__).resolve().parents[1]


def coverage_metrics(text, reference):
    # Token multiset recall avoids scoring unannotated image text as OCR insertions.
    actual = Counter(normalize_text(text).split())
    expected = Counter(normalize_text(reference).split())
    numbers = lambda t: Counter(re.findall(r"\d+(?:[.,٫٬]\d+)*", t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))))
    an, en = numbers(text), numbers(reference)
    return {"annotated_token_recall": sum((actual & expected).values()) / sum(expected.values()) if expected else None,
            "annotated_numeric_token_recall": sum((an & en).values()) / sum(en.values()) if en else None,
            "full_ocr_cer": None, "coverage_note": "Partial transcription; token recall only, not full OCR accuracy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["tesseract", "paddle"], required=True)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache/paddlex"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".cache/modelscope"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from app.providers.ocr import run_ocr
    manifest_path = ROOT / "data/manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT / "data/dev_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = ROOT / "results/ocr_development" / args.engine
    output.mkdir(parents=True, exist_ok=True)

    def process(record):
        truth = json.loads((ROOT / record["annotation"]).read_text())
        start = time.perf_counter()
        try:
            result = run_ocr((ROOT / record["image"]).read_bytes(), args.engine, language=record["language"])
            metrics = coverage_metrics(result["text"], truth["ocr_text"])
            data = {"id": record["id"], "language": record["language"], "ocr": result, "metrics": metrics, "latency_seconds": time.perf_counter() - start, "status": "ok"}
        except Exception as exc:
            data = {"id": record["id"], "language": record["language"], "status": "error", "error": type(exc).__name__ + ": " + str(exc)[:500], "latency_seconds": time.perf_counter() - start}
        (output / f'{record["id"]}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(data["id"], data["status"], round(data["latency_seconds"], 2), data.get("metrics", {}), flush=True)
        return data

    with Profile() as profile:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(process, manifest["development"]))
    good = [row for row in rows if row["status"] == "ok"]
    summary = {"engine": args.engine, "concurrency": args.concurrency, "count": len(rows), "succeeded": len(good), "profile": profile.result,
               "receipts_per_minute": len(rows) * 60 / profile.result["wall_seconds"],
               "mean_latency_seconds": statistics.mean(row["latency_seconds"] for row in rows),
               "initialization_seconds": sum(row.get("ocr", {}).get("initialization_seconds", 0) for row in rows),
               "language_hint": "known corpus language routes Arabic/English recognizer; API uses OCR_LANGUAGE env for Paddle",
               "metrics": {k: statistics.mean(row["metrics"][k] for row in good if row["metrics"][k] is not None) if any(row["metrics"][k] is not None for row in good) else None for k in ["annotated_token_recall", "annotated_numeric_token_recall"]}}
    (output / f"summary_c{args.concurrency}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
