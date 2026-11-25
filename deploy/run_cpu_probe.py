"""Run real Linux x86 OCR probes on development images only."""
import json
from pathlib import Path
import time
import os
import argparse

os.environ["MODAL_APP_NAME"] = "receipt-intelligence-cpu-probes"
from deploy.modal_app import app, linux_cpu_probe, linux_paddle_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("tesseract", "paddle"), default="tesseract")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / "data/dev_manifest.json").read_text())["development"]
    images = [(root / r["image"]).read_bytes() for r in entries]
    languages = ["ar" if r["dataset"] == "AraReceipt" else "en" for r in entries]
    results = []
    with app.run():
        for concurrency in (1, 2, 4):
            start = time.monotonic()
            result = (linux_paddle_probe.remote(images, languages, concurrency=concurrency)
                      if args.engine == "paddle" else linux_cpu_probe.remote(images, concurrency=concurrency))
            result["sample_ids"] = [r["id"] for r in entries]
            result["client_wall_seconds"] = time.monotonic() - start
            results.append(result)
            destination = "modal_linux_paddle.json" if args.engine == "paddle" else "modal_linux_cpu.json"
            (root / "results" / destination).write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print({"concurrency": concurrency, "wall_seconds": result["resource"]["wall_seconds"],
                   "receipts_per_minute": result["resource"]["receipts_per_minute"]})


if __name__ == "__main__":
    main()
