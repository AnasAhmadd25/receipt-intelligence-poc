"""Compare OCR sources with a fixed cheap text-only structurer.

Oracle source is development-only, using partial annotated transcripts, never
target JSON. It diagnoses the structurer and is not a deployable OCR result.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from app.providers.cloud import GeminiTextProvider
from app.providers.base import ProviderError
from app.providers.ocr import group_detection_rows
from benchmark.metrics import evaluate
from benchmark.acceptance import summarize_acceptance
from benchmark.cost import Budget, inference_cost

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    load_dotenv(ROOT / ".env")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if args.source == "oracle" and args.split != "development":
        raise ValueError("Oracle diagnostics are development-only")
    records = manifest[args.split if args.split == "development" else "test"]
    if args.split == "test":
        records = [r for r in records if r["id"] in manifest["initial_test_ids"]]
    destination = ROOT / "results" / ("ocr_structuring_" + args.split) / args.source
    destination.mkdir(parents=True, exist_ok=True)
    pending_ids = {r["id"] for r in records if not (destination / (r["id"] + ".json")).exists()}
    budget = Budget(ROOT / "results/spend_ledger.json")
    reservation = f"ocr_structuring/{args.split}/{args.source}"
    if pending_ids:
        budget.reserve(reservation, len(pending_ids) * .04)
    rows = []
    for record in records:
        path = destination / (record["id"] + ".json")
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        if rows:
            await asyncio.sleep(args.delay)
        truth = json.loads((ROOT / record["annotation"]).read_text())
        if args.source == "oracle":
            observed = {"text": truth["ocr_text"], "lines": [], "engine": "annotation_partial_transcript_control"}
        elif args.source == "vl16":
            observed = json.loads((ROOT / "results/ocr_vl16_dev" / (record["id"] + ".json")).read_text())
            observed = {"text": observed["text"], "lines": [], "engine": "paddle_vl16_element_ocr"}
        else:
            observed = json.loads((ROOT / "results/ocr_cache/paddle" / (record["id"] + ".json")).read_text())
            if args.source == "paddle_rows":
                observed["lines"] = group_detection_rows(observed["lines"])
                observed["text"] = "\n".join(row["text"] for row in observed["lines"])
                observed["engine"] = "paddle_grouped_rows"
        row = {"id": record["id"], "dataset": record["dataset"], "split": args.split,
               "source": args.source, "image_sha256": record["sha256"], "provider": "ocr_gemini",
               "ocr_sha256": hashlib.sha256(json.dumps(observed, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        started = time.monotonic()
        try:
            row.update(await GeminiTextProvider().extract(b"", "image/jpeg", ocr=observed))
            row["status"] = "ok"
        except ProviderError as exc:
            row.update(status="error", error=str(exc), metadata=exc.metadata)
        row["wall_seconds"] = time.monotonic() - started
        row["metrics"] = evaluate(row.get("receipt") or {}, truth)
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
        rows.append(row)
        print(args.source, row["id"], row["status"], row["metrics"]["field_correct"], row["metrics"]["field_total"], flush=True)
    costs = [inference_cost(row.get("metadata", {})) for row in rows]
    summary = summarize_acceptance(rows)
    summary.update(field_correct=sum(r["metrics"]["field_correct"] for r in rows),
                   field_total=sum(r["metrics"]["field_total"] for r in rows),
                   input_tokens=sum(r.get("metadata", {}).get("input_tokens", 0) for r in rows),
                   output_tokens=sum(r.get("metadata", {}).get("output_tokens", 0) for r in rows),
                   missing_usage_requests=sum(c is None for c in costs),
                   mean_seconds=sum(r["wall_seconds"] for r in rows) / len(rows),
                   ocr_hosting_excluded=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    new_costs = [inference_cost(r.get("metadata", {})) for r in rows if r["id"] in pending_ids]
    if pending_ids and all(c is not None for c in new_costs):
        budget.settle(reservation, sum(new_costs))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["paddle", "paddle_rows", "oracle", "vl16"], required=True)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--delay", type=float, default=15, help="Seconds between new API requests; respect account rate limits")
    asyncio.run(run(parser.parse_args()))
