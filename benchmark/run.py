"""Run reproducible held-out/provider experiments with cached OCR and raw evidence."""
import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv
from app.providers.base import ProviderError
from benchmark.metrics import evaluate
from benchmark.profiler import Profile
from benchmark.cost import Budget, inference_cost

ROOT = Path(__file__).resolve().parents[1]


def selected_records(manifest, split, limit):
    if split == "development":
        return manifest["development"][:limit]
    if limit == 10:
        ids = set(manifest["initial_test_ids"])
        return [r for r in manifest["test"] if r["id"] in ids]
    return manifest["test"][:limit]


async def run(args):
    load_dotenv(ROOT / ".env")
    from app.providers.cloud import get_provider
    from app.providers.cloud import INSTRUCTIONS, TEMPLATE_REVISION
    from app.schema import provider_schema
    manifest_path = ROOT / "data/manifest.json"
    if args.split == "development" and not manifest_path.exists():
        manifest_path = ROOT / "data/dev_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = selected_records(manifest, args.split, args.limit)
    if args.ids:
        if args.split != "development":
            raise ValueError("Targeted prompt diagnostics are development-only")
        records = [r for r in records if r["id"] in args.ids]
        if len(records) != len(set(args.ids)):
            raise ValueError("Requested ID outside the development set")
    output = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    run_config = {"record_ids": [r["id"] for r in records], "split": args.split, "limit": args.limit, "concurrency": args.concurrency}
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Run sampling/concurrency changed; choose a new --run-name")
    config_path.write_text(json.dumps(run_config, indent=2))
    semaphore = asyncio.Semaphore(args.concurrency)
    ocr_engine = os.getenv("OCR_ENGINE", "tesseract")
    ocr_signature = hashlib.sha256(json.dumps({"engine": ocr_engine, "revision": 2, "tesseract": "ara+eng psm6 thread1", "paddle": "PP-OCRv5 mobile det+language mobile rec CPU thread1"}, sort_keys=True).encode()).hexdigest()
    cache_dir = ROOT / "results/ocr_cache" / ocr_engine
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Build this ONCE before both text providers; no provider-specific OCR changes.
    ocrs = {}
    if any(p.startswith("ocr_") for p in args.providers):
        from app.providers.ocr import run_ocr
        for r in records:
            cache = cache_dir / f'{r["id"]}.json'
            if cache.exists():
                ocrs[r["id"]] = json.loads(cache.read_text())
            if r["id"] not in ocrs or ocrs[r["id"]].get("image_sha256") != r["sha256"] or ocrs[r["id"]].get("config_signature") != ocr_signature:
                with Profile() as profile:
                    result = await asyncio.to_thread(run_ocr, (ROOT / r["image"]).read_bytes(), ocr_engine, r["language"])
                result["profile"] = profile.result
                result["image_sha256"] = r["sha256"]
                result["config_signature"] = ocr_signature
                cache.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                ocrs[r["id"]] = result

    def experiment(record, provider):
        keys = {"ocr_ollama": "OLLAMA_MODEL", "ocr_nvidia": "NVIDIA_MODEL", "gemini": "GEMINI_MODEL", "modal_hybrid": "MODAL_MODEL_ID", "ocr_openrouter": "OPENROUTER_MODEL"}
        result = {"image_sha256": record["sha256"], "provider": provider, "model": os.getenv(keys[provider]),
                "prompt_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
                "schema_sha256": hashlib.sha256(json.dumps(provider_schema(), sort_keys=True).encode()).hexdigest(),
                "ocr_config_signature": ocr_signature if provider.startswith("ocr_") else None}
        if provider.startswith("ocr_"):
            result["text_template_revision"] = TEMPLATE_REVISION
        return result

    async def one(record, provider):
        destination = output / provider / f'{record["id"]}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            cached = json.loads(destination.read_text())
            if cached.get("experiment") != experiment(record, provider):
                raise ValueError("Cached prediction configuration changed; choose a new --run-name")
            return cached
        async with semaphore:
            started = time.perf_counter()
            result = {"id": record["id"], "dataset": record["dataset"], "language": record["language"], "provider": provider, "split": args.split,
                      "ocr": {"engine": ocr_engine, "cache": str((cache_dir / f'{record["id"]}.json').relative_to(ROOT))} if provider.startswith("ocr_") else None}
            result["experiment"] = experiment(record, provider)
            result["started_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                response = await get_provider(provider).extract((ROOT / record["image"]).read_bytes(), mimetypes.guess_type(record["image"])[0] or "image/png", ocr=ocrs.get(record["id"]))
                result.update(response)
                truth = json.loads((ROOT / record["annotation"]).read_text())
                result["metrics"] = evaluate(response["receipt"], truth)
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                # Never serialize arbitrary HTTP exception text: SDK errors may include API keys.
                result["error"] = type(exc).__name__
                if isinstance(exc, ProviderError):
                    result["error"] = str(exc)[:250]
                    result["metadata"] = exc.metadata
                result["metrics"] = evaluate({}, json.loads((ROOT / record["annotation"]).read_text()))
            result["wall_seconds"] = time.perf_counter() - started
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            print(provider, record["id"], result["status"], round(result["wall_seconds"], 2), flush=True)
            return result

    summaries = []
    budget = Budget(ROOT / "results/spend_ledger.json")
    # Keep profiles attributable: providers sequential; samples within a provider bounded.
    for provider in args.providers:
        pending = [r for r in records if args.overwrite or not (output / provider / f'{r["id"]}.json').exists()]
        invocation = f"{args.run_name}/{provider}/{uuid.uuid4().hex[:12]}"
        if pending:
            # Modal development/cold-start trials have their own reservation.
            # This is an additional inference reservation, not a reuse of it.
            budget.reserve(invocation, len(pending) * (0.08 if provider == "modal_hybrid" else 0.10))
        with Profile() as profile:
            rows = await asyncio.gather(*(one(r, provider) for r in records))
        costs = [inference_cost(r.get("metadata", {})) for r in rows]
        pending_ids = {r["id"] for r in pending}
        new_costs = [inference_cost(r.get("metadata", {})) for r in rows if r["id"] in pending_ids]
        if pending and all(c is not None for c in new_costs):
            budget.settle(invocation, sum(new_costs))
        elif pending and provider == "ocr_nvidia" and os.getenv("NVIDIA_MODEL") == "nvidia/nemotron-3.5-bolt-e4b" and os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1") == "https://integrate.api.nvidia.com/v1":
            budget._update(invocation, 0, "Official free developer endpoint; experimental cash estimate0, commercial inference price unavailable")
        elif pending and provider == "ocr_openrouter" and os.getenv("OPENROUTER_MODEL", "").endswith(":free"):
            budget._update(invocation, 0, "Fixed free OpenRouter variant; quota-limited experiment, production paid twin not measured")
        elif pending and provider != "modal_hybrid":
            # Retain a conservative per-receipt bound for responses with missing
            # usage (e.g. a503 followed by success), rather than assuming free retries.
            budget.reserve(invocation, sum(c if c is not None else 0.10 for c in new_costs))
        summary = {"provider": provider, "split": args.split, "concurrency": args.concurrency, "count": len(rows), "succeeded": sum(r["status"] == "ok" for r in rows),
                   "profile": profile.result, "receipts_per_minute": len(rows) * 60 / profile.result["wall_seconds"], "ocr_excluded_from_profile": provider.startswith("ocr_"),
                   "estimated_inference_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                   "new_requests": len(pending), "timed_run_complete": len(pending) == len(records),
                   "cached_rows": sum((output / provider / f'{r["id"]}.json').exists() and r.get("wall_seconds", 0) > profile.result["wall_seconds"] for r in rows)}
        (output / provider / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--providers", nargs="+", default=["ocr_ollama", "ocr_nvidia", "gemini", "modal_hybrid"])
    p.add_argument("--split", choices=["development", "test"], default="test")
    p.add_argument("--limit", type=int, choices=[1, 2, 8, 10, 50], default=10)
    p.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--run-name", default="test10")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ids", nargs="+", help="Targeted development diagnostics only")
    args = p.parse_args()
    # Initial experiment maximum, including up to one repair/request. This is
    # a conservative bound; actual token costs and Modal resource costs reported later.
    estimate = len(args.providers) * args.limit * (0.04 if "modal_hybrid" not in args.providers else 0.08)
    if estimate > 5:
        p.error("Conservative run estimate exceeds $5. Split the run after explicit spending approval.")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
