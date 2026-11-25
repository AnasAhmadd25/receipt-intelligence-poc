"""Development-only HTTP driver, avoiding intermittent local Modal gRPC routing."""
import argparse
import base64
import json
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    entries = json.loads((root / "data/dev_manifest.json").read_text())["development"]
    groups = {}
    for entry in entries:
        groups.setdefault(entry["dataset"], []).append(entry)
    selected = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                selected.append(group.pop(0))
    destination = root / "results" / ("modal_dev_" + args.model.split("/")[-1] + ".json")
    rows = []
    with httpx.Client(timeout=480, follow_redirects=True, trust_env=False) as client:
        for entry in selected[:args.limit]:
            started = time.monotonic()
            response = client.post(os.environ["MODAL_ENDPOINT"].rstrip("/") + "/v1/extract",
                                   headers={"Authorization": "Bearer " + os.environ["MODAL_API_KEY"]},
                                   json={"image_b64": base64.b64encode((root / entry["image"]).read_bytes()).decode(),
                                         "mime_type": "image/jpeg"})
            result = response.json() if response.status_code == 200 else {"receipt": None,
                       "metadata": {"json_valid": False, "http_status": response.status_code},
                       "raw_output": response.text[:4000]}
            result.update({"sample_id": entry["id"], "client_wall_seconds": time.monotonic() - started})
            if response.status_code == 200 and result["metadata"]["model"] != args.model:
                raise RuntimeError("Deployed model differs from requested experiment model")
            rows.append(result)
            destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            print({"sample_id": entry["id"], "http_status": response.status_code,
                   "json_valid": result["metadata"].get("json_valid"),
                   "seconds": round(result["client_wall_seconds"], 3)}, flush=True)


if __name__ == "__main__":
    main()
