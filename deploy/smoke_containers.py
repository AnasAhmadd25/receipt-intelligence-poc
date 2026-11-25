"""Start one local A/B container and exercise its real default extraction API.

Secrets are read locally and passed to Docker by environment variable name only.
The smoke test makes one authenticated hosted extraction call and leaves the
container running so its interactive documentation remains available.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schema import Receipt


VARIANTS = {
    "a": {"port": 8001, "provider": "ocr_nvidia", "keys": ["API_KEY", "NVIDIA_API_KEY", "NVIDIA_MODEL"]},
    "b": {"port": 8002, "provider": "gemini", "keys": ["API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL"]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--image", required=True, help="Final built Docker image tag")
    parser.add_argument("--receipt", default="data/images/ara_0077.jpg")
    args = parser.parse_args()
    settings = VARIANTS[args.variant]
    receipt_path = (ROOT / args.receipt).resolve()
    image_bytes = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(image_bytes).hexdigest()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = json.loads((ROOT / "data/dev_manifest.json").read_text())["development"]
    entry = next((item for item in entries if (ROOT / item["image"]).resolve() == receipt_path), None)
    if entry is None or entry["sha256"] != receipt_sha256 or entry["dataset"] not in {"AraReceipt", "CORD-v2"}:
        raise SystemExit("Smoke tests require an unchanged public benchmark image listed in data/dev_manifest.json")
    source = manifest["datasets"][entry["dataset"]]
    env = {**os.environ, **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}}
    if not env.get("API_KEY", "").strip():
        raise SystemExit("API_KEY must be configured before a smoke test")
    if args.variant == "b":
        env["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
    missing = [name for name in settings["keys"] if not env.get(name)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    secrets = [value for name, value in env.items() if ("KEY" in name or "TOKEN" in name or "SECRET" in name) and len(value) >= 6]

    def safe(value):
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value

    def docker(*arguments, check=True):
        result = subprocess.run(["docker", *arguments], env=env, capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(safe(result.stderr.strip()) or "Docker command failed")
        return result

    image_info = json.loads(docker("image", "inspect", "--format", "{{json .Id}}", args.image).stdout)
    image_platform = docker("image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", args.image).stdout.strip()
    docker_host_architecture = docker("info", "--format", "{{.Architecture}}").stdout.strip()
    name = f"receipt-intelligence-api-{args.variant}"
    exists = docker("container", "inspect", "--format", '{{index .Config.Labels "receipt-intelligence.smoke"}}', name, check=False)
    if exists.returncode == 0:
        if exists.stdout.strip() != args.variant:
            raise SystemExit("Container name is already used by a container this script does not own")
        docker("container", "rm", "--force", name)
    arguments = ["run", "--detach", "--name", name, "--label", f"receipt-intelligence.smoke={args.variant}",
                 "-p", f"127.0.0.1:{settings['port']}:8000"]
    if args.variant == "a":
        arguments += ["--platform", "linux/amd64", "-e", "OCR_LANGUAGE=ar",
                      "-v", "receipt-intelligence-paddle-models:/home/receipt/.paddlex",
                      "-v", "receipt-intelligence-ocr-cache:/home/receipt/.cache"]
    for key in settings["keys"]:
        arguments += ["-e", key]
    arguments.append(args.image)
    container_id = docker(*arguments).stdout.strip()
    report = {
        "variant": args.variant, "started_at": datetime.now(timezone.utc).isoformat(),
        "image_tag": args.image, "image_id": image_info, "container_id": container_id,
        "image_platform": image_platform, "docker_host_architecture": docker_host_architecture,
        "performance_scope": "Local smoke test only; cross-architecture emulation timings are not native amd64 performance",
        "base_url": f"http://127.0.0.1:{settings['port']}", "default_provider_expected": settings["provider"],
        "receipt_id": receipt_path.stem, "receipt_sha256": receipt_sha256,
        "public_dataset": {"name": entry["dataset"], "source": source["card"], "revision": source["revision"],
                           "license": source["license"], "split": entry["source_split"], "row_index": entry["source_row_index"]},
        "checks": {}, "extraction": {}, "success": False,
    }
    output = ROOT / "results" / f"container_smoke_{args.variant}.json"

    def response_evidence(response):
        try:
            body = response.json()
        except ValueError:
            body = {"non_json_body_preview": response.text[:2000]}
        return {"status_code": response.status_code,
                "headers": {name: response.headers[name] for name in ("content-type", "content-length", "date", "server") if name in response.headers},
                "body": safe(body)}

    try:
        with httpx.Client(base_url=report["base_url"], trust_env=False, timeout=5) as client:
            deadline = time.monotonic() + 120
            while True:
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Container did not become healthy within 120 seconds")
                time.sleep(2)
            report["checks"]["health"] = response_evidence(response)
            for label, path in (("ready", "/ready"), ("docs", "/docs"), ("openapi", "/openapi.json")):
                response = client.get(path)
                report["checks"][label] = {"status_code": response.status_code,
                                           "content_type": response.headers.get("content-type")}
                if label == "ready":
                    report["checks"][label]["body"] = safe(response.json())
                if label == "openapi" and response.status_code == 200:
                    document = response.json()
                    operation = document["paths"]["/v1/extract"]["post"]
                    report["checks"][label].update(
                        security=operation.get("security"),
                        request_content_types=list(operation.get("requestBody", {}).get("content", {})),
                        response_schema=operation["responses"]["200"]["content"]["application/json"]["schema"],
                        provider_enum=document["components"]["schemas"]["ExtractionInput"]["properties"]["provider"]["enum"],
                    )
            unauthenticated = client.post("/v1/extract", files={"file": (receipt_path.name, image_bytes, "image/jpeg")})
            report["checks"]["unauthenticated_extract"] = response_evidence(unauthenticated)
            if unauthenticated.status_code != 401:
                raise RuntimeError("Unauthenticated extraction was not rejected with 401")
            started = time.perf_counter()
            response = client.post("/v1/extract", headers={"X-API-Key": env["API_KEY"]},
                                   files={"file": (receipt_path.name, image_bytes, "image/jpeg")}, timeout=360)
            report["extraction"] = response_evidence(response)
            report["extraction"]["wall_seconds"] = time.perf_counter() - started
            if response.status_code == 200:
                body = response.json()
                receipt = Receipt.model_validate(body["receipt"])
                report["extraction"]["receipt_schema_valid"] = True
                report["extraction"]["has_observed_content"] = bool(receipt.raw_text.strip() or receipt.merchant_name or receipt.total is not None or receipt.line_items)
                report["extraction"]["observed_provider"] = body["metadata"].get("provider")
                report["success"] = (report["extraction"]["has_observed_content"] and body["metadata"].get("provider") == settings["provider"]
                                     and all(report["checks"][label]["status_code"] == 200 for label in ("health", "ready", "docs", "openapi")))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": safe(str(exc))}
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"variant": args.variant, "success": report["success"],
                          "status_code": report["extraction"].get("status_code"), "report": str(output)}))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
