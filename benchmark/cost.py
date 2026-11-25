"""Usage-based list-price estimates and a conservative cumulative experiment cap."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATES = {"gemma4:31b": (0.14, 0.40), "gemma4:cloud": (0.14, 0.40), "gpt-oss:20b": (0.07, 0.30), "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50)}
MODAL_RATES = {"T4": 0.000164, "L4": 0.000222, "cpu_core": 0.0000131, "ram_gib": 0.00000222}


def token_cost(metadata, input_rate, output_rate):
    if not metadata.get("usage_complete", False):
        return None
    # Provider adapters already include thoughts once in output_tokens.
    return (metadata.get("input_tokens", 0) * input_rate + metadata.get("output_tokens", 0) * output_rate) / 1_000_000


def project_cost(per_receipt, volume):
    return None if per_receipt is None else per_receipt * volume


class Budget:
    """Reserve upper bounds before paid actions, settle only with reliable usage."""
    def __init__(self, path, limit=5.0):
        self.path, self.limit = Path(path), limit

    def _update(self, name, value, status):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            content = handle.read()
            data = json.loads(content) if content else {"limit_usd": self.limit, "entries": {}}
            entry = {"usd": value, "status": status}
            projected = sum(e["usd"] for k, e in data["entries"].items() if k != name) + value
            if projected > data["limit_usd"] + 1e-9:
                raise ValueError("Cumulative reserved experiment spending would exceed $5; user approval required")
            data["entries"][name] = entry
            data["reserved_or_estimated_usd"] = projected
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle, indent=2)
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)

    def reserve(self, name, upper_bound_usd):
        self._update(name, upper_bound_usd, "reserved upper bound, not actual charge")

    def settle(self, name, estimated_usd):
        self._update(name, estimated_usd, "usage × official list price; billing not verified")


def inference_cost(metadata):
    if metadata.get("provider") == "ocr_openrouter" and metadata.get("model", "").endswith(":free"):
        # Verified fixed free variants have zero token rates even when an error
        # omits token usage; this is quota-limited trial cost, not commercial pricing.
        return metadata.get("reported_cost_usd") or 0.0
    model = metadata.get("model", "")
    if model in RATES:
        return token_cost(metadata, *RATES[model])
    # NVIDIA developer credits have no verified production list price.
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test10")
    args = parser.parse_args()
    rows = []
    for directory in sorted((ROOT / "results" / args.run_name).glob("*/")):
        receipts = [json.loads(f.read_text()) for f in directory.glob("*.json") if f.name != "summary.json"]
        if not receipts:
            continue
        costs = [inference_cost(r.get("metadata", {})) for r in receipts]
        mean = sum(costs) / len(costs) if all(c is not None for c in costs) else None
        for volume in (10, 50, 100, 1000, 10000, 100000):
            rows.append({"provider": directory.name, "volume": volume, "inference_cost_usd_estimated": project_cost(mean, volume), "measured_receipts": len(receipts), "basis": "actual tokens × official list price; excludes infrastructure; N/A if usage/price unknown"})
    with (ROOT / "results/cost_projections.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider", "volume", "inference_cost_usd_estimated", "measured_receipts", "basis"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/cost_projections.csv")


if __name__ == "__main__":
    main()
