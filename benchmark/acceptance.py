"""Explicit receipt acceptance and inference cost per accepted output."""
import argparse
import csv
import json
from pathlib import Path

from pydantic import ValidationError

from app.schema import Receipt
from benchmark.cost import inference_cost
from benchmark.metrics import evaluate
from benchmark.report import PROVIDERS, has_model_output, read, selected_records

ROOT = Path(__file__).resolve().parents[1]


def acceptance(row: dict) -> dict:
    """Strictness applies only to annotation coverage, never unknown fields.

    Unavailable providers without generated output supply no model-quality evidence.
    Generated but invalid output fails if the ground truth has any covered fields.
    """
    metrics = row.get("metrics", {})
    scalar_count = metrics.get("field_total", 0) or 0
    item_count = sum(value.get("total", 0) or 0 for value in metrics.get("item_fields", {}).values())
    if not has_model_output(row):
        return {"strict_pass": None, "scorable": False, "critical_total_pass": None, "reason": "provider unavailable or no generated output; model quality unscorable"}
    if scalar_count + item_count == 0:
        return {"strict_pass": None, "scorable": False, "critical_total_pass": None, "reason": "all relevant fields unannotated; no quality pass can be claimed"}
    valid_final = row.get("status") == "ok"
    if valid_final:
        try:
            Receipt.model_validate(row.get("receipt"))
        except (ValidationError, TypeError):
            valid_final = False
    total_metric = metrics.get("fields", {}).get("total", {})
    total_covered = (total_metric.get("total", 0) or 0) > 0
    critical = (valid_final and total_metric.get("correct", 0) == total_metric["total"]) if total_covered else None
    scalar_ok = metrics.get("field_correct", 0) == scalar_count
    grouped_lines = metrics.get("line_items")
    lines_ok = grouped_lines is None or grouped_lines.get("f1") == 1
    strict = bool(valid_final and scalar_ok and lines_ok)
    failures = []
    if not valid_final:
        failures.append("invalid final receipt")
    if not scalar_ok:
        failures.append("annotated scalar error")
    if not lines_ok:
        failures.append("grouped line-item F1 below 1")
    reason = "; ".join(failures) if failures else "passes covered annotations only; unannotated fields remain unscorable"
    return {"strict_pass": strict, "scorable": True, "critical_total_pass": critical, "reason": reason}


def summarize_acceptance(rows: list[dict], economic_cost: float | None = None) -> dict:
    decisions = [acceptance(row) for row in rows]
    strict_passes = sum(item["strict_pass"] is True for item in decisions)
    critical_passes = sum(item["critical_total_pass"] is True for item in decisions)
    if economic_cost is None and rows:
        costs = [inference_cost(row.get("metadata", {})) for row in rows]
        if all(cost is not None for cost in costs):
            economic_cost = sum(costs)
    def price(accepted):
        if not accepted:
            return None, "no acceptable receipts"
        if economic_cost is None:
            return None, "unknown price, usage, or allocated lifecycle cost"
        return economic_cost / accepted, "all input inference cost divided by accepted receipts"
    strict_price, strict_reason = price(strict_passes)
    critical_price, critical_reason = price(critical_passes)
    return {"attempted_inputs": len(rows), "strict_scorable": sum(item["scorable"] for item in decisions), "strict_passes": strict_passes,
            "critical_total_scorable": sum(item["critical_total_pass"] is not None for item in decisions), "critical_total_passes": critical_passes,
            "economic_cost_usd": economic_cost, "cost_per_strict_accepted_usd": strict_price, "strict_cost_reason": strict_reason,
            "cost_per_critical_total_accepted_usd": critical_price, "critical_cost_reason": critical_reason}


def money(value):
    return "N/A" if value is None else f"${value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test10")
    args = parser.parse_args()
    run_root = ROOT / "results" / args.run_name
    manifest = read(ROOT / "data/manifest.json")
    records, split = selected_records(run_root, manifest)
    comparison = {entry["provider"]: entry for entry in read(ROOT / "results/comparison.json", [])}
    summaries = []
    detail = []
    extra = ["ocr_openrouter"] if (run_root / "ocr_openrouter").exists() else []
    for provider in PROVIDERS + extra:
        rows = []
        for record in records:
            row = read(run_root / provider / f'{record["id"]}.json')
            if row is None:
                continue
            # Recompute annotation-aware metrics from the stored original prediction,
            # avoiding stale derived metrics when the evaluator has been corrected.
            truth = read(ROOT / record["annotation"])
            row["metrics"] = evaluate(row.get("receipt") or {}, truth)
            rows.append(row)
            decision = acceptance(row)
            detail.append(f"| {provider} | {record['id']} | {decision['strict_pass']} | {decision['critical_total_pass']} | {decision['reason']} |")
        aggregate = comparison.get(provider, {})
        # Comparison can provide Modal allocated-lifecycle costs, which do not fit
        # the token-only cost helper. Reject an obviously stale/partial aggregate.
        models = {row.get("metadata", {}).get("model") for row in rows if row.get("metadata", {}).get("model")}
        matches = (aggregate.get("attempted", aggregate.get("evaluated")) == len(rows)
                   and aggregate.get("expected_count", len(records)) == len(records)
                   and (not models or models == {aggregate.get("model")}))
        cost = aggregate.get("cost") if matches else None
        summary = summarize_acceptance(rows, economic_cost=cost)
        summary.update(provider=provider, expected_inputs=len(records), complete=len(rows) == len(records),
                       experimental_inference_cash_usd_estimated=0.0 if provider == "ocr_nvidia" and rows else None,
                       commercial_price_known=summary["economic_cost_usd"] is not None)
        summaries.append(summary)
    destination = ROOT / "results/acceptance.csv"
    with destination.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    text = f"# Cost per acceptable receipt\n\nRun `{args.run_name}`; split `{split}`; {len(records)} fixed selected inputs per provider. Counts below use only saved predictions; missing providers remain unmeasured. These are explicit experimental acceptance rules, not a client-approved production SLA.\n\n"
    text += "**Strict:** valid final Pydantic receipt with status `ok`, every annotated scalar correct, and line-item F1 exactly 1 when ground truth supports grouped items. At least one annotated scalar or product field is required. Unannotated fields remain unscorable; passing a receipt with only its total annotated does not certify its merchant, date, tax, or products.\n\n"
    text += "**Critical-total sensitivity:** valid final receipt plus correct annotated receipt total. If total is not annotated, the rule is N/A. This deliberately weaker rule does not establish item or other-field quality. Unavailable providers without generated output are N/A; generated invalid output fails.\n\n"
    text += "| Provider | Saved/selected | Strict passes/scorable | Critical-total passes/scorable | All-input inference cost, economic estimate | Cost/strict accepted | Cost/critical-total accepted |\n|---|---|---|---|---|---|---|\n"
    for row in summaries:
        text += f"| {row['provider']} | {row['attempted_inputs']}/{row['expected_inputs']} | {row['strict_passes']}/{row['strict_scorable']} | {row['critical_total_passes']}/{row['critical_total_scorable']} | {money(row['economic_cost_usd'])} | {money(row['cost_per_strict_accepted_usd'])} | {money(row['cost_per_critical_total_accepted_usd'])} |\n"
    text += "\nCost per accepted receipt divides **all saved input inference cost, including generated failures and repairs**, by accepted outputs. It does not discard spending on rejected receipts. Unknown prices or usage produce N/A; zero accepted receipts means no acceptable output and no finite cost per accepted receipt. Raw attempt metadata supplies hosted token costs; the comparison aggregate can supply Modal lifecycle estimates when its model and counts match this run. Regenerate `python -m benchmark.report --run-name " + args.run_name + "` before this command to keep lifecycle estimates current.\n\n"
    text += "Gemini uses paid list-price inference even if the account has promotional credits; thoughts/retries are included once in recorded usage. NVIDIA developer inference is treated separately as an **estimated $0 experimental inference debit** under the tested free developer access; commercial inference price is unknown, so its economic cost per accepted receipt remains N/A. This does not make local OCR free: monthly infrastructure, measured OCR compute, operations and human review are excluded here and must be added for total operating cost. No invoice debit is asserted without billing evidence. See [cost analysis](cost_analysis.md).\n\n"
    text += "Partial runs cannot establish a complete ten- or fifty-input acceptance result. Do not rank a provider on a smaller/easier saved subset. Full annotation coverage is also limited: AraReceipt has no reliable grouped product ground truth, whereas CORD has item annotations. The strict and total-only denominators therefore answer different questions. A new commercial acceptance threshold must be selected with the client before promoting these POC results.\n\n"
    text += "| Provider | Strict cost qualification | Total-rule cost qualification |\n|---|---|---|\n"
    for row in summaries:
        text += f"| {row['provider']} | {row['strict_cost_reason']} | {row['critical_cost_reason']} |\n"
    text += "\n## Per-receipt decisions\n\n| Provider | Receipt | Strict | Critical total | Reason |\n|---|---|---|---|---|\n" + "\n".join(detail) + "\n"
    (ROOT / "reports/acceptance.md").write_text(text)
    print("Generated reports/acceptance.md and results/acceptance.csv")


if __name__ == "__main__":
    main()
