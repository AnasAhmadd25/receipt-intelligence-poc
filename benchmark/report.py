"""Generate evidence-linked comparison, resource CSV and conditional sizing guidance."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from benchmark.cost import inference_cost, MODAL_RATES

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ["ocr_ollama", "ocr_nvidia", "gemini", "modal_hybrid"]
LABELS = ["OCR + Ollama Cloud", "OCR + NVIDIA", "Gemini Flash-Lite", "Modal Hybrid"]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(value, percent=False, digits=3):
    if value is None:
        return "N/A"
    if isinstance(value, int) and not percent:
        return str(value)
    if isinstance(value, (float, int)):
        return f"{value * 100:.1f}%" if percent else f"{value:,.{digits}f}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows) + "\n"


def ratio(correct, total):
    return correct / total if total else None


def selected_records(run_root, manifest):
    """Use the runner's frozen selection; older runs retain their original ten."""
    config = read(run_root / "run_config.json")
    if config is None:
        ids, split = manifest["initial_test_ids"], "test"
    else:
        ids, split = config["record_ids"], config["split"]
    available = {record["id"]: record for record in manifest.get(split, [])}
    if len(set(ids)) != len(ids) or any(identity not in available for identity in ids):
        raise ValueError("Run configuration has duplicate or unknown receipt IDs")
    return [available[identity] for identity in ids], split


def has_model_output(row):
    """Access/network errors without generation are not model accuracy evidence."""
    if row.get("status") == "ok" or row.get("receipt"):
        return True
    metadata = row.get("metadata", {})
    if any(row.get(key) or metadata.get(key) for key in ("raw_output", "raw_response")):
        return True
    if metadata.get("output_tokens", 0):
        return True
    return any(attempt.get("raw_response") or attempt.get("raw_output") or attempt.get("output_tokens", 0)
               for attempt in metadata.get("attempts", []))


def aggregate_field(rows, name, item=False):
    key = "item_fields" if item else "fields"
    values = [r.get("metrics", {}).get(key, {}).get(name, {}) for r in rows if has_model_output(r)]
    c, n = sum(v.get("correct", 0) for v in values), sum(v.get("total", 0) for v in values)
    return ratio(c, n), c, n


def modal_duration(rows):
    groups = {}
    for r in rows:
        m = r.get("metadata", {})
        resource = m.get("resource", {})
        identifier = m.get("container_id")
        if not identifier:
            return None, 0
        elapsed = resource.get("container_uptime_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            return None, 0
        groups[identifier] = max(groups.get(identifier, 0), elapsed)
    return sum(t + 2 for t in groups.values()) if groups else None, len(groups)


def modal_allocation_rate(resource):
    """Recorded allocation, not measured utilization, determines list-price rate."""
    gpu = resource.get("allocated_gpu")
    cpu = resource.get("allocated_cpu_physical_cores")
    ram = resource.get("allocated_ram_mib")
    region = resource.get("region_selection")
    multiplier = resource.get("region_price_multiplier")
    if gpu not in ("T4", "L4") or not region:
        return None
    if not all(isinstance(value, (int, float)) and value > 0 for value in (cpu, ram, multiplier)):
        return None
    return (MODAL_RATES[gpu] + cpu * MODAL_RATES["cpu_core"] + ram / 1024 * MODAL_RATES["ram_gib"]) * multiplier


def modal_resource_cost(rows):
    if not rows:
        return None
    groups = {}
    for row in rows:
        metadata = row.get("metadata", {})
        resource = metadata.get("resource", {})
        identity = metadata.get("container_id")
        elapsed = resource.get("container_uptime_seconds")
        rate = modal_allocation_rate(resource)
        if not identity or rate is None or not isinstance(elapsed, (int, float)) or elapsed < 0:
            return None
        if identity in groups and groups[identity][1] != rate:
            return None
        groups[identity] = (max(elapsed, groups.get(identity, (0, rate))[0]), rate)
    return sum((elapsed + 2) * rate for elapsed, rate in groups.values())


def modal_allocation_label(rows):
    resources = [row.get("metadata", {}).get("resource", {}) for row in rows]
    labels = set()
    for resource in resources:
        if modal_allocation_rate(resource) is None:
            return "N/A (allocation evidence unavailable)"
        labels.add(f'{resource["allocated_gpu"]} + {resource["allocated_cpu_physical_cores"]} physical CPU + {resource["allocated_ram_mib"] / 1024:g}GiB ({resource["region_selection"]})')
    return "; ".join(sorted(labels)) if labels else "N/A (no measured allocation)"


def summarize(root, provider, expected):
    folder = root / provider
    rows = [read(folder / f'{r["id"]}.json') for r in expected]
    rows = [r for r in rows if r]
    summary = read(folder / "summary.json", {})
    if not rows:
        return {"provider": provider, "rows": [], "attempted": 0, "evaluated": 0, "unavailable": 0, "successes": 0, "field_accuracy": None, "json_validity": None, "complete_expected": False, "expected_count": len(expected)}
    evaluated_rows = [row for row in rows if has_model_output(row)]
    ocrs = [read(ROOT / r["ocr"]["cache"], {}) if r.get("ocr") else {} for r in rows]
    c = sum(r["metrics"]["field_correct"] for r in evaluated_rows)
    n = sum(r["metrics"]["field_total"] for r in evaluated_rows)
    lang = {}
    for label, dataset in [("arabic", "AraReceipt"), ("non_arabic", "CORD-v2")]:
        subset = [r for r in evaluated_rows if r["dataset"] == dataset]
        lang[label] = ratio(sum(r["metrics"]["field_correct"] for r in subset), sum(r["metrics"]["field_total"] for r in subset))
    li = [r["metrics"]["line_items"] for r in evaluated_rows if r["metrics"].get("line_items") is not None]
    matched, predicted, expected_items = (sum(x[k] for x in li) for k in ("matched", "predicted", "expected"))
    line_f1 = 2 * matched / (predicted + expected_items) if li and predicted + expected_items else None
    latencies = [r["wall_seconds"] + o.get("latency_seconds", 0) for r, o in zip(rows, ocrs)]
    api_cpu = summary.get("profile", {}).get("cpu_core_seconds") if summary.get("timed_run_complete") else None
    ocr_cpu = sum(o.get("profile", {}).get("cpu_core_seconds", 0) for o in ocrs)
    profile = summary.get("profile", {})
    peaks = [profile.get("peak_ram_mib", 0)] + [o.get("profile", {}).get("peak_ram_mib", 0) for o in ocrs]
    costs = [inference_cost(r.get("metadata", {})) for r in rows]
    total_cost = sum(costs) if all(c is not None for c in costs) else None
    modal_seconds, containers = modal_duration(rows) if provider == "modal_hybrid" else (None, None)
    if provider == "modal_hybrid":
        total_cost = modal_resource_cost(rows)
    modal_resources = [r.get("metadata", {}).get("resource", {}) for r in rows]
    if provider == "modal_hybrid":
        gpu_cpu = sum(m.get("cpu_core_seconds", 0) for m in modal_resources)
        init_cpu = sum(r.get("metadata", {}).get("initialization_resource", {}).get("cpu_core_seconds", 0) for r in rows if r.get("metadata", {}).get("cold_first_request"))
        cpu_each = (gpu_cpu + init_cpu) / len(rows) if any(modal_resources) else None
        peak_ram = max((m.get("peak_ram_bytes", 0) / 2**20 for m in modal_resources), default=0) or None
    else:
        cpu_each = (api_cpu + ocr_cpu) / len(rows) if api_cpu is not None else None
        peak_ram = max(peaks) or None
    return {"provider": provider, "rows": rows, "attempted": len(rows), "evaluated": len(evaluated_rows), "unavailable": len(rows) - len(evaluated_rows), "successes": sum(r["status"] == "ok" for r in rows),
            "complete_expected": bool(expected) and len(evaluated_rows) == len(expected), "expected_count": len(expected),
            "model": next((r.get("metadata", {}).get("model") for r in rows if r.get("metadata", {}).get("model")), rows[0].get("experiment", {}).get("model")),
            "field_accuracy": ratio(c, n), "field_correct": c, "field_total": n, **lang,
            "service_field_accuracy": ratio(sum(r["metrics"]["field_correct"] for r in rows), sum(r["metrics"]["field_total"] for r in rows)),
            "line_f1": line_f1, "line_matched": matched, "line_predicted": predicted, "line_expected": expected_items,
            "json_validity": mean([int(r.get("metadata", {}).get("json_valid", False)) for r in evaluated_rows]),
            "final_validity": mean([int(r["status"] == "ok") for r in evaluated_rows]),
            "service_final_validity": mean([int(r["status"] == "ok") for r in rows]),
            "mean_latency": mean(latencies), "median_latency": statistics.median(latencies),
            "api_mean_latency": mean([r["wall_seconds"] for r in rows]),
            "ocr_mean_latency": mean([o.get("latency_seconds") for o in ocrs]),
            "cpu_each": cpu_each, "peak_ram_mib": peak_ram,
            "gpu_peak_mib": max(((m.get("gpu_peak_allocated_bytes") or 0) / 2**20 for m in modal_resources), default=0) or None,
            "rpm": len(rows) * 60 / sum(latencies) if summary.get("timed_run_complete") else None,
            "cost": total_cost, "unit_cost": total_cost / len(rows) if total_cost is not None else None,
            "modal_allocated_seconds_estimated": modal_seconds, "modal_containers": containers,
            "repairs": sum(r.get("metadata", {}).get("repairs", 0) for r in rows),
            "input_tokens": sum(r.get("metadata", {}).get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("metadata", {}).get("output_tokens", 0) for r in rows),
            "thinking_tokens": sum(r.get("metadata", {}).get("thinking_tokens", 0) for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test10")
    args = parser.parse_args()
    manifest = read(ROOT / "data/manifest.json")
    expected, split = selected_records(ROOT / "results" / args.run_name, manifest)
    results = [summarize(ROOT / "results" / args.run_name, p, expected) for p in PROVIDERS]
    blocks = read(ROOT / "results/provider_blocks.json", {})
    for result in results:
        block = blocks.get(result["provider"])
        if block and not result["attempted"]:
            result["model"] = block.get("model", "") + " (blocked HTTP " + str(block.get("http_status")) + ")"
    (ROOT / "results/comparison.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in results], indent=2))
    successful = [r for r in results if r["successes"] and r.get("field_accuracy") is not None and r.get("complete_expected")]
    winner = max(successful, key=lambda r: (r["field_accuracy"], r.get("line_f1") or 0)) if successful else None
    arabic_count = sum(record.get("dataset") == "AraReceipt" for record in expected)
    cord_count = sum(record.get("dataset") == "CORD-v2" for record in expected)
    intro = f"# Receipt Intelligence benchmark\n\nVerified on 2026-11-25. Prepared **50 held-out receipts plus 8 disjoint development receipts**; run `{args.run_name}` selects **{len(expected)} {split} receipts ({arabic_count} Arabic, {cord_count} CORD)**. Attempted and evaluated counts below show each provider's observed coverage. "
    intro += "Unavailable requests with no model output have N/A model accuracy and JSON validity; generated invalid JSON counts as an evaluated failure. Service accuracy includes failed requests, and only providers with model output for the entire selection are ranked. Ground truth comes from dataset annotations, never model-generated labels.\n\n"
    if winner:
        intro += f"The highest measured annotated scalar-field accuracy was **{winner['model']} ({fmt(winner['field_accuracy'], True)})**. This is a small POC result, not a production accuracy guarantee. Compare line-item accuracy separately below.\n\n"
    metrics = [("Selected model", "model", False), ("Attempted receipts", "attempted", False), ("Evaluated model responses", "evaluated", False), ("Unavailable/no model output", "unavailable", False), ("Valid final receipts", "successes", False), ("Model field accuracy (scalar micro)", "field_accuracy", True), ("Service field accuracy (all attempts)", "service_field_accuracy", True), ("Arabic scalar accuracy", "arabic", True), ("Non-Arabic scalar accuracy", "non_arabic", True), ("Line-item F1 (CORD only)", "line_f1", True), ("JSON/schema validity, first attempt", "json_validity", True), ("JSON/schema validity, final", "final_validity", True), ("Average latency, serial end-to-end seconds", "mean_latency", False), ("Median latency seconds", "median_latency", False), ("OCR average seconds", "ocr_mean_latency", False), ("API/client average seconds", "api_mean_latency", False), ("CPU core-seconds/receipt", "cpu_each", False), ("Peak RAM MiB", "peak_ram_mib", False), ("GPU allocated peak MiB", "gpu_peak_mib", False), ("Receipts/minute, serial measured stages", "rpm", False), ("Usage-based cost for evaluated test set USD (estimate)", "cost", False), ("Input tokens incl repairs", "input_tokens", False), ("Output tokens incl reasoning/repairs", "output_tokens", False), ("Thinking tokens (subset of output)", "thinking_tokens", False), ("JSON repairs", "repairs", False)]
    rows = [[label] + [fmt(r.get(key), percent, 5 if key == "cost" else 3) for r in results] for label, key, percent in metrics]
    for volume in (50, 1000):
        rows.append([f"Projected inference cost for {volume} USD (estimated)"] + [fmt(r.get("unit_cost") * volume if r.get("unit_cost") is not None else None, digits=5) for r in results])
    modal_rows = next((result["rows"] for result in results if result["provider"] == "modal_hybrid"), [])
    rows += [["Recommended cloud machine / observed Modal allocation", "2vCPU/4GiB Linux, OCR concurrency 1 (estimate)", "2vCPU/4GiB Linux, OCR concurrency 1 (estimate)", "1vCPU/2GiB CPU proxy (estimate)", modal_allocation_label(modal_rows)], ["Monthly infrastructure USD", "24 + tokens (DO example)", "24 + unknown API price", "12 + tokens (DO example)", "Scale-zero variable; no fixed monthly GPU"]]
    report = intro + table(["Metric"] + LABELS, rows)
    field_rows = []
    for field in ["merchant_name", "purchase_date", "receipt_number", "total", "subtotal", "tax", "discount"]:
        field_rows.append([field] + [f"{fmt(aggregate_field(r['rows'],field)[0],True)} ({aggregate_field(r['rows'],field)[1]}/{aggregate_field(r['rows'],field)[2]})" for r in results])
    for field in ["raw_description", "quantity", "unit_price", "line_total"]:
        field_rows.append(["line_items." + field] + [f"{fmt(aggregate_field(r['rows'],field,True)[0],True)} ({aggregate_field(r['rows'],field,True)[1]}/{aggregate_field(r['rows'],field,True)[2]})" for r in results])
    report += "\n## Annotated field denominators\n\n" + table(["Field"] + LABELS, field_rows)
    missing = []
    for r in results:
        numerator = sum(len(x["metrics"]["missing_fields"]) for x in r["rows"] if has_model_output(x))
        denominator = sum(x["metrics"]["missing_denominator"] for x in r["rows"] if has_model_output(x))
        hallucinations = sum(len(x["metrics"]["hallucinated_fields"]) for x in r["rows"] if has_model_output(x))
        absence = sum(x["metrics"]["hallucination_denominator"] for x in r["rows"] if has_model_output(x))
        missing.append([r["provider"], fmt(ratio(numerator,denominator), True), f"{numerator}/{denominator}", fmt(ratio(hallucinations,absence),True), f"{hallucinations}/{absence}"])
    report += "\n## Missing fields and hallucination limits\n\n" + table(["Provider", "Missing annotated-field rate", "Count", "Hallucination rate", "Known-absent labels"], missing)
    report += "\nHallucination is **unscorable** where the dataset supplies no explicit known-absent fields. Unannotated predictions are retained as unscorable, never automatically called hallucinations. Wrong annotated values are accuracy errors. Raw predictions and per-receipt metrics remain in `results/" + args.run_name + "/`.\n"
    report += "\n## Interpretation and reproducibility\n\n- AraReceipt is Arabic/mixed Arabic-English with region classes and no reliable product-row grouping. CORD is Indonesian/Latin-script with mixed English terms, not an English-only benchmark.\n- Missing annotations are excluded. CORD merchant/date/currency are unscorable. Full-line F1 requires one-to-one matching, description similarity ≥0.8 and equality of each annotated numeric field; diagnostic field accuracy pairs descriptions one-to-one independently of amounts.\n- Field accuracy is micro-averaged annotated scalar fields; the Arabic/non-Arabic columns have different coverage and are not equivalent difficulty.\n- Text-provider latency adds the same cached measured OCR stage to live API latency. Cached OCR is shared byte-for-byte across Ollama/NVIDIA. OCR and API resource profiles are separate stages; reported RAM is their maximum, not an invented sum. API waiting CPU is not provider inference CPU.\n- JSON validity is schema validity of the original response; final validity includes any permitted repair. All attempts and usage are retained.\n- Costs are actual reported usage multiplied by official list rates, **not verified cash debits/invoices**. NVIDIA commercial rates are unavailable. Modal allocated-seconds estimates include observed container initialization, inference and 2s terminal idle; scheduler/pre-Python startup may be missing.\n- CPU/RAM measurements are local Apple M1 Pro unless explicitly marked Modal Linux. GPU values follow recorded remote allocation metadata; missing GPU measurements remain N/A. Concurrency evidence is below.\n- Dataset licenses/annotation conversion limitations: [dataset audit](dataset_audit.md). Provider/model/pricing references: [sources](sources.md).\n"
    ocrrows=[]
    for engine in ("tesseract", "paddle"):
        for c in (1,2,4):
            data=read(ROOT/f"results/ocr_development/{engine}/summary_c{c}.json")
            if data:
                p=data["profile"]
                ocrrows.append([engine,c,data["succeeded"],fmt(data["metrics"]["annotated_token_recall"],True),fmt(data["metrics"]["annotated_numeric_token_recall"],True),fmt(data["receipts_per_minute"]),fmt(p["peak_ram_mib"]),fmt(p["cpu_core_seconds"]/data["count"])])
    report += "\n## Development OCR/resource experiments\n\n" + table(["CPU OCR", "Concurrency", "Completed", "Annotated token recall", "Numeric-token recall", "Receipts/min", "Peak MiB", "Core-sec/receipt"],ocrrows)
    report += "\nThese are partial-transcription recall proxies, not full CER. The selected PP-OCRv5 route uses the known language hint for Arabic vs English/Latin recognition. API deployments expose `OCR_LANGUAGE`; automatic script routing is not validated. Paddle adapter locks its model instances, so concurrency increases queueing; no independent multi-worker speedup is claimed. EasyOCR was unnecessary after a working higher-recall candidate was found.\n"
    api_concurrency=[]
    for folder in [ROOT/"results/test10", ROOT/"results/test10_c2", ROOT/"results/test10_c4"]:
        for provider in PROVIDERS:
            s=read(folder/provider/"summary.json",{})
            if s.get("timed_run_complete"):
                q=s["profile"]
                api_concurrency.append([provider,s["concurrency"],s["count"],s["succeeded"],fmt(q["wall_seconds"]),fmt(s["receipts_per_minute"]),fmt(q["peak_ram_mib"]),fmt(q["average_cpu_percent"])])
    report += "\n## Live API-stage concurrency measurements\n\n"+table(["Provider","Concurrency","Requests","Valid receipts","Batch seconds","Actual batch rpm","Local peak MiB","Local CPU% (100=one core)"],api_concurrency)
    report += "\nText-provider API-stage results exclude cached OCR; their serial full-pipeline latency above adds its measured OCR stage. Gemini has no OCR stage. Concurrency runs use the frozen ten-receipt set and are throughput trials, not prompt tuning. All retries remain in their run files; missing usage after server errors is unknown, not assumed free.\n\n## Acceptance and operating decision\n\nSee [cost per acceptable receipt](acceptance.md) for the strict all-annotated-fields rule and its actual pass counts. The API default is NVIDIA as requested; the client can select Gemini without installing OCR. NVIDIA developer access is a temporary experimental price basis, not a known commercial price. Ollama is implemented but inference is blocked by account credits (HTTP402), so it has no held-out accuracy result. [Ollama pricing](https://ollama.com/pricing) currently lists pay-as-you-go; no usable free starter allowance was confirmed for this account.\n"
    (ROOT/"reports/benchmark.md").write_text(report)
    records=[]
    resources=[]
    for r in results:
        for x in r["rows"]:
            records.append({"provider":r["provider"],"id":x["id"],"dataset":x["dataset"],"language":x["language"],"status":x["status"],"model_output_observed":has_model_output(x),"model":x.get("metadata",{}).get("model"),"field_correct":x["metrics"]["field_correct"],"field_total":x["metrics"]["field_total"],"field_accuracy":x["metrics"]["field_accuracy"],"line_f1":(x["metrics"].get("line_items") or {}).get("f1"),"api_wall_seconds":x["wall_seconds"],"json_valid":x.get("metadata",{}).get("json_valid"),"input_tokens":x.get("metadata",{}).get("input_tokens"),"output_tokens":x.get("metadata",{}).get("output_tokens"),"cost_usd_estimated":inference_cost(x.get("metadata",{}))})
        resources.append({"provider":r["provider"],"scope":"local OCR + client stages, or remote Modal inference", "concurrency":1,"evaluated":r["evaluated"],"core_seconds_per_receipt":r.get("cpu_each"),"peak_ram_mib":r.get("peak_ram_mib"),"gpu_peak_mib":r.get("gpu_peak_mib"),"rpm":r.get("rpm"),"modal_allocated_seconds_estimated":r.get("modal_allocated_seconds_estimated"),"baseline_ram_mib":None,"average_cpu_percent":None,"peak_cpu_percent":None})
    for folder in [ROOT/"results/test10",ROOT/"results/test10_c2",ROOT/"results/test10_c4"]:
        for provider in PROVIDERS:
            s=read(folder/provider/"summary.json",{})
            if s.get("timed_run_complete"):
                q=s["profile"]
                resources.append({"provider":provider,"scope":"live client/API stage; provider inference remote, OCR excluded","concurrency":s["concurrency"],"evaluated":s["count"],"core_seconds_per_receipt":q["cpu_core_seconds"]/s["count"],"peak_ram_mib":q["peak_ram_mib"],"gpu_peak_mib":None,"rpm":s["receipts_per_minute"],"modal_allocated_seconds_estimated":None,"baseline_ram_mib":q["baseline_ram_mib"],"average_cpu_percent":q["average_cpu_percent"],"peak_cpu_percent":q["peak_cpu_percent"]})
    for engine in ["paddle","tesseract"]:
        for c in [1,2,4]:
            s=read(ROOT/f"results/ocr_development/{engine}/summary_c{c}.json",{})
            if s:
                q=s["profile"]
                resources.append({"provider":engine,"scope":"local development OCR process tree","concurrency":c,"evaluated":s["count"],"core_seconds_per_receipt":q["cpu_core_seconds"]/s["count"],"peak_ram_mib":q["peak_ram_mib"],"gpu_peak_mib":None,"rpm":s["receipts_per_minute"],"modal_allocated_seconds_estimated":None,"baseline_ram_mib":q["baseline_ram_mib"],"average_cpu_percent":q["average_cpu_percent"],"peak_cpu_percent":q["peak_cpu_percent"]})
    for name in ["modal_linux_cpu.json","modal_linux_paddle.json"]:
        for s in read(ROOT/"results"/name,[]):
            q=s["resource"]
            resources.append({"provider":name,"scope":"remote x86 Linux development OCR process tree","concurrency":q["concurrency"],"evaluated":q["receipts"],"core_seconds_per_receipt":q["cpu_core_seconds"]/q["receipts"],"peak_ram_mib":q["peak_ram_bytes"]/2**20,"gpu_peak_mib":None,"rpm":q["receipts_per_minute"],"modal_allocated_seconds_estimated":None,"baseline_ram_mib":q["baseline_ram_bytes"]/2**20,"average_cpu_percent":q["cpu_percent_mean"],"peak_cpu_percent":q["cpu_percent_peak"]})
    for result in results:
        if result["provider"] == "modal_hybrid":
            for row in result["rows"]:
                m=row.get("metadata",{});q=m.get("resource",{})
                resources.append({"provider":"modal_hybrid","scope":"remote GPU per-receipt inference; initialization separate","id":row["id"],"concurrency":1,"evaluated":1,"core_seconds_per_receipt":q.get("cpu_core_seconds"),"peak_ram_mib":q.get("peak_ram_bytes",0)/2**20,"baseline_ram_mib":q.get("baseline_ram_bytes",0)/2**20,"average_cpu_percent":q.get("cpu_percent_mean"),"peak_cpu_percent":q.get("cpu_percent_peak"),"gpu_model":q.get("gpu_name"),"gpu_utilization_percent_mean":q.get("gpu_utilization_percent_mean"),"gpu_peak_mib":(q.get("gpu_peak_allocated_bytes") or 0)/2**20,"allocated_cpu_physical_cores":q.get("allocated_cpu_physical_cores"),"allocated_ram_mib":q.get("allocated_ram_mib"),"initialization_seconds":m.get("initialization_seconds"),"cold_first_request":m.get("cold_first_request"),"container_id":m.get("container_id"),"container_uptime_seconds":q.get("container_uptime_seconds")})
    init=read(ROOT/"results/ocr_initialization.json",{})
    if init:
        q=init["profile"]
        resources.append({"provider":"paddle","scope":"SDK import and both model initializations; no image inference","concurrency":1,"evaluated":0,"baseline_ram_mib":q["baseline_ram_mib"],"peak_ram_mib":q["peak_ram_mib"],"average_cpu_percent":q["average_cpu_percent"],"peak_cpu_percent":q["peak_cpu_percent"],"initialization_seconds":q["wall_seconds"]})
    for name, values in [("benchmark.csv",records),("resource_profiles.csv",resources)]:
        if values:
            with (ROOT/"results"/name).open("w") as f:
                columns=list(dict.fromkeys(key for value in values for key in value))
                writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(values)
    write_cost_report(results)
    write_infrastructure(results)
    print("Generated benchmark, cost, infrastructure reports and CSVs")


def write_cost_report(results):
    rows=[]
    csvrows=[]
    for r in results:
        for volume in (50,100,1000,10000,100000):
            cost=r.get("unit_cost")*volume if r.get("unit_cost") is not None else None
            rows.append([r["provider"],volume,fmt(cost,digits=5)])
            csvrows.append({"provider":r["provider"],"volume":volume,"inference_cost_usd_estimated":cost,"evaluated_receipts":r["evaluated"],"basis":"measured tokens or observed allocated resource lifecycle extrapolation; excludes fixed CPU server"})
    with (ROOT/"results/cost_projections.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    text="# Cost analysis\n\nAll monetary results below are **estimated economic charges**, not account billing records. Expenditure cannot be confirmed without provider invoices/balance records. Measurements include usage from repairs/retries where returned; failures with missing usage remain unknown. Price check: 2026-11-25.\n\n"
    text+=table(["Provider","Volume","Inference USD (estimated)"],rows)
    text+="\nOllama gemma4:31b uses $0.14/M input + $0.40/M output; gpt-oss:20b $0.07/$0.30. Pay-as-you-go is distinct from Pro $20/month with $60 credits and Max $100 with $300 credits. Subscription cash and included inference credits must not be double-counted. [Official Ollama prices](https://ollama.com/pricing).\n\nGemini 3.5 Flash-Lite uses $0.30/M image/text input + $2.50/M output including thoughts; 3.1 uses $0.25/$1.50. Thinking is already included once in output_tokens. Image/prompt modality token detail is retained in attempt metadata. [Official Google prices](https://aistudio.google.com/docs/pricing).\n\nNVIDIA developer access succeeded; commercial token pricing and cash debits are unavailable. Promotional/credit access is not evidence that production inference is free. NVIDIA projections remain N/A.\n\nLocal OCR has no metered invoice on this Mac; its CPU work still has economic cost. At Modal CPU list rates the variable compute estimate is `core_seconds×0.0000131 + allocated_GiB×allocated_seconds×0.00000222`, plus any region multiplier. A continuously running server instead costs its monthly plan, allocated over actual monthly volume. For the $48/month 4vCPU/8GiB example, infrastructure adds $48, $4.80 or $0.48 per 1,000 at 1k,10k or100k receipts/month respectively, before API inference. These are different utilization scenarios, not additional simultaneous bills.\n"
    modal=next((r for r in results if r["provider"]=="modal_hybrid"),{})
    warm=[x.get("metadata",{}).get("resource",{}).get("wall_seconds") for x in modal.get("rows",[]) if not x.get("metadata",{}).get("cold_first_request")]
    init=[x.get("metadata",{}).get("initialization_seconds") for x in modal.get("rows",[]) if x.get("metadata",{}).get("cold_first_request")]
    w,i=mean(warm),mean(init)
    rates = [modal_allocation_rate(x.get("metadata", {}).get("resource", {})) for x in modal.get("rows", [])]
    rate = rates[0] if rates and None not in rates and len(set(rates)) == 1 and modal_resource_cost(modal.get("rows", [])) is not None else None
    scenarios=[]
    if w is not None and i is not None and rate is not None:
        scenarios=[["50 continuous, one container",fmt(i+50*w+2),fmt((i+50*w+2)*rate,digits=5)], ["50 sporadic, each container has gone idle",fmt(50*(i+w+2)),fmt(50*(i+w+2)*rate,digits=5)], ["50 sustained, warm serial service",fmt(50*w),fmt(50*w*rate,digits=5)]]
    text+="\n## Modal arrival scenarios\n\n"+table(["Scenario (estimated)","Allocated seconds","USD"],scenarios)
    if not scenarios:
        text += "\nModal scenario costs are N/A: complete, consistent allocation and container-lifecycle evidence is unavailable.\n"
    text+="\nThe sporadic case assumes inter-arrival gaps exceed the configured 2s scale-down period; not every request is assumed to cause a cold start in the batch or sustained cases. T4 is $0.000164/s, L4 $0.000222/s, CPU physical core $0.0000131/s, RAM GiB $0.00000222/s. US placement has a1.15× example multiplier; scenarios use recorded allocations and region multipliers, and remain unavailable without stable container IDs. Model load occurs once per container, cache persists in a volume, and min_containers=0. Model revisions/weights/init cost and observed container uptimes are retained. Actual billed lifecycle may include unobserved startup; reported estimates are lower bounds where applicable. Credits reduce cash expenditure but not resource cost. [Official Modal prices](https://modal.com/pricing).\n\nGPU L4 is 35.4% more expensive per GPU-second than T4; it needs measured latency savings to justify that change. It was not selected solely from specifications. CPU-only document VLM is not claimed practical without measurement.\n"
    api=next((r for r in results if r["provider"]=="gemini"),{})
    if w is not None and rate is not None and api.get("unit_cost") is not None:
        gpu_unit=w*rate
        difference=api["unit_cost"]-gpu_unit
        if difference>0:
            text+=f"\nAt fully warm measured latency, self-host variable cost is ${gpu_unit:.6f}/receipt vs Gemini ${api['unit_cost']:.6f}. With $48 fixed monthly infrastructure, the arithmetic break-even is approximately {48/difference:,.0f} receipts/month, before idle/cold-starts, operations and accuracy differences.\n"
        else:
            text+=f"\nAt measured warm latency the self-host variable cost is ${gpu_unit:.6f}/receipt vs Gemini ${api['unit_cost']:.6f}. **There is no positive volume break-even under these measured rates**: increasing volume does not overcome a higher variable cost. Accuracy and operational effort also matter.\n"
    text+="\nThe cumulative authorization cap and reserved unknown-price costs are in `results/spend_ledger.json`. Development runs are separate from the selected evaluation-run cost; their raw usage is retained in `results/development_*` and Modal experiment files.\n"
    (ROOT/"reports/cost_analysis.md").write_text(text)


def write_infrastructure(results):
    pricing=read(ROOT/"configs/pricing.json",{})
    paddle=read(ROOT/"results/ocr_development/paddle/summary_c1.json",{})
    p=paddle.get("profile",{})
    peak=p.get("peak_ram_mib",0)
    cpu=p.get("cpu_core_seconds",0)/paddle.get("count",1)
    linux_paddle=read(ROOT/"results/modal_linux_paddle.json",[])
    linux_baseline=next((r.get("resource",{}) for r in linux_paddle if r.get("resource",{}).get("concurrency")==1),{})
    sizing_scope="Apple M1 Pro (unverified cloud transfer)"
    if linux_baseline:
        peak=linux_baseline["peak_ram_bytes"]/2**20
        cpu=linux_baseline["cpu_core_seconds"]/linux_baseline["receipts"]
        sizing_scope="measured x86 Linux Modal, concurrency 1, one physical core/two vCPU equivalent"
    rows=[]
    for plan in pricing.get("cloud_servers",[]):
        memory_ok=plan["ram_gib"]*1024>=peak*1.25
        concurrency=1 if memory_ok else 0
        # Current adapter has a lock, so it can use at most one OCR core at once.
        # Two vCPU are the measured Modal physical-core reservation. Other
        # vendors' SMT/share levels differ; scale below that anchor conservatively.
        usable=min(.75,plan["vcpu"]*.375)
        rpm=usable*60/cpu if cpu and memory_ok else None
        util=usable/(plan["vcpu"]/2)*100 if rpm else None
        headroom=plan["ram_gib"]*1024-peak if memory_ok else None
        rows.append([plan["provider"],f'{plan["vcpu"]}/{plan["ram_gib"]}',f'{plan["currency"]} {plan["monthly"]}',"unavailable" if plan.get("currently_unavailable") else concurrency,fmt(rpm),fmt(rpm*1440*.7 if rpm else None, digits=0),fmt(util),fmt(headroom),f'{plan["currency"]} {plan["monthly"]/10:.3f}',"No"])
    text="# Infrastructure recommendation\n\nSizing below uses measured OCR core-seconds and peak process-tree memory with 25% RAM headroom and a 75% CPU target. **Cloud performance values are estimates, not direct Apple-to-x86 equivalence.** Linux Modal CPU validation is retained separately. Listed monthly prices were checked 2026-11-25, are region-specific, and exclude taxes/network extras.\n\n"
    text+=f"Sizing anchor: **{sizing_scope}**. Selected Paddle development evidence: {peak:.1f}MiB peak RAM; {cpu:.3f} core-seconds/receipt; minimum planned memory with 25% headroom {peak*1.25/1024:.2f}GiB. The Mac peaked near 5GiB, while Linux was lower, demonstrating why direct Apple-to-x86 sizing was inappropriate. The adapter serializes OCR inference under a lock; multiple waiting HTTP requests do not imply multiple model workers.\n\n"
    text+=table(["Cloud","vCPU/GiB","Monthly","Safe active OCR workers (est)","OCR rpm (est ceiling)","Daily at70% availability (est)","CPU% machine(est)","RAM headroom MiB (est)","Infra/1k at 10k/mo","GPU?"],rows)
    text+="\nAll rpm values are conservative CPU-only estimates based on the Linux core-seconds anchor where available, capped at one OCR worker and 75% utilization of its physical-core-equivalent capacity. Cloud vendor vCPU sharing varies; validate throughput on the purchased plan. External API latency/rate limits lower end-to-end throughput. Daily capacity assumes 70% availability for retries/queue gaps; it is not an SLA. Memory headroom is provisioned RAM minus measured peak.\n\n## Deployment choices\n\n- **Low volume:** Gemini-only API, no dedicated GPU and no local OCR dependencies. Start 1vCPU/2GB ($12/month DigitalOcean example) or a scale-to-zero CPU wrapper. Actual client concurrency 1/2/4 measurements are retained; remote inference CPU/RAM are managed by Google. The local client CPU profile includes SDK startup and API waiting.\n- **Medium volume requiring local OCR:** **2vCPU/4GB, one active OCR worker**, bounded upstream requests, using the measured Linux1-worker memory anchor. DigitalOcean Basic $24/month or Hetzner CPX22 €16.99/month are examples. At1vCPU/2GB the 25% RAM margin is insufficient for this Paddle corpus. Concurrent preprocessing raises RAM even though the model lock serializes inference; cap active OCR to 1 on 4GB. 4vCPU/8GB ($48/month DO / €30.49 Hetzner) gives room for concurrent preprocessing and API tasks, but the present one-worker lock gains no OCR throughput from extra CPUs.\n- **High volume:** load-test API quotas and shard OCR only after validating per-worker memory.8vCPU/16GB is not justified by this one-worker POC. Multiple process workers are a further experiment, not a claimed result. For owning inference, use the measured scale-zero T4 configuration only if its accuracy and warm unit cost justify it; L4 and CPU-only VLM need separate evidence before recommendation.\n\nLightsail's actual bundles differ from requested CPU/RAM ratios; for example 8GB is 2vCPU, and 16GB is 4vCPU. Hetzner CX plans shown as unavailable are not purchase recommendations. Currency conversion is not guessed; EUR and USD remain separate.\n\n## Linux validation and limits\n\n"
    linux=read(ROOT/"results/modal_linux_cpu.json",[])
    linuxrows=[]
    for run in linux:
        q=run.get("resource",{})
        linuxrows.append([q.get("platform"),q.get("concurrency"),q.get("allocated_vcpu_equivalent"),q.get("allocated_ram_mib"),fmt(q.get("receipts_per_minute")),fmt(q.get("peak_ram_bytes",0)/2**20),fmt(q.get("cpu_core_seconds",0)/q.get("receipts",1))])
    text+=table(["Platform (Tesseract)","Concurrency","Provisioned vCPU","Provisioned MiB","rpm measured","Peak MiB measured","Core-sec/receipt"],linuxrows)
    paddlerows=[]
    for run in linux_paddle:
        q=run.get("resource",{})
        paddlerows.append([q.get("concurrency"),q.get("allocated_vcpu_equivalent"),q.get("allocated_ram_mib"),fmt(q.get("receipts_per_minute")),fmt(q.get("peak_ram_bytes",0)/2**20),fmt(q.get("cpu_core_seconds",0)/q.get("receipts",1))])
    text+="\nSelected Paddle on Linux:\n\n"+table(["Concurrency","Provisioned vCPU","Provisioned MiB","rpm measured","Peak MiB measured","Core-sec/receipt"],paddlerows)
    text+="\nBoth Linux probes used the same eight development images. Their provisioned RAM was not their observed consumption; this supports a smaller deployment recommendation but does not replace testing a 4GiB allocation directly. The remote model evidence is in [Modal experiments](modal_experiments.md). Modal physical CPU core represents two vCPUs; local percent100 means one fully occupied core.\n\nSee [cost analysis](cost_analysis.md) for continuous/sporadic/sustained GPU scenarios and break-even. See [official sources](sources.md) for plan scope and prices.\n"
    text+='\nFor the1vCPU/2GiB Gemini proxy, a conservative planning estimate is concurrency2,25receipts/minute and25,200receipts/day at70%availability, subject to APIquota. This is anchored to the measured client trial, not an x86 server benchmark. Estimated CPU utilization is20–50%of one vCPU, with over1.5GiB memory headroom against the measured216MiB client peak plus web overhead. At10,000rawreceipts/month, the$12proxy adds$1.20/1,000 before paid inference; no dedicatedGPU is needed.\n\nThe2vCPU/4GiB OCR recommendation uses `MAX_IN_FLIGHT=1` for memory safety. Including measured NVIDIA APIwaiting, its planning throughput is approximately3receipts/minute (about3,000/day at70%availability), below the OCR-only ceiling in the table; CPU utilization should remain below saturation. These are estimates requiring target-host validation. Use8GiB when allowing multiple queued/preprocessed large images.\n\nThe direct VLM experiment peaked near7.6GiB processRSS. A production trial needs roughly12GiB RAM to provide25%headroom, plusT4(16GiBVRAM; measuredpeakabout4.5GiB) and the experimentally allocated2physicalCPUcores. This is a conditional machine specification, not a recommendation to self-host: measured quality-adjusted cost favored the external API. GPUconcurrency2/4, CPU-only VLM andL4performance were not measured; the present service serializes one model instance, so no GPUconcurrency speedup is claimed.\n'
    (ROOT/"reports/infrastructure_recommendation.md").write_text(text)


if __name__ == "__main__":
    main()
