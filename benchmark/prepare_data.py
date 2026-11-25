"""Download a small, reproducible public sample without whole parquet shards.

Run: python -m benchmark.prepare_data. Only dataset-server JSON rows and chosen
image URLs are downloaded. Standardized truth is derived from human annotations;
no extraction model is used to create benchmark truth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from benchmark.metrics import normalize_date, normalize_number, normalize_time

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261125
SOURCES = {
    "AraReceipt": {"repo": "IslamMesabah/AraReceipt", "revision": "d995859bd253ad37053faee8bce24b5fa1c265e2", "license": "MIT", "language": "Arabic / mixed Arabic-English", "card": "https://huggingface.co/datasets/IslamMesabah/AraReceipt", "source_split": "train"},
    "CORD-v2": {"repo": "naver-clova-ix/cord-v2", "revision": "7f0115a4b758a71d6473b8d085751692da2fef98", "license": "CC-BY-4.0", "language": "Indonesian / Latin script / mixed English", "card": "https://huggingface.co/datasets/naver-clova-ix/cord-v2", "source_split": "test (evaluation), validation (development)"},
}
FIELDS = ("merchant_name", "receipt_number", "purchase_date", "purchase_time", "currency", "subtotal", "discount", "tax", "total")


def empty_receipt():
    return {**dict.fromkeys(FIELDS), "line_items": [], "raw_text": "", "missing_fields": [], "uncertain_fields": [], "validation_warnings": []}


def _unique(values):
    return list(dict.fromkeys(v for v in values if v is not None and v != ""))


def convert_ara(row):
    receipt = empty_receipt()
    annotated, omitted = [], {}
    annotations = row.get("annotations", [])
    mapping = {"merchant_name": 1, "purchase_date": 7, "purchase_time": 9, "subtotal": 17, "tax": 19, "total": 23}
    for field, label in mapping.items():
        values = [a["text"].strip() for a in annotations if a["label"] == label and a["text"].strip()]
        if not values:
            continue
        normalized = values
        if field in {"subtotal", "tax", "total"}:
            normalized = [normalize_number(v) if "%" not in v and "٪" not in v else None for v in values]
        elif field == "purchase_date":
            normalized = [normalize_date(v) for v in values]
            normalized = [v if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) else None for v in normalized]
        elif field == "purchase_time":
            normalized = [normalize_time(v) for v in values]
            normalized = [v if re.fullmatch(r"\d{2}:\d{2}:\d{2}", v) else None for v in normalized]
        candidates = _unique(normalized)
        # Multiple distinct values are not disambiguated with geometric guesses.
        if len(candidates) == 1 and None not in normalized:
            receipt[field] = candidates[0]
            annotated.append(field)
        else:
            omitted[field] = {"reason": "multiple distinct values or unsupported numeric/date/time format", "values": values}
    ocr_text = "\n".join(a["text"] for a in annotations if a["label"] != 0)
    receipt["raw_text"] = ocr_text
    return {"receipt": receipt, "annotated_fields": annotated, "line_items_supported": False, "item_annotated_fields": [], "ocr_text": ocr_text, "ocr_text_complete": False, "source_annotations": {"class_regions": annotations, "omitted_scalar_fields": omitted}, "annotation_notes": ["Region classes do not supply item grouping; product boxes are retained only as source annotations.", "Unannotated fields are unknown, never asserted absent."]}


def cord_money(value):
    """CORD uses Indonesian thousands separators (comma or dot), not decimals.

    A terminal group of 3 digits is treated as thousands. Mixed separators use
    the last separator as decimal only when it has 1-2 terminal digits.
    """
    if value is None:
        return None
    text = str(value).strip().replace("Rp", "").replace("RP", "").replace("@", "").replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(?:[.,]\d{3})+", text):
        text = text.replace(",", "").replace(".", "")
    elif "," in text and "." in text:
        last = max(text.rfind(","), text.rfind("."))
        if len(text) - last - 1 in (1, 2):
            text = text[:last].replace(",", "").replace(".", "") + "." + text[last + 1:]
    elif "," in text and re.fullmatch(r"[-+]?\d+,\d{1,2}", text):
        text = text.replace(",", ".")
    return normalize_number(text)


def _as_list(value):
    return value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])


def convert_cord(ground_truth):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    parse = gt.get("gt_parse", {})
    receipt = empty_receipt()
    annotated, coverage = [], []
    scalar_paths = {"subtotal": ("sub_total", "subtotal_price"), "discount": ("sub_total", "discount_price"), "tax": ("sub_total", "tax_price"), "total": ("total", "total_price")}
    for field, (section, key) in scalar_paths.items():
        section_value = parse.get(section, {})
        if isinstance(section_value, dict) and key in section_value:
            number = cord_money(section_value[key])
            if number is not None:
                receipt[field] = number
                annotated.append(field)

    def add_items(items):
        for source in _as_list(items):
            item = dict.fromkeys(("raw_description", "quantity", "unit_price", "line_total"))
            supported = []
            if isinstance(source.get("nm"), str) and source["nm"].strip():
                item["raw_description"] = source["nm"].strip()
                supported.append("raw_description")
            if "cnt" in source:
                quantity = re.fullmatch(r"\s*[xX]?\s*([-+]?\d+(?:[.,]\d+)?)\s*[xX]?\s*", str(source["cnt"]))
                if quantity:
                    item["quantity"] = normalize_number(quantity[1].replace(",", "."))
                    supported.append("quantity")
            for field, key in (("unit_price", "unitprice"), ("line_total", "price")):
                if key in source:
                    number = cord_money(source[key])
                    if number is not None:
                        item[field] = number
                        supported.append(field)
            if supported:
                receipt["line_items"].append(item)
                coverage.append(supported)
            # Nested modifier entries remain distinct annotated printed lines;
            # no missing quantity is invented and no unit price is derived.
            add_items(source.get("sub"))

    add_items(parse.get("menu"))
    ocr_text = "\n".join(" ".join(w.get("text", "") for w in line.get("words", [])) for line in gt.get("valid_line", []))
    receipt["raw_text"] = ocr_text
    return {"receipt": receipt, "annotated_fields": annotated, "line_items_supported": bool(receipt["line_items"]), "item_annotated_fields": coverage, "ocr_text": ocr_text, "ocr_text_complete": False, "annotation_notes": ["CORD header fields removed upstream are unknown, including merchant, date, time, receipt number, currency.", "OCR transcription contains valid_line annotations only, not all image text.", "Nested menu.sub annotations are flattened as modifier lines with original grouping retained in original annotation."]}


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "ReceiptIntelligenceResearch/1.0"}), timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(dataset, split, offset=0, length=100):
    query = urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": length})
    payload = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
    if any(r.get("truncated_cells") for r in payload["rows"]):
        raise RuntimeError("Dataset server truncated annotations; refusing incomplete truth")
    return payload["rows"]


def prepare(root=ROOT):
    root = Path(root)
    rng = random.Random(SEED)
    for name, source in SOURCES.items():
        metadata = json.loads(fetch("https://huggingface.co/api/datasets/" + source["repo"]))
        if metadata.get("sha") != source["revision"]:
            raise RuntimeError(f"{name} revision changed: review annotation conversion before resampling")
        base = "https://huggingface.co/datasets/" + source["repo"] + "/raw/" + source["revision"] + "/"
        target = root / "data/sources" / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_bytes(fetch(base + "README.md"))
        write_json(target / "hub_metadata.json", metadata)
        if name == "AraReceipt":
            (target / "class_list.txt").write_bytes(fetch(base + "class_list.txt"))
    ara = rows(SOURCES["AraReceipt"]["repo"], "train")
    rng.shuffle(ara)
    cord_test = list(range(100))
    cord_dev = list(range(100))
    rng.shuffle(cord_test)
    rng.shuffle(cord_dev)
    jobs = [("AraReceipt", "train", r, "test" if i < 30 else "development") for i, r in enumerate(ara[:35])]
    jobs += [("CORD-v2", "test", r, "test") for r in cord_test[:20]]
    jobs += [("CORD-v2", "validation", r, "development") for r in cord_dev[:3]]

    def download(job):
        name, split, entry, partition = job
        if isinstance(entry, int):
            entry = rows(SOURCES[name]["repo"], split, offset=entry, length=1)[0]
        row, index = entry["row"], entry["row_idx"]
        identity = f"ara_{row['id']:04d}" if name == "AraReceipt" else f"cord_{split}_{index:04d}"
        source = SOURCES[name]
        image_path = Path("data/images") / f"{identity}.jpg"
        annotation_path = Path("data/annotations") / f"{identity}.json"
        original_path = Path("data/original_annotations") / f"{identity}.json"
        dest = root / image_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Existing image cache makes rerunning the same pinned manifest cheap.
        if not dest.exists():
            url = row["image"]["src"]
            if name == "AraReceipt":
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/images/{row['id']:04d}.jpg"
            dest.write_bytes(fetch(url))
        content = dest.read_bytes()
        if not content.startswith(b"\xff\xd8") and not content.startswith(b"\x89PNG"):
            raise RuntimeError(f"Invalid image bytes: {identity}")
        original = {key: value for key, value in row.items() if key != "image"}
        gt = convert_ara(row) if name == "AraReceipt" else convert_cord(row["ground_truth"])
        write_json(root / annotation_path, gt)
        write_json(root / original_path, original)
        return partition, {"id": identity, "dataset": name, "language": source["language"], "source_split": split, "source_row_index": index, "image": str(image_path), "annotation": str(annotation_path), "original_annotation": str(original_path), "sha256": hashlib.sha256(content).hexdigest()}

    # Publish development samples first so bounded OCR/model selection can start.
    dev_jobs = [job for job in jobs if job[3] == "development"]
    test_jobs = [job for job in jobs if job[3] == "test"]
    with ThreadPoolExecutor(max_workers=5) as executor:
        downloaded = list(executor.map(download, dev_jobs))
        write_json(root / "data/dev_manifest.json", {"seed": SEED, "development": [r for _, r in downloaded]})
        print("Development images ready: data/dev_manifest.json", flush=True)
        downloaded += list(executor.map(download, test_jobs))
    hashes = [r["sha256"] for _, r in downloaded]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate image hashes: review overlap before using this sample")
    test = [r for partition, r in downloaded if partition == "test"]
    development = [r for partition, r in downloaded if partition == "development"]
    initial = [r["id"] for r in test if r["dataset"] == "AraReceipt"][:6] + [r["id"] for r in test if r["dataset"] == "CORD-v2"][:4]
    manifest = {"seed": SEED, "development": development, "test": test, "initial_test_ids": initial, "datasets": SOURCES, "selection": "Python random.Random(seed); shuffle full 100-row source splits in Ara train, CORD test, CORD validation order; first 30+5,20,3 respectively; fixed first 6+4 test IDs.", "overlap_check": {"unique_sha256": len(set(hashes)), "total": len(hashes), "receipt_sense_included_separately": False}}
    write_json(root / "data/manifest.json", manifest)
    print(json.dumps({"test": len(test), "development": len(development), "unique_hashes": len(set(hashes)), "initial_test_ids": initial}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    prepare(parser.parse_args().root)
