"""Annotation-aware receipt metrics. Unknown fields never imply known absence.

Item F1 requires a one-to-one match: description similarity >= 0.8 and exact
normalized values for every other annotated item field. Prices use Decimal.
"""
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import re
import unicodedata

SCALAR_FIELDS = ("merchant_name", "receipt_number", "purchase_date", "purchase_time", "currency", "subtotal", "discount", "tax", "total")
NUMERIC_FIELDS = {"subtotal", "discount", "tax", "total", "quantity", "unit_price", "line_total"}
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_DIGITS).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn" and c != "ـ")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def normalize_number(value):
    if value is None or value == "":
        return None
    text = unicodedata.normalize("NFKC", str(value)).translate(_DIGITS).strip().replace("٬", "").replace("٫", ".")
    # Predictions are expected in canonical decimal format; accept comma thousands.
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    try:
        number = Decimal(text)
        if not number.is_finite():
            return None
        return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")
    except InvalidOperation:
        return None


def normalize_date(value):
    text = str(value or "").translate(_DIGITS).strip()
    # Explicit day-first policy for AraReceipt, recorded in the dataset audit.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return text


def normalize_time(value):
    text = str(value or "").translate(_DIGITS).strip().lower()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p"):
        try:
            result = datetime.strptime(text, fmt)
            return result.strftime("%H:%M:%S")
        except ValueError:
            pass
    return text


def _present(value):
    return value is not None and value != "" and value != []


def _equal(field, actual, expected):
    if not _present(expected):
        return not _present(actual)
    if not _present(actual):
        return False
    if field in NUMERIC_FIELDS:
        left, right = normalize_number(actual), normalize_number(expected)
        return left is not None and right is not None and Decimal(left) == Decimal(right)
    if field == "purchase_date":
        return normalize_date(actual) == normalize_date(expected)
    if field == "purchase_time":
        return normalize_time(actual) == normalize_time(expected)
    return normalize_text(actual) == normalize_text(expected)


def _similarity(left, right):
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def _coverage(truth, index, item):
    coverage = truth.get("item_annotated_fields", [])
    if coverage and isinstance(coverage[0], list):
        return coverage[index] if index < len(coverage) else []
    return coverage or [key for key, value in item.items() if _present(value)]


def evaluate(prediction, truth):
    """Return JSON-safe counts/rates, preserving unscorable annotation gaps."""
    receipt = truth.get("receipt", {})
    prediction = prediction.get("receipt", prediction)
    annotated = set(truth.get("annotated_fields", [])) & set(SCALAR_FIELDS)
    fields, missing, hallucinated = {}, [], []
    present_count = absent_count = 0
    for field in SCALAR_FIELDS:
        if field not in annotated:
            continue
        expected, actual = receipt.get(field), prediction.get(field)
        correct = int(_equal(field, actual, expected))
        fields[field] = {"correct": correct, "total": 1, "accuracy": float(correct)}
        if _present(expected):
            present_count += 1
            if not _present(actual):
                missing.append(field)
        else:
            absent_count += 1
            if _present(actual):
                hallucinated.append(field)
    correct, total = sum(v["correct"] for v in fields.values()), len(fields)
    result = {
        "field_correct": correct, "field_total": total,
        "field_accuracy": correct / total if total else None, "fields": fields,
        "missing_fields": missing, "hallucinated_fields": hallucinated,
        "missing_rate": len(missing) / present_count if present_count else None,
        "hallucination_rate": len(hallucinated) / absent_count if absent_count else None,
        "missing_denominator": present_count, "hallucination_denominator": absent_count,
        "unscorable_fields": [field for field in SCALAR_FIELDS if field not in annotated and _present(prediction.get(field))],
        "description_similarity": None, "line_items": None, "item_fields": {},
    }
    if not truth.get("line_items_supported", False):
        return result
    expected_items = receipt.get("line_items", []) or []
    actual_items = prediction.get("line_items", []) or []
    edges = []
    for index, item in enumerate(expected_items):
        coverage = _coverage(truth, index, item)
        candidates = []
        for j, actual in enumerate(actual_items):
            if not coverage:
                continue
            if "raw_description" in coverage and _similarity(actual.get("raw_description"), item.get("raw_description")) < .8:
                continue
            if all(_equal(key, actual.get(key), item.get(key)) for key in coverage if key != "raw_description"):
                candidates.append(j)
        edges.append(candidates)
    # Augmenting paths yield maximum-cardinality one-to-one matching, including
    # duplicate product descriptions. No greedy double credit is possible.
    matched_actual = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched_actual or augment(matched_actual[j], seen):
                matched_actual[j] = i
                return True
        return False

    for i in range(len(expected_items)):
        augment(i, set())
    matched = len(matched_actual)
    precision = matched / len(actual_items) if actual_items else (1.0 if not expected_items else 0.0)
    recall = matched / len(expected_items) if expected_items else 1.0
    result["line_items"] = {"matched": matched, "predicted": len(actual_items), "expected": len(expected_items), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    # Pair independently by description for quantity/price diagnostic accuracy.
    # Numeric correctness never determines these pairings, avoiding inflated
    # monetary metrics caused by matching only already-correct lines.
    diagnostic_pairs = {}
    if expected_items and actual_items:
        from scipy.optimize import linear_sum_assignment
        weights = [[_similarity(p.get("raw_description"), item.get("raw_description"))
                    if "raw_description" in _coverage(truth, i, item) else 0.0
                    for p in actual_items] for i, item in enumerate(expected_items)]
        left, right = linear_sum_assignment(weights, maximize=True)
        diagnostic_pairs = {int(i): int(j) for i, j in zip(left, right) if weights[i][j] >= .5}
    description_scores = []
    for i, item in enumerate(expected_items):
        coverage = _coverage(truth, i, item)
        actual = actual_items[diagnostic_pairs[i]] if i in diagnostic_pairs else {}
        if "raw_description" in coverage:
            description_scores.append(_similarity(actual.get("raw_description"), item.get("raw_description")) if actual else 0.0)
        for field in coverage:
            metric = result["item_fields"].setdefault(field, {"correct": 0, "total": 0})
            metric["total"] += 1
            metric["correct"] += int(_equal(field, actual.get(field), item.get(field)))
    for metric in result["item_fields"].values():
        metric["accuracy"] = metric["correct"] / metric["total"]
    result["description_similarity"] = sum(description_scores) / len(description_scores) if description_scores else None
    return result
