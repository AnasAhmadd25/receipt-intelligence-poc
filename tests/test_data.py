import hashlib
import json
from pathlib import Path
import unittest

from benchmark.prepare_data import convert_ara, convert_cord, cord_money


class DataConversionTests(unittest.TestCase):
    def test_ara_multiple_totals_not_arbitrarily_selected(self):
        truth = convert_ara({"annotations": [{"text": "10", "label": 23}, {"text": "20", "label": 23}, {"text": "Store", "label": 1}]})
        self.assertNotIn("total", truth["annotated_fields"])
        self.assertIn("merchant_name", truth["annotated_fields"])
        self.assertFalse(truth["line_items_supported"])

    def test_ara_tax_percent_is_not_monetary_tax(self):
        truth = convert_ara({"annotations": [{"text": "14%", "label": 19}]})
        self.assertNotIn("tax", truth["annotated_fields"])

    def test_cord_honors_per_item_coverage_and_subitems(self):
        truth = convert_cord({"gt_parse": {"menu": [{"nm": "Tea", "cnt": "2 x", "price": "12.000", "sub": {"nm": "Jelly", "price": "4,000"}}], "total": {"total_price": "16.000"}}, "valid_line": []})
        self.assertEqual(truth["receipt"]["total"], "16000")
        self.assertEqual(len(truth["receipt"]["line_items"]), 2)
        self.assertEqual(truth["receipt"]["line_items"][0]["quantity"], "2")
        self.assertNotIn("quantity", truth["item_annotated_fields"][1])
        self.assertNotIn("merchant_name", truth["annotated_fields"])
        self.assertNotIn("currency", truth["annotated_fields"])

    def test_cord_money_locale(self):
        self.assertEqual(cord_money("Rp 1.234.000"), "1234000")
        self.assertEqual(cord_money("@11,000"), "11000")
        self.assertEqual(cord_money("12,500.50"), "12500.5")


class DownloadedDataTests(unittest.TestCase):
    def test_manifest_counts_and_disjoint_hashes(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data/manifest.json"
        if not path.exists():
            self.skipTest("Run python -m benchmark.prepare_data first")
        manifest = json.loads(path.read_text())
        self.assertEqual(len(manifest["test"]), 50)
        self.assertEqual(len(manifest["development"]), 8)
        all_records = manifest["test"] + manifest["development"]
        self.assertEqual(len({r["id"] for r in all_records}), 58)
        self.assertEqual(len({r["sha256"] for r in all_records}), 58)
        self.assertEqual(sum(r["dataset"] == "AraReceipt" for r in manifest["test"]), 30)
        self.assertEqual(sum(r["dataset"] == "CORD-v2" for r in manifest["test"]), 20)
        self.assertEqual(sum(r["dataset"] == "AraReceipt" for r in manifest["development"]), 5)
        self.assertEqual(sum(r["dataset"] == "CORD-v2" for r in manifest["development"]), 3)
        ids = manifest["initial_test_ids"]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertTrue(set(ids) <= {r["id"] for r in manifest["test"]})
        self.assertEqual(sum(r["id"] in ids and r["dataset"] == "AraReceipt" for r in manifest["test"]), 6)
        self.assertEqual(sum(r["id"] in ids and r["dataset"] == "CORD-v2" for r in manifest["test"]), 4)
        for record in all_records:
            self.assertEqual(hashlib.sha256((root / record["image"]).read_bytes()).hexdigest(), record["sha256"])
            truth = json.loads((root / record["annotation"]).read_text())
            self.assertFalse(truth["ocr_text_complete"])
            for field in truth["annotated_fields"]:
                self.assertIn(field, truth["receipt"])


if __name__ == "__main__":
    unittest.main()
