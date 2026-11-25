import unittest

from benchmark.metrics import evaluate, normalize_number


class MetricsTests(unittest.TestCase):
    def test_only_annotated_fields_count(self):
        truth = {"receipt": {"total": "12.50"}, "annotated_fields": ["total"], "line_items_supported": False}
        result = evaluate({"total": "١٢٫٥٠", "merchant_name": "Unannotated shop"}, truth)
        self.assertEqual(result["field_correct"], 1)
        self.assertEqual(result["field_total"], 1)
        self.assertEqual(result["hallucinated_fields"], [])
        self.assertIn("merchant_name", result["unscorable_fields"])

    def test_known_absence_and_unknown_are_different(self):
        truth = {"receipt": {"total": "10", "tax": None}, "annotated_fields": ["total", "tax"], "line_items_supported": False}
        result = evaluate({"total": None, "tax": "1", "currency": "EGP"}, truth)
        self.assertEqual(result["missing_fields"], ["total"])
        self.assertEqual(result["hallucinated_fields"], ["tax"])
        self.assertEqual(result["field_total"], 2)

    def test_one_to_one_items_prevents_duplicate_credit(self):
        truth = {"receipt": {"line_items": [{"raw_description": "Milk", "quantity": "1", "line_total": "10"}]}, "annotated_fields": [], "line_items_supported": True, "item_annotated_fields": [["raw_description", "quantity", "line_total"]]}
        item = {"raw_description": "Milk", "quantity": "1", "line_total": "10"}
        result = evaluate({"line_items": [item, item]}, truth)
        self.assertEqual(result["line_items"]["matched"], 1)
        self.assertEqual(result["line_items"]["precision"], .5)
        self.assertEqual(result["line_items"]["recall"], 1)

    def test_wrong_money_does_not_match_item(self):
        truth = {"receipt": {"line_items": [{"raw_description": "Milk", "line_total": "10"}]}, "annotated_fields": [], "line_items_supported": True, "item_annotated_fields": [["raw_description", "line_total"]]}
        result = evaluate({"line_items": [{"raw_description": "Milk", "line_total": "11"}]}, truth)
        self.assertEqual(result["line_items"]["matched"], 0)
        self.assertEqual(result["description_similarity"], 1)

    def test_unsupported_grouping_is_unscorable(self):
        result = evaluate({"line_items": [{"raw_description": "Milk"}]}, {"receipt": {}, "annotated_fields": [], "line_items_supported": False})
        self.assertIsNone(result["line_items"])

    def test_decimal_has_no_float_rounding_and_arabic_normalization(self):
        self.assertEqual(normalize_number("١٬٢٣٤٫٥٠"), "1234.5")
        self.assertEqual(normalize_number("9999999999999999.99"), "9999999999999999.99")

    def test_identical_descriptions_use_maximum_matching(self):
        items = [{"raw_description": "Tea", "line_total": "10"}, {"raw_description": "Tea", "line_total": "20"}]
        truth = {"receipt": {"line_items": items}, "annotated_fields": [], "line_items_supported": True, "item_annotated_fields": [["raw_description", "line_total"], ["raw_description", "line_total"]]}
        self.assertEqual(evaluate({"line_items": list(reversed(items))}, truth)["line_items"]["matched"], 2)

    def test_item_quantity_metric_independent_of_numeric_correctness(self):
        truth = {"receipt": {"line_items": [{"raw_description": "Milk", "quantity": "2", "unit_price": "10", "line_total": "20"}]}, "annotated_fields": [], "line_items_supported": True, "item_annotated_fields": [["raw_description", "quantity", "unit_price", "line_total"]]}
        result = evaluate({"line_items": [{"raw_description": "Milk", "quantity": "1", "unit_price": "10", "line_total": "10"}]}, truth)
        self.assertEqual(result["item_fields"]["quantity"]["accuracy"], 0)
        self.assertEqual(result["item_fields"]["unit_price"]["accuracy"], 1)
        self.assertEqual(result["line_items"]["matched"], 0)

    def test_description_similarity_is_one_to_one(self):
        truth = {"receipt": {"line_items": [{"raw_description": "Milk"}, {"raw_description": "Milk"}]}, "annotated_fields": [], "line_items_supported": True, "item_annotated_fields": [["raw_description"], ["raw_description"]]}
        result = evaluate({"line_items": [{"raw_description": "Milk"}]}, truth)
        self.assertEqual(result["description_similarity"], .5)


if __name__ == "__main__":
    unittest.main()
