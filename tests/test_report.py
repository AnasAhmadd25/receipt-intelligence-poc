import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.metrics import evaluate
from benchmark.report import modal_resource_cost, selected_records, summarize
from benchmark import report


class ReportTests(unittest.TestCase):
    def test_run_config_controls_record_selection_and_order(self):
        manifest = {"test": [{"id": "a"}, {"id": "b"}], "development": [{"id": "c"}], "initial_test_ids": ["a"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(json.dumps({"record_ids": ["b", "a"], "split": "test", "limit": 50}))
            records, split = selected_records(root, manifest)
            self.assertEqual([r["id"] for r in records], ["b", "a"])
            self.assertEqual(split, "test")

    def test_legacy_fallback_only_uses_initial_test(self):
        with tempfile.TemporaryDirectory() as directory:
            records, split = selected_records(Path(directory), {"test": [{"id": "a"}, {"id": "b"}], "initial_test_ids": ["b"]})
            self.assertEqual([r["id"] for r in records], ["b"])

    def test_unavailable_and_invalid_model_output_have_different_denominators(self):
        truth = {"receipt": {"total": "10"}, "annotated_fields": ["total"], "line_items_supported": False}
        row = {"id": "a", "dataset": "AraReceipt", "status": "error", "wall_seconds": 1, "metadata": {"attempts": [{"status": "error", "http_status": 402, "raw_response": ""}]}, "metrics": evaluate({}, truth)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "gemini"
            folder.mkdir()
            path = folder / "a.json"
            path.write_text(json.dumps(row))
            result = summarize(root, "gemini", [{"id": "a"}])
            self.assertEqual(result["attempted"], 1)
            self.assertEqual(result["evaluated"], 0)
            self.assertIsNone(result["field_accuracy"])
            self.assertIsNone(result["json_validity"])
            self.assertEqual(result["service_field_accuracy"], 0)
            row["metadata"]["attempts"][0]["raw_response"] = "not JSON"
            path.write_text(json.dumps(row))
            result = summarize(root, "gemini", [{"id": "a"}])
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["field_accuracy"], 0)
            self.assertEqual(result["json_validity"], 0)

    def test_partial_provider_is_not_ranked_and_report_uses_selected_run(self):
        truth = {"receipt": {"total": "10"}, "annotated_fields": ["total"], "line_items_supported": False}
        row = {"id": "a", "dataset": "AraReceipt", "language": "Arabic", "status": "ok", "wall_seconds": 1, "metadata": {"json_valid": True, "resource": {"gpu_peak_allocated_bytes": None}}, "metrics": evaluate({"total": "10"}, truth)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "reports").mkdir()
            run = root / "results/custom"
            (run / "gemini").mkdir(parents=True)
            (root / "data/manifest.json").write_text(json.dumps({"test": [{"id": "a", "dataset": "AraReceipt"}, {"id": "b", "dataset": "CORD-v2"}], "initial_test_ids": ["a"]}))
            (run / "run_config.json").write_text(json.dumps({"record_ids": ["a", "b"], "split": "test", "limit": 2}))
            (run / "gemini/a.json").write_text(json.dumps(row))
            summary = summarize(run, "gemini", [{"id": "a"}, {"id": "b"}])
            self.assertFalse(summary["complete_expected"])
            self.assertIsNone(summary["gpu_peak_mib"])
            with patch.object(report, "ROOT", root), patch("sys.argv", ["report", "--run-name", "custom"]):
                report.main()
            text = (root / "reports/benchmark.md").read_text()
            self.assertIn("2 test receipts (1 Arabic, 1 CORD)", text)
            self.assertNotIn("highest measured", text)

    def test_modal_cost_uses_recorded_allocations_and_unique_container(self):
        resource = {"allocated_gpu": "L4", "allocated_cpu_physical_cores": 4, "allocated_ram_mib": 16384, "region_selection": "us", "region_price_multiplier": 1.15, "container_uptime_seconds": 10}
        first = {"metadata": {"container_id": "container-a", "resource": resource}}
        second = {"metadata": {"container_id": "container-a", "resource": {**resource, "container_uptime_seconds": 20}}}
        expected_rate = (.000222 + 4 * .0000131 + 16 * .00000222) * 1.15
        self.assertAlmostEqual(modal_resource_cost([first, second]), 22 * expected_rate)
        second["metadata"].pop("container_id")
        self.assertIsNone(modal_resource_cost([first, second]))

    def test_modal_missing_allocation_and_conflicting_container_data_are_unknown(self):
        resource = {"allocated_gpu": "T4", "allocated_cpu_physical_cores": 2, "allocated_ram_mib": 8192, "region_selection": "us", "region_price_multiplier": 1.15, "container_uptime_seconds": 10}
        first = {"metadata": {"container_id": "a", "resource": resource}}
        self.assertIsNotNone(modal_resource_cost([first]))
        bad = {"metadata": {"container_id": "a", "resource": {**resource, "allocated_ram_mib": 16384}}}
        self.assertIsNone(modal_resource_cost([first, bad]))
        first["metadata"]["resource"] = {"container_uptime_seconds": 10}
        self.assertIsNone(modal_resource_cost([first]))


if __name__ == "__main__":
    unittest.main()
