import pytest

from benchmark.acceptance import acceptance, summarize_acceptance


def row(*, correct=1, total=1, status="ok", line_f1=None, total_correct=1):
    return {"status": status, "receipt": {"total": "10"}, "metadata": {"provider": "gemini", "model": "gemini-3.5-flash-lite", "input_tokens": 1000, "output_tokens": 100, "usage_complete": True},
            "metrics": {"field_correct": correct, "field_total": total, "fields": {"total": {"correct": total_correct, "total": 1}} if total else {}, "item_fields": {"raw_description": {"correct": 1, "total": 1}} if line_f1 is not None else {}, "line_items": {"f1": line_f1} if line_f1 is not None else None}}


def test_strict_requires_all_annotated_scalars_and_grouped_lines():
    assert acceptance(row())["strict_pass"] is True
    assert acceptance(row(correct=1, total=2))["strict_pass"] is False
    decision = acceptance(row(line_f1=.8))
    assert decision["strict_pass"] is False
    assert decision["critical_total_pass"] is True
    assert acceptance(row(line_f1=1))["strict_pass"] is True


def test_unannotated_is_unscorable_not_vacuous_success():
    decision = acceptance(row(correct=0, total=0))
    assert decision["scorable"] is False
    assert decision["strict_pass"] is None
    assert decision["critical_total_pass"] is None
    assert "unannotated" in decision["reason"]


def test_line_only_annotation_is_scorable_but_total_rule_na():
    decision = acceptance(row(correct=0, total=0, line_f1=1))
    assert decision["strict_pass"] is True
    assert decision["critical_total_pass"] is None


def test_invalid_generated_json_fails_and_unavailable_provider_is_na():
    invalid = row(status="error")
    invalid["receipt"] = None
    invalid["metadata"]["attempts"] = [{"raw_response": "invalid", "status": "invalid_json_or_schema"}]
    assert acceptance(invalid)["strict_pass"] is False
    assert acceptance(invalid)["critical_total_pass"] is False
    unavailable = row(status="error")
    unavailable["receipt"] = None
    unavailable["metadata"] = {"attempts": [{"status": "error", "error": "Provider returned HTTP 402"}]}
    assert acceptance(unavailable)["strict_pass"] is None
    assert acceptance(unavailable)["scorable"] is False


def test_cost_per_accepted_includes_rejected_rows_and_failed_attempt_usage():
    accepted = row()
    rejected = row(correct=0, total_correct=0)
    failed = row(status="error")
    failed["receipt"] = None
    failed["metadata"]["attempts"] = [{"raw_response": "bad JSON"}]
    result = summarize_acceptance([accepted, rejected, failed])
    # Each row costs 1000*.30/M + 100*2.50/M = $.00055.
    assert result["economic_cost_usd"] == pytest.approx(.00165)
    assert result["cost_per_strict_accepted_usd"] == pytest.approx(.00165)
    assert result["strict_passes"] == 1


def test_unknown_prices_and_zero_passes_stay_na():
    nvidia = row()
    nvidia["metadata"]["model"] = "nvidia/nemotron-3.5-bolt-e4b"
    result = summarize_acceptance([nvidia])
    assert result["economic_cost_usd"] is None
    assert result["cost_per_strict_accepted_usd"] is None
    rejected = summarize_acceptance([row(correct=0, total_correct=0)])
    assert rejected["cost_per_strict_accepted_usd"] is None
    assert rejected["strict_cost_reason"] == "no acceptable receipts"


def test_aggregate_cost_can_include_modal_lifecycle_and_all_attempts():
    result = summarize_acceptance([row(), row(correct=0)], economic_cost=.12)
    assert result["cost_per_strict_accepted_usd"] == .12
    assert result["cost_per_critical_total_accepted_usd"] == .06
