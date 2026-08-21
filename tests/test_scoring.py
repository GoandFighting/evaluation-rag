import pytest

from app.evaluation.scoring import calculate_module_scores


def summary_item(name: str, average: float | None, success_count: int = 1):
    return {
        "dimension": "end_to_end",
        "metric_name": name,
        "metric_label": name,
        "average": average,
        "success_count": success_count,
    }


def test_module_score_renormalizes_weights_over_successful_metrics():
    result = calculate_module_scores(
        [
            summary_item("answer_correctness", 0.8),
            summary_item("token_f1", 0.5),
            summary_item("format_compliance", None, success_count=0),
        ]
    )[0]

    assert result["score"] == pytest.approx((0.8 * 0.25 + 0.5 * 0.03) / 0.28)
    assert result["successful_metric_count"] == 2
    assert result["configured_weight_sum"] == pytest.approx(0.28)
    assert {item["metric_name"] for item in result["components"]} == {
        "answer_correctness",
        "token_f1",
    }


def test_module_score_is_empty_without_successful_metric_scores():
    result = calculate_module_scores(
        [summary_item("answer_correctness", None, success_count=0)]
    )[0]

    assert result["score"] is None
    assert result["components"] == []


def test_module_score_is_omitted_when_end_to_end_module_was_not_run():
    assert calculate_module_scores([]) == []
