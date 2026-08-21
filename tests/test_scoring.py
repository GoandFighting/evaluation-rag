import pytest

from app.evaluation.scoring import (
    GENERATION_METRIC_WEIGHTS,
    RETRIEVAL_METRIC_WEIGHTS,
    calculate_module_scores,
)


def summary_item(
    name: str,
    average: float | None,
    success_count: int = 1,
    dimension: str = "end_to_end",
):
    return {
        "dimension": dimension,
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


def test_retrieval_and_generation_weights_each_sum_to_one():
    assert sum(RETRIEVAL_METRIC_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(GENERATION_METRIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_retrieval_and_generation_receive_separate_module_scores():
    results = calculate_module_scores(
        [
            summary_item("recall_at_k", 0.8, dimension="retrieval"),
            summary_item("context_precision", 0.6, dimension="retrieval"),
            summary_item("faithfulness_llm", 0.9, dimension="generation"),
            summary_item("factual_correctness", 0.7, dimension="generation"),
        ]
    )

    by_dimension = {item["dimension"]: item for item in results}
    assert by_dimension["retrieval"]["score"] == pytest.approx(
        (0.8 * 0.20 + 0.6 * 0.20) / 0.40
    )
    assert by_dimension["generation"]["score"] == pytest.approx(
        (0.9 * 0.25 + 0.7 * 0.25) / 0.50
    )
