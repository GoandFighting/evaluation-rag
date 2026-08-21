from __future__ import annotations

from typing import Any


# There is no universal aggregate-score weighting in RAGAS or LangSmith. These
# defaults prioritize answer correctness, then relevance and completeness. Low-
# cost and model-backed variants of the same concept share that concept's weight
# so duplicated signals do not dominate the module score.
END_TO_END_METRIC_WEIGHTS: dict[str, float] = {
    "answer_relevance_lexical": 0.03,
    "answer_relevance_semantic": 0.07,
    "answer_relevance_llm": 0.10,
    "answer_correctness": 0.25,
    "semantic_similarity": 0.07,
    "token_f1": 0.03,
    "completeness_llm": 0.12,
    "key_point_completeness": 0.08,
    "format_compliance": 0.10,
    "refusal_quality": 0.05,
    "refusal_quality_llm": 0.10,
}


def calculate_module_scores(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate module scores from successful metric-level averages only."""

    end_to_end_summary = [
        item for item in summary if item["dimension"] == "end_to_end"
    ]
    if not end_to_end_summary:
        return []
    eligible = [
        item
        for item in end_to_end_summary
        if item["metric_name"] in END_TO_END_METRIC_WEIGHTS
        and item["average"] is not None
        and item["success_count"] > 0
    ]
    weight_sum = sum(
        END_TO_END_METRIC_WEIGHTS[item["metric_name"]] for item in eligible
    )
    components = []
    for item in eligible:
        base_weight = END_TO_END_METRIC_WEIGHTS[item["metric_name"]]
        normalized_weight = base_weight / weight_sum
        contribution = item["average"] * normalized_weight
        components.append(
            {
                "metric_name": item["metric_name"],
                "metric_label": item["metric_label"],
                "average": item["average"],
                "success_count": item["success_count"],
                "base_weight": base_weight,
                "normalized_weight": normalized_weight,
                "contribution": contribution,
            }
        )

    return [
        {
            "dimension": "end_to_end",
            "dimension_label": "端到端回答评测",
            "label": "端到端综合得分",
            "score": sum(item["contribution"] for item in components)
            if components
            else None,
            "successful_metric_count": len(components),
            "configured_weight_sum": weight_sum,
            "components": components,
        }
    ]
