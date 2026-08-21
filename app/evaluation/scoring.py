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

RETRIEVAL_METRIC_WEIGHTS: dict[str, float] = {
    "hit_rate_at_k": 0.05,
    "recall_at_k": 0.20,
    "precision_at_k": 0.15,
    "mrr": 0.10,
    "ndcg_at_k": 0.15,
    "context_relevance": 0.05,
    "context_relevance_semantic": 0.10,
    "context_precision": 0.20,
}

GENERATION_METRIC_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.05,
    "faithfulness_semantic": 0.10,
    "faithfulness_llm": 0.25,
    "factual_consistency": 0.05,
    "factual_correctness": 0.25,
    "context_utilization": 0.15,
    "citation_correctness": 0.10,
    "citation_completeness": 0.05,
}

MODULE_CONFIG = {
    "end_to_end": {
        "dimension_label": "端到端回答评测",
        "label": "端到端综合得分",
        "weights": END_TO_END_METRIC_WEIGHTS,
    },
    "retrieval": {
        "dimension_label": "检索模块评测",
        "label": "检索综合得分",
        "weights": RETRIEVAL_METRIC_WEIGHTS,
    },
    "generation": {
        "dimension_label": "生成模块评测",
        "label": "生成综合得分",
        "weights": GENERATION_METRIC_WEIGHTS,
    },
}


def calculate_module_scores(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate module scores from successful metric-level averages only."""

    scores = []
    for dimension, config in MODULE_CONFIG.items():
        dimension_summary = [
            item for item in summary if item["dimension"] == dimension
        ]
        if not dimension_summary:
            continue
        weights = config["weights"]
        eligible = [
            item
            for item in dimension_summary
            if item["metric_name"] in weights
            and item["average"] is not None
            and item["success_count"] > 0
        ]
        weight_sum = sum(weights[item["metric_name"]] for item in eligible)
        components = []
        for item in eligible:
            base_weight = weights[item["metric_name"]]
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
        scores.append(
            {
                "dimension": dimension,
                "dimension_label": config["dimension_label"],
                "label": config["label"],
                "score": sum(item["contribution"] for item in components)
                if components
                else None,
                "successful_metric_count": len(components),
                "configured_weight_sum": weight_sum,
                "components": components,
            }
        )
    return scores
