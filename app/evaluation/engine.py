from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.evaluation.base import has_field
from app.evaluation.registry import DIMENSION_LABELS, MetricRegistry, registry
from app.schemas import EvaluationCase


def evaluate_dataset(
    cases: list[EvaluationCase],
    metric_names: list[str] | None = None,
    *,
    metric_registry: MetricRegistry = registry,
) -> dict[str, Any]:
    selected = metric_names or [metric.name for metric in metric_registry.all() if metric.implemented]
    unknown = [name for name in selected if metric_registry.get(name) is None]
    if unknown:
        raise ValueError(f"未知指标：{', '.join(unknown)}")

    results: list[dict[str, Any]] = []
    for case in cases:
        for name in selected:
            metric = metric_registry.get(name)
            assert metric is not None
            missing = [field for field in metric.required_fields if not has_field(case, field)]
            base = {
                "sample_id": case.sample_id,
                "question_id": case.question_id,
                "metric_name": metric.name,
                "metric_label": metric.label,
                "dimension": metric.dimension,
                "dimension_label": DIMENSION_LABELS[metric.dimension],
            }
            if missing:
                results.append({**base, "status": "not_applicable", "score": None, "passed": None, "reason": f"缺少字段：{', '.join(missing)}", "evidence": {}})
                continue
            if metric.evaluator is None:
                results.append({**base, "status": "not_implemented", "score": None, "passed": None, "reason": "指标接口已定义，评测实现待开发。", "evidence": {}})
                continue
            try:
                outcome = metric.evaluator(case)
                results.append({**base, "status": "success", "score": max(0.0, min(1.0, outcome.score)), "passed": outcome.passed, "reason": outcome.reason, "evidence": outcome.evidence})
            except Exception as exc:  # A single metric must not abort the dataset.
                results.append({**base, "status": "failed", "score": None, "passed": None, "reason": f"{type(exc).__name__}: {exc}", "evidence": {}})

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["dimension"], result["metric_name"])].append(result)
    summary = []
    for (dimension, name), items in grouped.items():
        scores = [item["score"] for item in items if item["status"] == "success"]
        summary.append(
            {
                "dimension": dimension,
                "dimension_label": DIMENSION_LABELS[dimension],
                "metric_name": name,
                "metric_label": items[0]["metric_label"],
                "average": sum(scores) / len(scores) if scores else None,
                "success_count": len(scores),
                "not_applicable_count": sum(item["status"] == "not_applicable" for item in items),
                "failed_count": sum(item["status"] == "failed" for item in items),
                "not_implemented_count": sum(item["status"] == "not_implemented" for item in items),
            }
        )
    return {"summary": summary, "results": results, "sample_count": len(cases)}
