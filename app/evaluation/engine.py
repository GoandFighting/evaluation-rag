from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from app.evaluation.base import MetricSpec, has_field
from app.evaluation.providers import EvaluationContext
from app.evaluation.registry import DIMENSION_LABELS, MetricRegistry, registry
from app.evaluation.scoring import calculate_module_scores
from app.schemas import EvaluationCase


def _base_result(case: EvaluationCase, metric: MetricSpec) -> dict[str, Any]:
    return {
        "sample_id": case.sample_id,
        "question_id": case.question_id,
        "metric_name": metric.name,
        "metric_label": metric.label,
        "dimension": metric.dimension,
        "dimension_label": DIMENSION_LABELS[metric.dimension],
    }


def _missing_fields(case: EvaluationCase, metric: MetricSpec) -> list[str]:
    missing = [
        field for field in metric.required_fields if not has_field(case, field)
    ]
    if not missing and metric.any_of_fields:
        if not any(has_field(case, field) for field in metric.any_of_fields):
            missing = list(metric.any_of_fields)
    return missing


async def _evaluate_one(
    case: EvaluationCase,
    metric: MetricSpec,
    context: EvaluationContext,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    base = _base_result(case, metric)
    missing_fields = _missing_fields(case, metric)
    if missing_fields:
        return {
            **base,
            "status": "not_applicable",
            "score": None,
            "passed": None,
            "reason": f"缺少字段：{', '.join(missing_fields)}",
            "evidence": {},
        }

    if not metric.implemented:
        return {
            **base,
            "status": "not_implemented",
            "score": None,
            "passed": None,
            "reason": "指标接口已定义，评测实现待开发。",
            "evidence": {},
        }

    missing_capabilities = context.missing_capabilities(
        metric.required_capabilities
    )
    if missing_capabilities:
        return {
            **base,
            "status": "not_configured",
            "score": None,
            "passed": None,
            "reason": f"缺少评测能力：{', '.join(missing_capabilities)}",
            "evidence": {"missing_capabilities": missing_capabilities},
        }

    try:
        async with semaphore:
            if metric.async_evaluator is not None:
                outcome = await asyncio.wait_for(
                    metric.async_evaluator(case, context),
                    timeout=context.timeout_seconds,
                )
            else:
                assert metric.evaluator is not None
                outcome = metric.evaluator(case)
        return {
            **base,
            "status": "success",
            "score": max(0.0, min(1.0, outcome.score)),
            "passed": outcome.passed,
            "reason": outcome.reason,
            "evidence": outcome.evidence,
        }
    except TimeoutError:
        return {
            **base,
            "status": "failed",
            "score": None,
            "passed": None,
            "reason": f"TimeoutError: 指标执行超过 {context.timeout_seconds:g} 秒。",
            "evidence": {},
        }
    except Exception as exc:  # A single metric must not abort the dataset.
        return {
            **base,
            "status": "failed",
            "score": None,
            "passed": None,
            "reason": f"{type(exc).__name__}: {exc}",
            "evidence": {},
        }


def _summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["dimension"], result["metric_name"])].append(result)

    summary = []
    for (dimension, name), items in grouped.items():
        scores = [
            item["score"] for item in items if item["status"] == "success"
        ]
        summary.append(
            {
                "dimension": dimension,
                "dimension_label": DIMENSION_LABELS[dimension],
                "metric_name": name,
                "metric_label": items[0]["metric_label"],
                "average": sum(scores) / len(scores) if scores else None,
                "success_count": len(scores),
                "not_applicable_count": sum(
                    item["status"] == "not_applicable" for item in items
                ),
                "not_configured_count": sum(
                    item["status"] == "not_configured" for item in items
                ),
                "failed_count": sum(
                    item["status"] == "failed" for item in items
                ),
                "not_implemented_count": sum(
                    item["status"] == "not_implemented" for item in items
                ),
            }
        )
    return summary


async def evaluate_dataset_async(
    cases: list[EvaluationCase],
    metric_names: list[str] | None = None,
    *,
    metric_registry: MetricRegistry = registry,
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Evaluate a dataset without blocking model-backed async providers."""

    runtime = context or EvaluationContext()
    if runtime.max_concurrency < 1:
        raise ValueError("max_concurrency 必须大于等于 1。")
    if runtime.timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0。")

    selected = (
        metric_names
        if metric_names is not None
        else [metric.name for metric in metric_registry.all() if metric.implemented]
    )
    unknown = [name for name in selected if metric_registry.get(name) is None]
    if unknown:
        raise ValueError(f"未知指标：{', '.join(unknown)}")

    semaphore = asyncio.Semaphore(runtime.max_concurrency)
    tasks = []
    for case in cases:
        for name in selected:
            metric = metric_registry.get(name)
            assert metric is not None
            tasks.append(_evaluate_one(case, metric, runtime, semaphore))

    results = list(await asyncio.gather(*tasks)) if tasks else []
    summary = _summarize(results)
    return {
        "summary": summary,
        "module_scores": calculate_module_scores(summary),
        "results": results,
        "sample_count": len(cases),
    }


def evaluate_dataset(
    cases: list[EvaluationCase],
    metric_names: list[str] | None = None,
    *,
    metric_registry: MetricRegistry = registry,
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Synchronous compatibility wrapper for tests, scripts, and CLI callers."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            evaluate_dataset_async(
                cases,
                metric_names,
                metric_registry=metric_registry,
                context=context,
            )
        )
    raise RuntimeError(
        "evaluate_dataset 不能在异步事件循环中调用；请 await evaluate_dataset_async。"
    )
