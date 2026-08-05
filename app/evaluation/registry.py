from __future__ import annotations

from app.evaluation.base import MetricSpec, has_field
from app.evaluation.modules.end_to_end import METRICS as END_TO_END_METRICS
from app.evaluation.modules.generation import METRICS as GENERATION_METRICS
from app.evaluation.modules.retrieval import METRICS as RETRIEVAL_METRICS
from app.evaluation.providers import EvaluationContext
from app.schemas import EvaluationCase


DIMENSION_LABELS = {
    "end_to_end": "端到端回答评测",
    "retrieval": "检索模块评测",
    "generation": "生成模块评测",
}


class MetricRegistry:
    def __init__(self, metrics: list[MetricSpec]) -> None:
        self._metrics: dict[str, MetricSpec] = {}
        for metric in metrics:
            if metric.name in self._metrics:
                raise ValueError(f"指标名称重复：{metric.name}")
            self._metrics[metric.name] = metric

    def get(self, name: str) -> MetricSpec | None:
        return self._metrics.get(name)

    def all(self) -> list[MetricSpec]:
        return list(self._metrics.values())

    def describe_for(
        self,
        cases: list[EvaluationCase],
        context: EvaluationContext | None = None,
    ) -> list[dict[str, object]]:
        runtime = context or EvaluationContext()
        descriptions = []
        for metric in self.all():
            eligible = sum(
                all(has_field(case, field) for field in metric.required_fields)
                and (
                    not metric.any_of_fields
                    or any(has_field(case, field) for field in metric.any_of_fields)
                )
                for case in cases
            )
            missing_capabilities = runtime.missing_capabilities(
                metric.required_capabilities
            )
            configured = not missing_capabilities
            descriptions.append(
                {
                    "name": metric.name,
                    "label": metric.label,
                    "dimension": metric.dimension,
                    "dimension_label": DIMENSION_LABELS[metric.dimension],
                    "description": metric.description,
                    "required_fields": list(metric.required_fields),
                    "any_of_fields": list(metric.any_of_fields),
                    "required_capabilities": list(metric.required_capabilities),
                    "missing_capabilities": missing_capabilities,
                    "implemented": metric.implemented,
                    "configured": configured,
                    "eligible_samples": eligible,
                    "total_samples": len(cases),
                    "field_ready": eligible > 0,
                    "runnable": (
                        metric.implemented and configured and eligible > 0
                    ),
                }
            )
        return descriptions


registry = MetricRegistry(END_TO_END_METRICS + RETRIEVAL_METRICS + GENERATION_METRICS)
