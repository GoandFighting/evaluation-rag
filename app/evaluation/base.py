from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.evaluation.providers import Capability, EvaluationContext
from app.schemas import Dimension, EvaluationCase


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    score: float
    passed: bool | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


Evaluator = Callable[[EvaluationCase], MetricOutcome]
AsyncEvaluator = Callable[
    [EvaluationCase, EvaluationContext], Awaitable[MetricOutcome]
]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    label: str
    dimension: Dimension
    description: str
    required_fields: tuple[str, ...]
    evaluator: Evaluator | None = None
    any_of_fields: tuple[str, ...] = ()
    async_evaluator: AsyncEvaluator | None = None
    required_capabilities: tuple[Capability, ...] = ()

    def __post_init__(self) -> None:
        if self.evaluator is not None and self.async_evaluator is not None:
            raise ValueError(
                f"指标 {self.name} 不能同时声明 evaluator 和 async_evaluator。"
            )

    @property
    def implemented(self) -> bool:
        return self.evaluator is not None or self.async_evaluator is not None


def has_field(case: EvaluationCase, path: str) -> bool:
    current: Any = case
    for part in path.split("."):
        if isinstance(current, list):
            if not current:
                return False
            return any(_has_value(getattr(item, part, None)) for item in current)
        current = getattr(current, part, None)
        if current is None:
            return False
    return _has_value(current)


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}
