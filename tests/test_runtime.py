import asyncio

import pytest

from app.evaluation.base import MetricOutcome, MetricSpec
from app.evaluation.engine import evaluate_dataset, evaluate_dataset_async
from app.evaluation.providers import EvaluationContext
from app.evaluation.registry import MetricRegistry
from app.schemas import EvaluationCase


class FakeEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        return 0.75 if vec_a and vec_b else 0.0


def make_case() -> EvaluationCase:
    return EvaluationCase(
        sample_id="runtime-1",
        question_id="q-1",
        answer="回答",
        reference_answer="参考回答",
    )


async def semantic_similarity(
    case: EvaluationCase, context: EvaluationContext
) -> MetricOutcome:
    assert context.embedding_provider is not None
    vectors = await context.embedding_provider.embed(
        [case.answer or "", case.reference_answer or ""]
    )
    score = context.embedding_provider.similarity(vectors[0], vectors[1])
    return MetricOutcome(score=score, reason="fake embedding score")


def semantic_registry() -> MetricRegistry:
    return MetricRegistry(
        [
            MetricSpec(
                name="semantic_similarity_test",
                label="语义相似度测试",
                dimension="end_to_end",
                description="测试异步能力注入。",
                required_fields=("answer", "reference_answer"),
                async_evaluator=semantic_similarity,
                required_capabilities=("embedding",),
            )
        ]
    )


def test_missing_provider_returns_not_configured():
    result = evaluate_dataset(
        [make_case()],
        ["semantic_similarity_test"],
        metric_registry=semantic_registry(),
    )

    item = result["results"][0]
    assert item["status"] == "not_configured"
    assert item["evidence"]["missing_capabilities"] == ["embedding"]
    assert result["summary"][0]["not_configured_count"] == 1


def test_async_metric_runs_with_configured_provider():
    context = EvaluationContext(embedding_provider=FakeEmbeddingProvider())
    result = asyncio.run(
        evaluate_dataset_async(
            [make_case()],
            ["semantic_similarity_test"],
            metric_registry=semantic_registry(),
            context=context,
        )
    )

    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["score"] == 0.75


def test_registry_reports_capability_configuration():
    metric_registry = semantic_registry()
    unavailable = metric_registry.describe_for([make_case()])[0]
    available = metric_registry.describe_for(
        [make_case()], EvaluationContext(embedding_provider=FakeEmbeddingProvider())
    )[0]

    assert unavailable["configured"] is False
    assert unavailable["missing_capabilities"] == ["embedding"]
    assert unavailable["runnable"] is False
    assert available["configured"] is True
    assert available["runnable"] is True


def test_explicit_empty_metric_selection_runs_nothing():
    result = evaluate_dataset([make_case()], [])

    assert result["summary"] == []
    assert result["results"] == []


def test_async_metric_timeout_isolated_as_failure():
    async def slow_metric(
        case: EvaluationCase, context: EvaluationContext
    ) -> MetricOutcome:
        await asyncio.sleep(0.02)
        return MetricOutcome(score=1.0)

    metric_registry = MetricRegistry(
        [
            MetricSpec(
                name="slow",
                label="慢指标",
                dimension="end_to_end",
                description="测试超时隔离。",
                required_fields=("answer",),
                async_evaluator=slow_metric,
            )
        ]
    )
    result = asyncio.run(
        evaluate_dataset_async(
            [make_case()],
            ["slow"],
            metric_registry=metric_registry,
            context=EvaluationContext(timeout_seconds=0.001),
        )
    )

    assert result["results"][0]["status"] == "failed"
    assert result["summary"][0]["failed_count"] == 1


def test_metric_cannot_declare_sync_and_async_evaluators():
    def sync_metric(case: EvaluationCase) -> MetricOutcome:
        return MetricOutcome(score=1.0)

    with pytest.raises(ValueError, match="不能同时声明"):
        MetricSpec(
            name="ambiguous",
            label="冲突指标",
            dimension="end_to_end",
            description="不允许两种执行入口。",
            required_fields=("answer",),
            evaluator=sync_metric,
            async_evaluator=semantic_similarity,
        )


def test_registry_rejects_duplicate_metric_names():
    metric = MetricSpec(
        name="duplicate",
        label="重复指标",
        dimension="end_to_end",
        description="名称必须全局唯一。",
        required_fields=("answer",),
    )

    with pytest.raises(ValueError, match="指标名称重复"):
        MetricRegistry([metric, metric])
