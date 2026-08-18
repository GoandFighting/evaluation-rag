import asyncio

from app.evaluation.engine import evaluate_dataset_async
from app.evaluation.providers import EvaluationContext
from app.evaluation.registry import registry
from app.schemas import EvaluationCase


class FakeEmbeddingProvider:
    def __init__(self, similarity: float = 0.8) -> None:
        self.similarity_score = similarity
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0], [0.8, 0.6]]

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        return self.similarity_score


class FakeLLMJudge:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {
            "score": 0.85,
            "passed": True,
            "reason": "回答满足当前评分要求。",
            "details": {"matched": ["十天带薪年假"]},
        }
        self.prompts: list[str] = []

    async def judge(self, prompt: str, **kwargs) -> str:
        return "unused"

    async def judge_json(self, prompt: str, **kwargs) -> dict:
        self.prompts.append(prompt)
        assert "system_prompt" in kwargs
        return self.response


def make_case(**overrides) -> EvaluationCase:
    values = {
        "sample_id": "model-metric-1",
        "question_id": "q-1",
        "query": "正式员工每年有多少天带薪年假？",
        "answer": "正式员工每年享有十天带薪年假。",
        "reference_answer": "正式员工每年享有十天带薪年假。",
        "key_points": ["十天", "带薪年假"],
        "expected_behavior": "answer",
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_model_metrics_are_registered_with_capability_contracts():
    context = EvaluationContext(
        embedding_provider=FakeEmbeddingProvider(),
        llm_judge=FakeLLMJudge(),
    )
    descriptions = {
        item["name"]: item for item in registry.describe_for([make_case()], context)
    }

    assert descriptions["semantic_similarity"]["required_capabilities"] == [
        "embedding"
    ]
    assert descriptions["answer_correctness"]["required_capabilities"] == [
        "llm_judge"
    ]
    assert descriptions["answer_relevance_semantic"]["runnable"] is True
    assert descriptions["answer_relevance_llm"]["runnable"] is True


def test_embedding_and_llm_end_to_end_metrics_run_successfully():
    embedding = FakeEmbeddingProvider(0.8)
    judge = FakeLLMJudge()
    metric_names = [
        "answer_relevance_semantic",
        "answer_relevance_llm",
        "answer_correctness",
        "completeness_llm",
        "semantic_similarity",
        "refusal_quality_llm",
    ]

    result = asyncio.run(
        evaluate_dataset_async(
            [make_case()],
            metric_names,
            context=EvaluationContext(
                embedding_provider=embedding,
                llm_judge=judge,
            ),
        )
    )

    assert len(result["results"]) == len(metric_names)
    assert all(item["status"] == "success" for item in result["results"])
    scores = {item["metric_name"]: item["score"] for item in result["results"]}
    assert scores["semantic_similarity"] == 0.8
    assert scores["answer_relevance_semantic"] == 0.8
    assert scores["answer_correctness"] == 0.85
    assert len(embedding.calls) == 2
    assert len(judge.prompts) == 4


def test_reference_based_model_metric_is_not_applicable_without_reference():
    result = asyncio.run(
        evaluate_dataset_async(
            [make_case(reference_answer=None)],
            ["answer_correctness", "semantic_similarity"],
            context=EvaluationContext(
                embedding_provider=FakeEmbeddingProvider(),
                llm_judge=FakeLLMJudge(),
            ),
        )
    )

    assert all(item["status"] == "not_applicable" for item in result["results"])


def test_invalid_judge_score_is_isolated_as_metric_failure():
    result = asyncio.run(
        evaluate_dataset_async(
            [make_case()],
            ["answer_correctness"],
            context=EvaluationContext(
                llm_judge=FakeLLMJudge(
                    {
                        "score": 5,
                        "passed": True,
                        "reason": "invalid scale",
                    }
                )
            ),
        )
    )

    item = result["results"][0]
    assert item["status"] == "failed"
    assert "0 到 1" in item["reason"]
