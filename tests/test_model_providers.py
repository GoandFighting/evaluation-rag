import asyncio
import json
import math

import httpx
import pytest

from app.evaluation.model_providers import (
    EmbeddingConfig,
    LLMJudgeConfig,
    ModelProviderError,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMJudge,
    build_evaluation_context_from_environment,
)


def _model_service_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen3-8b"},
                    {"id": "qwen3-embedding-0.6b"},
                ]
            },
        )
    if request.url.path == "/v1/chat/completions":
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-8b"
        assert payload["messages"][-1]["role"] == "user"
        assert payload["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>检查依据</think>\n"
                                "{\"score\": 1, \"reason\": \"正确\"}"
                            )
                        }
                    }
                ]
            },
        )
    if request.url.path == "/v1/embeddings":
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-embedding-0.6b"
        assert all(payload["input"])
        if len(payload["input"]) == 1:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1, 0]}]},
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )
    return httpx.Response(404)


def test_llm_judge_calls_openai_compatible_endpoint_and_parses_json():
    provider = OpenAICompatibleLLMJudge(
        LLMJudgeConfig(base_url="http://llm.test"),
        transport=httpx.MockTransport(_model_service_handler),
    )

    result = asyncio.run(provider.judge_json("判断回答是否正确"))

    assert result == {"score": 1, "reason": "正确"}


def test_embedding_provider_preserves_input_order_and_cosine_similarity():
    provider = OpenAICompatibleEmbeddingProvider(
        EmbeddingConfig(base_url="http://embedding.test"),
        transport=httpx.MockTransport(_model_service_handler),
    )

    vectors = asyncio.run(provider.embed(["问题", "参考答案"]))

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert provider.similarity(vectors[0], vectors[1]) == 0.0
    assert provider.similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(
        1 / math.sqrt(2)
    )


def test_provider_probes_call_the_core_inference_endpoints():
    llm = OpenAICompatibleLLMJudge(
        LLMJudgeConfig(base_url="http://llm.test"),
        transport=httpx.MockTransport(_model_service_handler),
    )
    embedding = OpenAICompatibleEmbeddingProvider(
        EmbeddingConfig(base_url="http://embedding.test"),
        transport=httpx.MockTransport(_model_service_handler),
    )

    llm_result = asyncio.run(llm.probe())
    embedding_result = asyncio.run(embedding.probe())

    assert llm_result["endpoint"] == "/v1/chat/completions"
    assert llm_result["ok"] is True
    assert embedding_result["endpoint"] == "/v1/embeddings"
    assert embedding_result["vector_dimension"] == 2


@pytest.mark.parametrize(
    ("provider", "expected_name"),
    [
        (
            OpenAICompatibleLLMJudge(
                LLMJudgeConfig(base_url="http://llm.test"),
                transport=httpx.MockTransport(_model_service_handler),
            ),
            "llm_judge",
        ),
        (
            OpenAICompatibleEmbeddingProvider(
                EmbeddingConfig(base_url="http://embedding.test"),
                transport=httpx.MockTransport(_model_service_handler),
            ),
            "embedding",
        ),
    ],
)
def test_provider_healthcheck_verifies_health_and_model(provider, expected_name):
    health = asyncio.run(provider.healthcheck())

    assert health.ok is True
    assert health.name == expected_name
    assert health.model is not None


def test_provider_surfaces_upstream_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": {"message": "model is loading"}}
        )

    provider = OpenAICompatibleLLMJudge(
        LLMJudgeConfig(base_url="http://llm.test"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelProviderError, match="503.*model is loading"):
        asyncio.run(provider.judge("判断回答是否正确"))


def test_environment_builder_only_enables_configured_providers(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("EVAL_EMBEDDING_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("EVAL_MODEL_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("EVAL_METRIC_TIMEOUT_SECONDS", "90")

    context = build_evaluation_context_from_environment()

    assert context.llm_judge is not None
    assert context.llm_judge.model == "qwen3-8b"
    assert context.embedding_provider is not None
    assert context.embedding_provider.model == "qwen3-embedding-0.6b"
    assert context.max_concurrency == 2
    assert context.timeout_seconds == 90


def test_environment_builder_leaves_unconfigured_providers_disabled(monkeypatch):
    monkeypatch.delenv("EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_EMBEDDING_BASE_URL", raising=False)

    context = build_evaluation_context_from_environment()

    assert context.llm_judge is None
    assert context.embedding_provider is None
