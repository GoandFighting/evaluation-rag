import asyncio
import json

import httpx
import pytest

from app.rag_adapters.base import RAGRequest
from app.rag_adapters.confluence_skill import (
    ConfluenceSkillAdapter,
    ConfluenceSkillConfig,
)
from app.rag_adapters.errors import AdapterExecutionError
from app.rag_adapters.smallrag import SmallRAGAdapter, SmallRAGConfig
from app.schemas import EvaluationCase
from app.services.invocations import invoke_cases


def test_confluence_skill_adapter_normalizes_search_results(tmp_path):
    captured = {}

    def search(query, **kwargs):
        captured.update({"query": query, **kwargs})
        return {
            "results": [
                {
                    "title": "休假制度",
                    "score": 0.91,
                    "excerpt": "正式员工每年享有十天带薪年假。",
                    "url": "https://kb.example/pages/viewpage.action?pageId=42",
                }
            ]
        }

    adapter = ConfluenceSkillAdapter(
        ConfluenceSkillConfig(
            skill_dir=tmp_path,
            server="https://kb-search.example",
            top_k=3,
            alpha=0.7,
        ),
        search_function=search,
    )
    output = asyncio.run(
        adapter.invoke(
            RAGRequest(
                request_id="request-1",
                sample_id="sample-1",
                query="年假有几天？",
                user_id="evaluator",
            )
        )
    )

    assert captured == {
        "query": "年假有几天？",
        "top_k": 3,
        "alpha": 0.7,
        "server": "https://kb-search.example",
    }
    assert output.chunks[0].chunk_id == "confluence:42"
    assert output.chunks[0].document_id == "42"
    assert output.chunks[0].retrieval_score == 0.91
    assert output.citations[0]["source"].endswith("pageId=42")
    assert "正式员工每年享有十天带薪年假" in output.answer


def test_confluence_document_id_is_stable_without_page_id(tmp_path):
    adapter = ConfluenceSkillAdapter(
        ConfluenceSkillConfig(skill_dir=tmp_path),
        search_function=lambda query, **kwargs: {"results": []},
    )
    item = {"title": "页面", "url": "https://kb.example/display/TEAM/Page"}

    first = adapter._document_id(item, item["url"], 1)
    second = adapter._document_id(item, item["url"], 9)

    assert first == second
    assert first.startswith("page-")


def test_failed_invocation_does_not_evaluate_stale_uploaded_answer(tmp_path):
    def broken_search(query, **kwargs):
        raise OSError("upstream unavailable")

    adapter = ConfluenceSkillAdapter(
        ConfluenceSkillConfig(skill_dir=tmp_path),
        search_function=broken_search,
    )
    case = EvaluationCase(
        sample_id="sample-1",
        question_id="question-1",
        query="问题",
        answer="文件里的旧答案",
        chunks=[{"content": "文件里的旧片段"}],
    )

    cases, invocations = asyncio.run(invoke_cases([case], adapter))

    assert invocations[0]["status"] == "failed"
    assert cases[0].answer is None
    assert cases[0].chunks == []
    assert cases[0].citations is None


def test_smallrag_adapter_returns_generated_answer_and_full_contexts():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready", "checks": {}})
        captured["request_id"] = request.headers.get("X-Request-ID")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "smallrag-trace-1",
                "answer": "正式员工每年享有十天带薪年假。[1]",
                "model": "test-model",
                "contexts": [
                    {
                        "chunk_id": "confluence:42",
                        "document_id": "42",
                        "document_name": "休假制度",
                        "content": "完整文档正文：正式员工每年享有十天带薪年假。",
                        "source": "https://kb.example/pages/42",
                        "rank": 1,
                        "retrieval_score": 0.98,
                        "space": "HR",
                        "truncated": False,
                    }
                ],
                "citations": [
                    {
                        "page_id": "42",
                        "title": "休假制度",
                        "url": "https://kb.example/pages/42",
                        "score": 0.98,
                        "excerpt": "年假制度摘要",
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 18},
                "latency_ms": {
                    "retrieval": 10,
                    "page_fetch": 20,
                    "generation": 30,
                    "total": 60,
                },
            },
        )

    adapter = SmallRAGAdapter(
        SmallRAGConfig(base_url="http://smallrag.test", top_k=3, alpha=0.7),
        transport=httpx.MockTransport(handler),
    )
    health = asyncio.run(adapter.healthcheck())
    output = asyncio.run(
        adapter.invoke(
            RAGRequest(
                request_id="evaluation-request-1",
                sample_id="sample-1",
                query="员工年假有几天？",
                user_id="evaluator",
            )
        )
    )

    assert health.ok is True
    assert captured["request_id"] == "evaluation-request-1"
    assert captured["payload"] == {
        "query": "员工年假有几天？",
        "top_k": 3,
        "alpha": 0.7,
        "include_retrieval": True,
    }
    assert output.answer.endswith("[1]")
    assert output.chunks[0].content.startswith("完整文档正文")
    assert output.chunks[0].model_extra["truncated"] is False
    assert output.citations[0]["chunk_id"] == "confluence:42"
    assert output.trace_id == "smallrag-trace-1"
    assert output.latency_ms == 60
    assert output.token_usage["input_tokens"] == 120


def test_smallrag_adapter_exposes_structured_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={
                "error": {
                    "code": "model_upstream_error",
                    "message": "Model gateway returned HTTP 429: rate limited",
                    "request_id": "smallrag-failed-request",
                }
            },
        )

    adapter = SmallRAGAdapter(
        SmallRAGConfig(base_url="http://smallrag.test"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AdapterExecutionError) as exc_info:
        asyncio.run(
            adapter.invoke(
                RAGRequest(
                    request_id="evaluation-request-2",
                    sample_id="sample-2",
                    query="问题",
                    user_id="evaluator",
                )
            )
        )

    message = str(exc_info.value)
    assert "model_upstream_error" in message
    assert "HTTP 429" in message
    assert "smallrag-failed-request" in message
