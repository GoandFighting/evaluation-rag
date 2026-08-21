import json

import httpx
from fastapi.testclient import TestClient

from app.main import app, evaluation_context
from app.rag_adapters import adapter_registry


def test_metrics_expose_runtime_capabilities(monkeypatch):
    monkeypatch.setattr(evaluation_context, "embedding_provider", None)
    with TestClient(app) as client:
        response = client.get("/api/metrics")

    assert response.status_code == 200
    metrics = {item["name"]: item for item in response.json()}
    semantic = metrics["faithfulness_semantic"]
    assert semantic["required_capabilities"] == ["embedding"]
    assert semantic["missing_capabilities"] == ["embedding"]
    assert semantic["configured"] is False
    assert semantic["runnable"] is False


def test_model_provider_api_exposes_configuration_state():
    with TestClient(app) as client:
        response = client.get("/api/model-providers")

    assert response.status_code == 200
    providers = {item["name"]: item for item in response.json()}
    assert providers["llm_judge"]["configured"] is (
        evaluation_context.llm_judge is not None
    )
    assert providers["embedding"]["configured"] is (
        evaluation_context.embedding_provider is not None
    )


def test_smallrag_adapter_is_exposed_to_the_frontend():
    with TestClient(app) as client:
        response = client.get("/api/rag-adapters")

    assert response.status_code == 200
    smallrag = next(item for item in response.json() if item["name"] == "smallrag")
    assert smallrag["available"] is True
    assert smallrag["capabilities"] == {
        "answer": True,
        "chunks": True,
        "citations": True,
        "trace": True,
        "streaming": False,
    }


def test_upload_then_run_evaluation():
    case = {
        "sample_id": "q-1::a",
        "question_id": "q-1",
        "query": "员工年假有几天？",
        "answer": "员工年假有十天。",
        "reference_answer": "员工年假有十天。",
    }
    with TestClient(app) as client:
        upload = client.post(
            "/api/datasets/upload",
            files={"file": ("sample.jsonl", json.dumps(case, ensure_ascii=False).encode(), "application/jsonl")},
        )
        assert upload.status_code == 200
        body = upload.json()
        assert body["sample_count"] == 1
        assert "reference_answer" in body["detected_fields"]

        run = client.post(
            "/api/evaluations/run",
            json={"dataset_id": body["dataset_id"], "metric_names": ["token_f1"]},
        )
        assert run.status_code == 200
        assert run.json()["summary"][0]["average"] == 1.0


def test_explicit_empty_metric_selection_runs_nothing():
    case = {
        "sample_id": "q-empty",
        "question_id": "q-empty",
        "query": "问题",
        "answer": "回答",
    }
    with TestClient(app) as client:
        upload = client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "sample.jsonl",
                    json.dumps(case, ensure_ascii=False).encode(),
                    "application/jsonl",
                )
            },
        )
        run = client.post(
            "/api/evaluations/run",
            json={"dataset_id": upload.json()["dataset_id"], "metric_names": []},
        )

    assert run.status_code == 200
    assert run.json()["results"] == []


def test_confluence_adapter_upload_and_run(monkeypatch):
    adapter = adapter_registry.get("confluence_skill")

    def search(query, **kwargs):
        return {
            "results": [
                {
                    "title": "休假制度",
                    "score": 0.98,
                    "excerpt": "正式员工每年享有十天带薪年假。",
                    "url": "https://kb.example/view?pageId=42",
                }
            ]
        }

    monkeypatch.setattr(adapter, "_search_function", search)
    generated_answer = (
        "1. 休假制度\n正式员工每年享有十天带薪年假。\n"
        "来源：https://kb.example/view?pageId=42"
    )
    case = {
        "sample_id": "q-live",
        "question_id": "q-live",
        "query": "员工年假有几天？",
        "reference_answer": generated_answer,
    }

    with TestClient(app) as client:
        adapters = client.get("/api/rag-adapters")
        assert adapters.status_code == 200
        confluence = next(
            item for item in adapters.json() if item["name"] == "confluence_skill"
        )
        assert confluence["capabilities"]["chunks"] is True

        upload = client.post(
            "/api/datasets/upload?adapter_name=confluence_skill",
            files={
                "file": (
                    "golden.jsonl",
                    json.dumps(case, ensure_ascii=False).encode(),
                    "application/jsonl",
                )
            },
        )
        assert upload.status_code == 200
        assert "answer" in upload.json()["effective_fields"]
        assert "chunks.content" in upload.json()["effective_fields"]

        run = client.post(
            "/api/evaluations/run",
            json={
                "dataset_id": upload.json()["dataset_id"],
                "metric_names": ["token_f1"],
                "adapter_name": "confluence_skill",
            },
        )

    assert run.status_code == 200
    assert run.json()["summary"][0]["average"] == 1.0
    assert run.json()["invocations"][0]["status"] == "success"
    assert run.json()["invocations"][0]["chunk_count"] == 1
    assert run.json()["invocations"][0]["answer"] == generated_answer
    assert run.json()["invocations"][0]["chunks"][0]["document_id"] == "42"


def test_smallrag_adapter_runs_through_the_evaluation_api(monkeypatch):
    adapter = adapter_registry.get("smallrag")
    generated_answer = "正式员工每年享有十天带薪年假。[1]"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready", "checks": {}})
        return httpx.Response(
            200,
            json={
                "request_id": "smallrag-api-trace",
                "answer": generated_answer,
                "contexts": [
                    {
                        "chunk_id": "confluence:42",
                        "document_id": "42",
                        "document_name": "休假制度",
                        "content": "正式员工每年享有十天带薪年假。",
                        "source": "https://kb.example/pages/42",
                        "rank": 1,
                        "retrieval_score": 0.98,
                    }
                ],
                "citations": [{"page_id": "42"}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "latency_ms": {"total": 75},
            },
        )

    monkeypatch.setattr(adapter, "_transport", httpx.MockTransport(handler))
    case = {
        "sample_id": "q-smallrag",
        "question_id": "q-smallrag",
        "query": "员工年假有几天？",
        "reference_answer": generated_answer,
    }

    with TestClient(app) as client:
        health = client.post("/api/rag-adapters/smallrag/healthcheck")
        upload = client.post(
            "/api/datasets/upload?adapter_name=smallrag",
            files={
                "file": (
                    "golden.jsonl",
                    json.dumps(case, ensure_ascii=False).encode(),
                    "application/jsonl",
                )
            },
        )
        run = client.post(
            "/api/evaluations/run",
            json={
                "dataset_id": upload.json()["dataset_id"],
                "metric_names": ["token_f1", "context_relevance"],
                "adapter_name": "smallrag",
            },
        )

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert upload.status_code == 200
    assert "chunks.content" in upload.json()["effective_fields"]
    assert run.status_code == 200
    assert run.json()["invocations"][0]["trace_id"] == "smallrag-api-trace"
    assert run.json()["invocations"][0]["token_usage"]["input_tokens"] == 100
    assert run.json()["invocations"][0]["chunks"][0]["content"].startswith("正式员工")
    token_f1 = next(
        item for item in run.json()["summary"] if item["metric_name"] == "token_f1"
    )
    assert token_f1["average"] == 1.0
