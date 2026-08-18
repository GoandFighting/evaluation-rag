from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.evaluation.engine import evaluate_dataset_async
from app.evaluation.model_providers import (
    ModelProviderError,
    build_evaluation_context_from_environment,
    describe_model_providers,
)
from app.evaluation.providers import EvaluationContext
from app.evaluation.registry import registry
from app.rag_adapters import adapter_registry
from app.rag_adapters.base import RAGAdapter
from app.schemas import RunRequest
from app.services.datasets import (
    DatasetError,
    dataset_store,
    detect_present_fields,
    parse_dataset,
)
from app.services.invocations import invoke_cases, project_cases_for_adapter


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RAG Evaluation Workbench", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
evaluation_context: EvaluationContext = build_evaluation_context_from_environment()
target_rag_max_concurrency = int(os.getenv("TARGET_RAG_MAX_CONCURRENCY", "2"))
target_rag_timeout_seconds = float(os.getenv("TARGET_RAG_TIMEOUT_SECONDS", "120"))

if not 1 <= target_rag_max_concurrency <= 32:
    raise ValueError("TARGET_RAG_MAX_CONCURRENCY 必须在 1 到 32 之间。")
if not 1 <= target_rag_timeout_seconds <= 600:
    raise ValueError("TARGET_RAG_TIMEOUT_SECONDS 必须在 1 到 600 之间。")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> list[dict[str, object]]:
    return registry.describe_for([], evaluation_context)


@app.get("/api/model-providers")
def model_providers() -> list[dict[str, object]]:
    return describe_model_providers(evaluation_context)


@app.post("/api/model-providers/{name}/healthcheck")
async def model_provider_healthcheck(name: str) -> dict[str, object]:
    provider = _get_model_provider(name)
    healthcheck = getattr(provider, "healthcheck", None)
    if not callable(healthcheck):
        raise HTTPException(status_code=500, detail=f"Provider 不支持健康检查：{name}")
    return (await healthcheck()).model_dump()


@app.post("/api/model-providers/{name}/probe")
async def model_provider_probe(name: str) -> dict[str, object]:
    provider = _get_model_provider(name)
    probe = getattr(provider, "probe", None)
    if not callable(probe):
        raise HTTPException(status_code=500, detail=f"Provider 不支持调用探针：{name}")
    try:
        return await probe()
    except ModelProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _get_model_provider(name: str) -> object:
    providers = {
        "llm_judge": evaluation_context.llm_judge,
        "embedding": evaluation_context.embedding_provider,
    }
    if name not in providers:
        raise HTTPException(status_code=422, detail=f"未知模型 Provider：{name}")
    provider = providers[name]
    if provider is None:
        raise HTTPException(status_code=422, detail=f"模型 Provider 尚未配置：{name}")
    return provider


@app.get("/api/rag-adapters")
def rag_adapters() -> list[dict[str, object]]:
    return adapter_registry.describe()


@app.post("/api/rag-adapters/{name}/healthcheck")
async def rag_adapter_healthcheck(name: str) -> dict[str, object]:
    adapter = _get_adapter(name)
    return (await adapter.healthcheck()).model_dump()


def _get_adapter(name: str) -> RAGAdapter:
    adapter = adapter_registry.get(name)
    if adapter is None:
        raise HTTPException(status_code=422, detail=f"未知 RAG Adapter：{name}")
    if not adapter.is_available():
        raise HTTPException(
            status_code=422,
            detail=f"RAG Adapter 尚不可用：{name}",
        )
    return adapter


@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...), adapter_name: str | None = None
) -> dict[str, object]:
    try:
        cases = parse_dataset(file.filename or "dataset.jsonl", await file.read())
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset_id = dataset_store.add(file.filename or "dataset.jsonl", cases)
    effective_cases = cases
    adapter_description = None
    if adapter_name:
        adapter = _get_adapter(adapter_name)
        effective_cases = project_cases_for_adapter(cases, adapter.capabilities)
        adapter_description = {
            "name": adapter.name,
            "label": adapter.label,
            "capabilities": adapter.capabilities.model_dump(),
        }
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "sample_count": len(cases),
        "detected_fields": detect_present_fields(cases),
        "effective_fields": detect_present_fields(effective_cases),
        "adapter": adapter_description,
        "metrics": registry.describe_for(effective_cases, evaluation_context),
    }


@app.post("/api/evaluations/run")
async def run_evaluation(request: RunRequest) -> dict[str, object]:
    dataset = dataset_store.get(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在或服务已重启，请重新上传。")
    cases = dataset.cases
    invocations: list[dict[str, object]] = []
    if request.adapter_name and request.metric_names != []:
        adapter = _get_adapter(request.adapter_name)
        cases, invocations = await invoke_cases(
            cases,
            adapter,
            max_concurrency=target_rag_max_concurrency,
            timeout_seconds=target_rag_timeout_seconds,
        )
    try:
        result = await evaluate_dataset_async(
            cases,
            request.metric_names,
            context=evaluation_context,
        )
        result["adapter_name"] = request.adapter_name
        result["invocations"] = invocations
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
