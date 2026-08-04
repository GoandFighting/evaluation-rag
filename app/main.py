from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.evaluation.engine import evaluate_dataset
from app.evaluation.registry import registry
from app.schemas import RunRequest
from app.services.datasets import (
    DatasetError,
    dataset_store,
    detect_present_fields,
    parse_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RAG Evaluation Workbench", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> list[dict[str, object]]:
    return registry.describe_for([])


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        cases = parse_dataset(file.filename or "dataset.jsonl", await file.read())
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset_id = dataset_store.add(file.filename or "dataset.jsonl", cases)
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "sample_count": len(cases),
        "detected_fields": detect_present_fields(cases),
        "metrics": registry.describe_for(cases),
    }


@app.post("/api/evaluations/run")
def run_evaluation(request: RunRequest) -> dict[str, object]:
    dataset = dataset_store.get(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在或服务已重启，请重新上传。")
    try:
        return evaluate_dataset(dataset.cases, request.metric_names)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
