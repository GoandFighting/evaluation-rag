from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import ValidationError

from app.schemas import Chunk, EvaluationCase


class DatasetError(ValueError):
    pass


def parse_dataset(filename: str, payload: bytes) -> list[EvaluationCase]:
    if len(payload) > 20 * 1024 * 1024:
        raise DatasetError("文件不能超过 20 MB。")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("数据集必须使用 UTF-8 编码。") from exc

    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif suffix == ".json":
            parsed = json.loads(text)
            records = parsed.get("items") if isinstance(parsed, dict) else parsed
        else:
            raise DatasetError("目前仅支持 .json 和 .jsonl 文件。")
    except json.JSONDecodeError as exc:
        raise DatasetError(f"JSON 解析失败：第 {exc.lineno} 行，第 {exc.colno} 列。") from exc

    if not isinstance(records, list) or not records:
        raise DatasetError("数据集必须包含至少一条 JSON 对象记录。")
    if not all(isinstance(record, dict) for record in records):
        raise DatasetError("每条数据必须是 JSON 对象。")

    try:
        return [_normalize_record(record, index) for index, record in enumerate(records, 1)]
    except (TypeError, ValidationError, ValueError) as exc:
        raise DatasetError(f"数据结构不合法：{exc}") from exc


def _normalize_record(record: dict[str, Any], index: int) -> EvaluationCase:
    data = dict(record)
    ground_truth = data.get("ground_truth") or {}
    if not isinstance(ground_truth, dict):
        raise DatasetError("ground_truth 必须是对象。")

    query = _first(data, "query", "question", "input")
    answer = _first(data, "answer", "result", "response")
    raw_chunks = _first(data, "chunks", "docs", "contexts") or []
    if isinstance(raw_chunks, str):
        raw_chunks = [raw_chunks]
    if not isinstance(raw_chunks, list):
        raise DatasetError("chunks/docs 必须是数组。")
    chunks = [_normalize_chunk(value, rank) for rank, value in enumerate(raw_chunks, 1)]

    question_id = str(data.get("question_id") or data.get("id") or f"q-{index:04d}")
    sample_id = str(data.get("sample_id") or f"{question_id}::sample-{index:04d}")
    key_points = data.get("key_points", ground_truth.get("key_points"))
    if isinstance(key_points, str):
        key_points = [key_points]

    return EvaluationCase(
        schema_version=str(data.get("schema_version", "1.0")),
        sample_id=sample_id,
        question_id=question_id,
        query=str(query).strip() if query is not None else None,
        chunks=chunks,
        answer=str(answer).strip() if answer is not None else None,
        reference_answer=data.get("reference_answer", ground_truth.get("reference_answer")),
        relevant_doc_ids=data.get("relevant_doc_ids", ground_truth.get("relevant_doc_ids")),
        relevant_chunk_ids=data.get(
            "relevant_chunk_ids", ground_truth.get("relevant_chunk_ids")
        ),
        key_points=key_points,
        expected_behavior=data.get(
            "expected_behavior", ground_truth.get("expected_behavior")
        ),
        expected_format=data.get("expected_format", ground_truth.get("expected_format")),
        citations=data.get("citations"),
        target=data.get("target") or {},
        execution=data.get("execution") or {},
        tags=data.get("tags") or [],
        metadata=data.get("metadata") or {},
    )


def _normalize_chunk(value: Any, rank: int) -> Chunk:
    if isinstance(value, str):
        return Chunk(content=value, rank=rank)
    if not isinstance(value, dict):
        raise DatasetError("每个 chunk 必须是字符串或对象。")
    normalized = dict(value)
    normalized["content"] = _first(normalized, "content", "text", "chunk") or ""
    normalized.setdefault("chunk_id", normalized.get("doc_id"))
    normalized.setdefault("document_id", normalized.get("doc_id"))
    normalized.setdefault("rank", rank)
    return Chunk.model_validate(normalized)


def _first(data: dict[str, Any], *names: str) -> Any:
    return next((data[name] for name in names if data.get(name) is not None), None)


@dataclass(slots=True)
class StoredDataset:
    filename: str
    cases: list[EvaluationCase]


class InMemoryDatasetStore:
    """MVP store; replace with persistent storage without changing API contracts."""

    def __init__(self) -> None:
        self._items: dict[str, StoredDataset] = {}
        self._lock = Lock()

    def add(self, filename: str, cases: list[EvaluationCase]) -> str:
        dataset_id = str(uuid.uuid4())
        with self._lock:
            self._items[dataset_id] = StoredDataset(filename=filename, cases=cases)
        return dataset_id

    def get(self, dataset_id: str) -> StoredDataset | None:
        with self._lock:
            return self._items.get(dataset_id)


dataset_store = InMemoryDatasetStore()


def detect_present_fields(cases: list[EvaluationCase]) -> list[str]:
    fields: set[str] = set()
    for case in cases:
        dumped = case.model_dump(exclude_none=True)
        for name, value in dumped.items():
            if name == "chunks":
                for chunk in value:
                    fields.update(
                        f"chunks.{key}"
                        for key, chunk_value in chunk.items()
                        if chunk_value not in (None, "", [], {})
                    )
            elif value not in (None, "", [], {}):
                fields.add(name)
    return sorted(fields)
