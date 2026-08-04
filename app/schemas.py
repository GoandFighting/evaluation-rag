from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Dimension = Literal["end_to_end", "retrieval", "generation"]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str | None = None
    content: str = ""
    document_id: str | None = None
    document_name: str | None = None
    source: str | None = None
    page: int | None = None
    rank: int | None = None
    retrieval_score: float | None = None


class EvaluationCase(BaseModel):
    """Flexible canonical record used by every evaluation module."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    sample_id: str
    question_id: str
    query: str | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    answer: str | None = None
    reference_answer: str | None = None
    relevant_doc_ids: list[str] | None = None
    relevant_chunk_ids: list[str] | None = None
    key_points: list[str] | None = None
    expected_behavior: Literal["answer", "refuse"] | None = None
    expected_format: str | dict[str, Any] | None = None
    citations: list[dict[str, Any]] | None = None
    target: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    dataset_id: str
    metric_names: list[str] | None = None
