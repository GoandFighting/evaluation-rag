"""Contracts shared by adapters that invoke external RAG systems."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.schemas import Chunk


class AdapterCapabilities(BaseModel):
    """Fields that an adapter can populate after invoking its target."""

    answer: bool = True
    chunks: bool = False
    citations: bool = False
    trace: bool = False
    streaming: bool = False


class RAGRequest(BaseModel):
    request_id: str
    sample_id: str
    query: str
    user_id: str
    variables: dict[str, Any] = Field(default_factory=dict)


class RAGRunOutput(BaseModel):
    """Canonical response consumed by the evaluation case assembler."""

    answer: str
    chunks: list[Chunk] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None
    latency_ms: int | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


class AdapterHealth(BaseModel):
    ok: bool
    message: str
    latency_ms: int | None = None


class RAGAdapter(Protocol):
    name: str
    label: str
    description: str
    capabilities: AdapterCapabilities

    def is_available(self) -> bool:
        """Return whether local configuration is sufficient to invoke it."""
        ...

    async def healthcheck(self) -> AdapterHealth:
        """Check both local configuration and the upstream RAG service."""
        ...

    async def invoke(self, request: RAGRequest) -> RAGRunOutput:
        """Invoke the target and normalize its response."""
        ...
