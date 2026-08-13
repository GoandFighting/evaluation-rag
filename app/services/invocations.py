from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from app.rag_adapters.base import AdapterCapabilities, RAGAdapter, RAGRequest
from app.schemas import Chunk, EvaluationCase


def project_cases_for_adapter(
    cases: list[EvaluationCase], capabilities: AdapterCapabilities
) -> list[EvaluationCase]:
    """Project fields an adapter is expected to add for metric selection."""

    projected: list[EvaluationCase] = []
    for case in cases:
        updates: dict[str, Any] = {}
        if capabilities.answer and not case.answer:
            updates["answer"] = "__adapter_answer__"
        if capabilities.chunks and not case.chunks:
            updates["chunks"] = [
                Chunk(
                    chunk_id="__adapter_chunk__",
                    document_id="__adapter_document__",
                    content="__adapter_chunk_content__",
                    rank=1,
                )
            ]
        if capabilities.citations and case.citations is None:
            updates["citations"] = [{"chunk_id": "__adapter_chunk__"}]
        projected.append(case.model_copy(update=updates))
    return projected


async def invoke_cases(
    cases: list[EvaluationCase],
    adapter: RAGAdapter,
    *,
    max_concurrency: int = 4,
    timeout_seconds: float = 45.0,
) -> tuple[list[EvaluationCase], list[dict[str, Any]]]:
    """Invoke one isolated RAG request per case without aborting the batch."""

    semaphore = asyncio.Semaphore(max_concurrency)

    async def invoke_one(
        case: EvaluationCase,
    ) -> tuple[EvaluationCase, dict[str, Any]]:
        cleared = case.model_copy(
            update={"answer": None, "chunks": [], "citations": None}
        )
        if not case.query:
            return cleared, {
                "sample_id": case.sample_id,
                "status": "not_applicable",
                "latency_ms": None,
                "chunk_count": 0,
                "answer": None,
                "chunks": [],
                "citations": [],
                "reason": "缺少 query，未调用目标 RAG。",
            }

        started = time.perf_counter()
        request = RAGRequest(
            request_id=str(uuid.uuid4()),
            sample_id=case.sample_id,
            query=case.query,
            user_id=f"evaluation-{case.sample_id}",
        )
        try:
            async with semaphore:
                output = await asyncio.wait_for(
                    adapter.invoke(request), timeout=timeout_seconds
                )
        except TimeoutError:
            return cleared, {
                "sample_id": case.sample_id,
                "status": "failed",
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "chunk_count": 0,
                "answer": None,
                "chunks": [],
                "citations": [],
                "reason": f"目标 RAG 调用超过 {timeout_seconds:g} 秒。",
            }
        except Exception as exc:
            return cleared, {
                "sample_id": case.sample_id,
                "status": "failed",
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "chunk_count": 0,
                "answer": None,
                "chunks": [],
                "citations": [],
                "reason": f"{type(exc).__name__}: {exc}",
            }

        target = dict(case.target)
        target.update(
            {
                "adapter_name": adapter.name,
                "adapter_label": adapter.label,
            }
        )
        execution = dict(case.execution)
        execution.update(
            {
                "adapter_name": adapter.name,
                "latency_ms": output.latency_ms,
                "trace_id": output.trace_id,
                "token_usage": output.token_usage,
            }
        )
        enriched = case.model_copy(
            update={
                "answer": output.answer,
                "chunks": output.chunks,
                "citations": output.citations,
                "target": target,
                "execution": execution,
            }
        )
        return enriched, {
            "sample_id": case.sample_id,
            "status": "success",
            "latency_ms": output.latency_ms,
            "chunk_count": len(output.chunks),
            "answer": output.answer,
            "chunks": [chunk.model_dump() for chunk in output.chunks],
            "citations": output.citations,
            "trace_id": output.trace_id,
            "token_usage": output.token_usage,
            "reason": "目标 RAG 调用成功。",
        }

    pairs = await asyncio.gather(*(invoke_one(case) for case in cases))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
