"""HTTP adapter for a SmallRAG target service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.rag_adapters.base import (
    AdapterCapabilities,
    AdapterHealth,
    RAGRequest,
    RAGRunOutput,
)
from app.rag_adapters.errors import (
    AdapterConfigurationError,
    AdapterExecutionError,
    AdapterResponseError,
)
from app.schemas import Chunk


@dataclass(frozen=True, slots=True)
class SmallRAGConfig:
    base_url: str = "http://127.0.0.1:8001"
    api_key: str | None = None
    top_k: int = 5
    alpha: float = 0.5
    timeout_seconds: float = 90.0
    verify_ssl: bool = True
    max_context_chars: int | None = None

    @classmethod
    def from_environment(cls) -> "SmallRAGConfig":
        max_context = os.getenv("SMALLRAG_MAX_CONTEXT_CHARS")
        return cls(
            base_url=os.getenv("SMALLRAG_BASE_URL", "http://127.0.0.1:8001"),
            api_key=os.getenv("SMALLRAG_API_KEY") or None,
            top_k=int(os.getenv("SMALLRAG_TOP_K", "5")),
            alpha=float(os.getenv("SMALLRAG_ALPHA", "0.5")),
            timeout_seconds=float(os.getenv("SMALLRAG_TIMEOUT_SECONDS", "90")),
            verify_ssl=os.getenv("SMALLRAG_VERIFY_SSL", "true").lower()
            not in {"0", "false", "no"},
            max_context_chars=int(max_context) if max_context else None,
        )


class SmallRAGAdapter:
    """Invoke SmallRAG and normalize its generated answer and fetched contexts."""

    name = "smallrag"
    label = "SmallRAG"
    description = "调用 SmallRAG HTTP API，返回模型回答及实际进入提示词的完整检索上下文。"
    capabilities = AdapterCapabilities(
        answer=True,
        chunks=True,
        citations=True,
        trace=True,
    )

    def __init__(
        self,
        config: SmallRAGConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config or SmallRAGConfig.from_environment()
        self._transport = transport
        self._validate_config()

    def is_available(self) -> bool:
        return bool(self.config.base_url.strip())

    async def healthcheck(self) -> AdapterHealth:
        if not self.is_available():
            return AdapterHealth(ok=False, message="未配置 SMALLRAG_BASE_URL。")
        started = time.perf_counter()
        try:
            async with self._client() as client:
                response = await client.get("/ready")
                data = self._json_object(response)
        except Exception as exc:
            return AdapterHealth(ok=False, message=str(exc))

        ok = response.is_success and data.get("status") == "ready"
        return AdapterHealth(
            ok=ok,
            message="SmallRAG 已就绪。" if ok else self._health_message(response, data),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def invoke(self, request: RAGRequest) -> RAGRunOutput:
        if not self.is_available():
            raise AdapterConfigurationError("未配置 SMALLRAG_BASE_URL。")

        payload: dict[str, Any] = {
            "query": request.query,
            "top_k": self.config.top_k,
            "alpha": self.config.alpha,
            "include_retrieval": True,
        }
        if self.config.max_context_chars is not None:
            payload["max_context_chars"] = self.config.max_context_chars

        started = time.perf_counter()
        try:
            async with self._client(request.request_id) as client:
                response = await client.post("/v1/query", json=payload)
                response.raise_for_status()
                data = self._json_object(response)
        except httpx.TimeoutException as exc:
            raise AdapterExecutionError("SmallRAG 请求超时。") from exc
        except httpx.HTTPStatusError as exc:
            raise AdapterExecutionError(
                f"SmallRAG 返回 HTTP {exc.response.status_code}。"
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterExecutionError(f"SmallRAG 请求失败：{exc}") from exc

        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise AdapterResponseError("SmallRAG 响应缺少有效的 answer。")

        chunks = self._normalize_contexts(data)
        citations = data.get("citations") or []
        if not isinstance(citations, list) or not all(
            isinstance(item, dict) for item in citations
        ):
            raise AdapterResponseError("SmallRAG 响应中的 citations 必须是对象数组。")

        latency = data.get("latency_ms") or {}
        reported_latency = latency.get("total") if isinstance(latency, dict) else None
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise AdapterResponseError("SmallRAG 响应中的 usage 必须是对象。")

        return RAGRunOutput(
            answer=answer.strip(),
            chunks=chunks,
            citations=self._normalize_citations(citations, chunks),
            trace_id=str(data.get("request_id") or request.request_id),
            latency_ms=self._integer_or_default(
                reported_latency,
                round((time.perf_counter() - started) * 1000),
            ),
            token_usage=usage,
            raw_response=data,
        )

    def _client(self, request_id: str | None = None) -> httpx.AsyncClient:
        headers = {"X-Request-ID": request_id} if request_id else {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=headers,
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_ssl,
            transport=self._transport,
        )

    def _validate_config(self) -> None:
        if not 1 <= self.config.top_k <= 20:
            raise AdapterConfigurationError("SMALLRAG_TOP_K 必须在 1 到 20 之间。")
        if not 0 <= self.config.alpha <= 1:
            raise AdapterConfigurationError("SMALLRAG_ALPHA 必须在 0 到 1 之间。")
        if self.config.timeout_seconds <= 0:
            raise AdapterConfigurationError("SMALLRAG_TIMEOUT_SECONDS 必须大于 0。")
        if (
            self.config.max_context_chars is not None
            and not 1_000 <= self.config.max_context_chars <= 200_000
        ):
            raise AdapterConfigurationError(
                "SMALLRAG_MAX_CONTEXT_CHARS 必须在 1000 到 200000 之间。"
            )

    @classmethod
    def _normalize_contexts(cls, data: dict[str, Any]) -> list[Chunk]:
        raw_contexts = data.get("contexts")
        if raw_contexts is None:
            retrieval = data.get("retrieval") or {}
            raw_contexts = retrieval.get("results", []) if isinstance(retrieval, dict) else []
        if not isinstance(raw_contexts, list):
            raise AdapterResponseError("SmallRAG 响应中的 contexts 必须是数组。")

        chunks: list[Chunk] = []
        for rank, item in enumerate(raw_contexts, 1):
            if not isinstance(item, dict):
                raise AdapterResponseError("SmallRAG 的每条 context 必须是对象。")
            document_id = item.get("document_id") or item.get("page_id") or item.get("id")
            chunk_id = item.get("chunk_id")
            if chunk_id is None and document_id is not None:
                chunk_id = f"smallrag:{document_id}"
            content = item.get("content")
            if content is None:
                content = item.get("excerpt", "")
            if not isinstance(content, str):
                raise AdapterResponseError("SmallRAG context 的 content 必须是字符串。")
            chunks.append(
                Chunk(
                    chunk_id=str(chunk_id) if chunk_id is not None else f"smallrag:rank-{rank}",
                    document_id=str(document_id) if document_id is not None else None,
                    document_name=item.get("document_name") or item.get("title"),
                    content=content,
                    source=item.get("source") or item.get("url"),
                    page=item.get("page"),
                    rank=cls._integer_or_default(item.get("rank"), rank),
                    retrieval_score=cls._optional_float(
                        item.get("retrieval_score", item.get("score"))
                    ),
                    truncated=bool(item.get("truncated", False)),
                )
            )
        return chunks

    @staticmethod
    def _normalize_citations(
        citations: list[dict[str, Any]], chunks: list[Chunk]
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, citation in enumerate(citations):
            item = dict(citation)
            if "chunk_id" not in item and index < len(chunks):
                item["chunk_id"] = chunks[index].chunk_id
            normalized.append(item)
        return normalized

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise AdapterResponseError("SmallRAG 返回的不是有效 JSON。") from exc
        if not isinstance(data, dict):
            raise AdapterResponseError("SmallRAG JSON 根节点必须是对象。")
        return data

    @staticmethod
    def _health_message(response: httpx.Response, data: dict[str, Any]) -> str:
        status = data.get("status") or "unknown"
        return f"SmallRAG 未就绪（HTTP {response.status_code}，status={status}）。"

    @staticmethod
    def _integer_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise AdapterResponseError(f"无效的检索分数：{value}") from exc
