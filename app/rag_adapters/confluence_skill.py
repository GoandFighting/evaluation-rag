"""Adapter for the local ``confluence-kb-query`` ZCode skill."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, urlparse

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


SearchFunction = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ConfluenceSkillConfig:
    skill_dir: Path
    server: str | None = None
    top_k: int = 5
    alpha: float = 0.5

    @classmethod
    def from_environment(cls) -> "ConfluenceSkillConfig":
        default_dir = (
            Path.home() / ".zcode" / "skills" / "confluence-kb-query"
        )
        return cls(
            skill_dir=Path(os.getenv("CONFLUENCE_SKILL_DIR", default_dir)),
            server=os.getenv("CONFLUENCE_AI_SERVER") or None,
            top_k=int(os.getenv("CONFLUENCE_SKILL_TOP_K", "5")),
            alpha=float(os.getenv("CONFLUENCE_SKILL_ALPHA", "0.5")),
        )


class ConfluenceSkillAdapter:
    """Run the real Skill search and expose its results as canonical chunks.

    The Skill is retrieval-only. Until a generation model is connected, the
    adapter uses the Skill-equivalent formatted search results as ``answer``.
    """

    name = "confluence_skill"
    label = "Confluence KB Skill"
    description = "调用本机 confluence-kb-query Skill，返回真实知识库检索结果。"
    capabilities = AdapterCapabilities(
        answer=True,
        chunks=True,
        citations=True,
    )

    def __init__(
        self,
        config: ConfluenceSkillConfig | None = None,
        *,
        search_function: SearchFunction | None = None,
    ) -> None:
        self.config = config or ConfluenceSkillConfig.from_environment()
        self._search_function = search_function
        self._module: ModuleType | None = None
        self._load_lock = Lock()
        self._validate_options()

    @property
    def script_path(self) -> Path:
        return self.config.skill_dir / "scripts" / "search.py"

    def is_available(self) -> bool:
        return self._search_function is not None or self.script_path.is_file()

    async def healthcheck(self) -> AdapterHealth:
        if not self.is_available():
            return AdapterHealth(
                ok=False,
                message=f"Skill 脚本不存在：{self.script_path}",
            )
        started = time.perf_counter()
        try:
            await asyncio.to_thread(self._search_sync, "knowledge", 1)
        except Exception as exc:
            return AdapterHealth(ok=False, message=str(exc))
        return AdapterHealth(
            ok=True,
            message="Confluence KB Skill 可访问。",
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def invoke(self, request: RAGRequest) -> RAGRunOutput:
        if not self.is_available():
            raise AdapterConfigurationError(
                f"Skill 脚本不存在：{self.script_path}"
            )
        started = time.perf_counter()
        data = await asyncio.to_thread(
            self._search_sync,
            request.query,
            self.config.top_k,
        )
        results = data.get("results", [])
        if not isinstance(results, list):
            raise AdapterResponseError("Skill 响应中的 results 必须是数组。")

        chunks: list[Chunk] = []
        citations: list[dict[str, Any]] = []
        for rank, item in enumerate(results, 1):
            if not isinstance(item, dict):
                raise AdapterResponseError("Skill 的每条检索结果必须是对象。")
            title = str(item.get("title") or f"Confluence 页面 {rank}")
            excerpt = str(item.get("excerpt") or "").strip()
            url = str(item.get("url") or "").strip()
            document_id = self._document_id(item, url, rank)
            chunk_id = f"confluence:{document_id}"
            score = self._score(item.get("score"))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=title,
                    content=excerpt,
                    source=url or None,
                    rank=rank,
                    retrieval_score=score,
                )
            )
            citations.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "document_name": title,
                    "source": url or None,
                    "rank": rank,
                }
            )

        return RAGRunOutput(
            answer=self._format_answer(results),
            chunks=chunks,
            citations=citations,
            latency_ms=round((time.perf_counter() - started) * 1000),
            raw_response=data,
        )

    def _validate_options(self) -> None:
        if self.config.top_k < 1 or self.config.top_k > 50:
            raise AdapterConfigurationError("CONFLUENCE_SKILL_TOP_K 必须在 1 到 50 之间。")
        if not 0.0 <= self.config.alpha <= 1.0:
            raise AdapterConfigurationError("CONFLUENCE_SKILL_ALPHA 必须在 0 到 1 之间。")

    def _load_search_function(self) -> SearchFunction:
        if self._search_function is not None:
            return self._search_function
        with self._load_lock:
            if self._search_function is not None:
                return self._search_function
            if not self.script_path.is_file():
                raise AdapterConfigurationError(
                    f"Skill 脚本不存在：{self.script_path}"
                )
            spec = importlib.util.spec_from_file_location(
                "evaluation_rag_confluence_skill_search",
                self.script_path,
            )
            if spec is None or spec.loader is None:
                raise AdapterConfigurationError("无法加载 Confluence Skill 脚本。")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise AdapterConfigurationError(
                    f"加载 Confluence Skill 失败：{exc}"
                ) from exc
            search = getattr(module, "search", None)
            if not callable(search):
                raise AdapterConfigurationError("Skill 脚本没有可调用的 search 函数。")
            self._module = module
            self._search_function = search
            return search

    def _search_sync(self, query: str, top_k: int) -> dict[str, Any]:
        search = self._load_search_function()
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "alpha": self.config.alpha,
        }
        if self.config.server:
            kwargs["server"] = self.config.server
        try:
            data = search(query, **kwargs)
        except SystemExit as exc:
            raise AdapterExecutionError(
                f"Confluence Skill 执行失败，退出码：{exc.code}"
            ) from exc
        except Exception as exc:
            raise AdapterExecutionError(
                f"Confluence Skill 执行失败：{exc}"
            ) from exc
        if not isinstance(data, dict):
            raise AdapterResponseError("Confluence Skill 必须返回 JSON 对象。")
        return data

    @staticmethod
    def _document_id(item: dict[str, Any], url: str, rank: int) -> str:
        explicit = item.get("page_id") or item.get("pageId") or item.get("id")
        if explicit is not None:
            return str(explicit)
        values = parse_qs(urlparse(url).query).get("pageId", [])
        if values:
            return str(values[0])
        identity = url or str(item.get("title") or f"result-{rank}")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"page-{digest}"

    @staticmethod
    def _score(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise AdapterResponseError(f"无效的检索分数：{value}") from exc

    @staticmethod
    def _format_answer(results: list[dict[str, Any]]) -> str:
        if not results:
            return "未找到相关结果"
        sections = []
        for index, item in enumerate(results, 1):
            title = str(item.get("title") or f"Confluence 页面 {index}")
            excerpt = str(item.get("excerpt") or "").strip()
            url = str(item.get("url") or "").strip()
            source = f"\n来源：{url}" if url else ""
            sections.append(f"{index}. {title}\n{excerpt}{source}".strip())
        return "\n\n".join(sections)
