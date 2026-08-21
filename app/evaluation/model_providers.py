"""OpenAI-compatible HTTP providers for model-backed evaluation metrics."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from app.evaluation.providers import EvaluationContext


class ModelProviderError(RuntimeError):
    """A configured evaluation model could not complete a request."""


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    name: str
    configured: bool
    ok: bool
    model: str | None
    message: str
    latency_ms: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _environment_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class LLMJudgeConfig:
    base_url: str
    model: str = "qwen3-8b"
    api_key: str | None = None
    timeout_seconds: float = 60.0
    verify_ssl: bool = True
    max_tokens: int = 1024
    temperature: float = 0.0
    enable_thinking: bool = False

    @classmethod
    def from_environment(cls) -> "LLMJudgeConfig | None":
        base_url = os.getenv("EVAL_LLM_BASE_URL", "").strip()
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            model=os.getenv("EVAL_LLM_MODEL", "qwen3-8b"),
            api_key=os.getenv("EVAL_LLM_API_KEY") or None,
            timeout_seconds=float(os.getenv("EVAL_LLM_TIMEOUT_SECONDS", "60")),
            verify_ssl=_environment_bool("EVAL_LLM_VERIFY_SSL"),
            max_tokens=int(os.getenv("EVAL_LLM_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("EVAL_LLM_TEMPERATURE", "0")),
            enable_thinking=_environment_bool(
                "EVAL_LLM_ENABLE_THINKING", default=False
            ),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    base_url: str
    model: str = "qwen3-embedding-0.6b"
    api_key: str | None = None
    timeout_seconds: float = 60.0
    verify_ssl: bool = True

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig | None":
        base_url = os.getenv("EVAL_EMBEDDING_BASE_URL", "").strip()
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            model=os.getenv(
                "EVAL_EMBEDDING_MODEL", "qwen3-embedding-0.6b"
            ),
            api_key=os.getenv("EVAL_EMBEDDING_API_KEY") or None,
            timeout_seconds=float(
                os.getenv("EVAL_EMBEDDING_TIMEOUT_SECONDS", "60")
            ),
            verify_ssl=_environment_bool("EVAL_EMBEDDING_VERIFY_SSL"),
        )


class _OpenAICompatibleProvider:
    provider_name: str

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        verify_ssl: bool,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("模型服务 base_url 不能为空。")
        if not model.strip():
            raise ValueError("模型名称不能为空。")
        if timeout_seconds <= 0:
            raise ValueError("模型请求超时时间必须大于 0。")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
            verify=self.verify_ssl,
            transport=self._transport,
        )

    async def healthcheck(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            async with self._client() as client:
                health_response = await client.get("/health")
                health_response.raise_for_status()
                models_response = await client.get("/v1/models")
                models_response.raise_for_status()
                models = self._json_object(models_response)
            model_ids = {
                str(item.get("id"))
                for item in models.get("data", [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            if self.model not in model_ids:
                raise ModelProviderError(
                    f"/v1/models 未返回已配置模型 {self.model}。"
                )
        except Exception as exc:
            return ProviderHealth(
                name=self.provider_name,
                configured=True,
                ok=False,
                model=self.model,
                message=str(exc),
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        return ProviderHealth(
            name=self.provider_name,
            configured=True,
            ok=True,
            model=self.model,
            message="模型服务可访问，且模型已加载。",
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def _post_json(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return self._json_object(response)
        except httpx.TimeoutException as exc:
            raise ModelProviderError(
                f"{self.provider_name} 请求超时。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._error_detail(exc.response)
            raise ModelProviderError(
                f"{self.provider_name} 返回 HTTP "
                f"{exc.response.status_code}：{detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelProviderError(
                f"{self.provider_name} 请求失败：{exc}"
            ) from exc

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelProviderError("模型服务返回的不是有效 JSON。") from exc
        if not isinstance(data, dict):
            raise ModelProviderError("模型服务 JSON 根节点必须是对象。")
        return data

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return "响应正文不是有效 JSON"
        if not isinstance(data, dict):
            return "响应正文格式无效"
        error = data.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("code")
        else:
            detail = data.get("detail") or data.get("message")
        return str(detail or "未知错误")[:600]


class OpenAICompatibleLLMJudge(_OpenAICompatibleProvider):
    provider_name = "llm_judge"

    def __init__(
        self,
        config: LLMJudgeConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if config.max_tokens < 1:
            raise ValueError("EVAL_LLM_MAX_TOKENS 必须大于 0。")
        if not 0 <= config.temperature <= 2:
            raise ValueError("EVAL_LLM_TEMPERATURE 必须在 0 到 2 之间。")
        super().__init__(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            verify_ssl=config.verify_ssl,
            transport=transport,
        )
        self.config = config

    async def _complete(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, str | None]:
        if not prompt.strip():
            raise ValueError("Judge prompt 不能为空。")
        system_prompt = kwargs.pop("system_prompt", None)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        user_prompt = prompt.strip()
        if not self.config.enable_thinking and not user_prompt.endswith("/no_think"):
            user_prompt = f"{user_prompt}\n\n/no_think"
        messages.append({"role": "user", "content": user_prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking
            },
        }
        payload.update(kwargs)
        data = await self._post_json("/v1/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError("LLM Judge 响应缺少 choices。")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError("LLM Judge 响应缺少有效的 message.content。")
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
        return content.strip(), (
            str(finish_reason) if finish_reason is not None else None
        )

    async def judge(self, prompt: str, **kwargs: Any) -> str:
        content, _ = await self._complete(prompt, **kwargs)
        return content

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for position, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(content[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    async def judge_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        content, finish_reason = await self._complete(prompt, **kwargs)
        parsed = self._extract_json_object(content)
        if parsed is not None:
            return parsed

        retry_prompt = (
            f"{prompt.rstrip()}\n\n"
            "上一次响应不是合法 JSON。请重新完成同一评分任务，只输出一个完整的 "
            "JSON 对象，不要输出思考过程、Markdown 或其他文字。"
        )
        retry_content, retry_finish_reason = await self._complete(
            retry_prompt, **kwargs
        )
        parsed = self._extract_json_object(retry_content)
        if parsed is not None:
            return parsed
        raise ModelProviderError(
            "LLM Judge 连续两次未返回有效的 JSON 对象"
            f"（finish_reason: {finish_reason or 'unknown'} -> "
            f"{retry_finish_reason or 'unknown'}）。"
        )

    async def probe(self) -> dict[str, Any]:
        output = await self.judge(
            "请只输出 OK，不要输出其他内容。 /no_think",
            max_tokens=128,
            temperature=0,
        )
        return {
            "name": self.provider_name,
            "ok": True,
            "model": self.model,
            "endpoint": "/v1/chat/completions",
            "output_preview": output[:200],
        }


class OpenAICompatibleEmbeddingProvider(_OpenAICompatibleProvider):
    provider_name = "embedding"

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            verify_ssl=config.verify_ssl,
            transport=transport,
        )
        self.config = config

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding 输入必须是非空字符串列表。")
        data = await self._post_json(
            "/v1/embeddings", {"model": self.model, "input": texts}
        )
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise ModelProviderError("Embedding 响应数量与输入数量不一致。")
        ordered: list[list[float] | None] = [None] * len(texts)
        for fallback_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ModelProviderError("Embedding data 必须是对象数组。")
            index = item.get("index", fallback_index)
            vector = item.get("embedding")
            if not isinstance(index, int) or not 0 <= index < len(texts):
                raise ModelProviderError("Embedding 响应包含无效 index。")
            if ordered[index] is not None:
                raise ModelProviderError("Embedding 响应包含重复 index。")
            if not isinstance(vector, list) or not vector:
                raise ModelProviderError("Embedding 响应缺少有效向量。")
            try:
                ordered[index] = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise ModelProviderError("Embedding 向量包含非数值元素。") from exc
        if any(vector is None for vector in ordered):
            raise ModelProviderError("Embedding 响应缺少部分输入的向量。")
        return [vector for vector in ordered if vector is not None]

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b:
            raise ValueError("余弦相似度要求两个非空向量。")
        if len(vec_a) != len(vec_b):
            raise ValueError("余弦相似度要求两个向量维度一致。")
        norm_a = math.sqrt(sum(value * value for value in vec_a))
        norm_b = math.sqrt(sum(value * value for value in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        score = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        return max(-1.0, min(1.0, score / (norm_a * norm_b)))

    async def probe(self) -> dict[str, Any]:
        vectors = await self.embed(["evaluation-rag 模型连接测试"])
        return {
            "name": self.provider_name,
            "ok": True,
            "model": self.model,
            "endpoint": "/v1/embeddings",
            "vector_dimension": len(vectors[0]),
        }


def build_evaluation_context_from_environment() -> EvaluationContext:
    llm_config = LLMJudgeConfig.from_environment()
    embedding_config = EmbeddingConfig.from_environment()
    return EvaluationContext(
        embedding_provider=(
            OpenAICompatibleEmbeddingProvider(embedding_config)
            if embedding_config is not None
            else None
        ),
        llm_judge=(
            OpenAICompatibleLLMJudge(llm_config)
            if llm_config is not None
            else None
        ),
        max_concurrency=int(os.getenv("EVAL_MODEL_MAX_CONCURRENCY", "4")),
        timeout_seconds=float(
            os.getenv("EVAL_METRIC_TIMEOUT_SECONDS", "60")
        ),
    )


def describe_model_providers(
    context: EvaluationContext,
) -> list[dict[str, Any]]:
    descriptions = []
    for name, provider in (
        ("llm_judge", context.llm_judge),
        ("embedding", context.embedding_provider),
    ):
        descriptions.append(
            {
                "name": name,
                "configured": provider is not None,
                "model": getattr(provider, "model", None),
            }
        )
    return descriptions
