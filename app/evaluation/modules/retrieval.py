from __future__ import annotations

import re

from app.evaluation.base import MetricOutcome, MetricSpec
from app.schemas import EvaluationCase


# --- shared lexical helpers (kept module-local for independence) -----------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*|[\u3400-\u9fff]")
DEFAULT_K = 5


def _tokens(text: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


# --- relevance resolution -------------------------------------------------------

def _resolve_relevant_ids(case: EvaluationCase) -> tuple[set[str], str]:
    """Return (relevant_id_set, level).

    *level* is ``"chunk"`` when ``relevant_chunk_ids`` is present, ``"doc"``
    when only ``relevant_doc_ids`` is available, or ``"none"``.
    """
    if case.relevant_chunk_ids:
        return set(case.relevant_chunk_ids), "chunk"
    if case.relevant_doc_ids:
        return set(case.relevant_doc_ids), "doc"
    return set(), "none"


def _retrieved_ids(case: EvaluationCase, level: str, k: int) -> list[str]:
    """Top-K retrieved identifiers at the given relevance level."""
    ids: list[str] = []
    for chunk in case.chunks[:k]:
        rid = chunk.chunk_id if level == "chunk" else chunk.document_id
        if rid is not None:
            ids.append(str(rid))
    return ids


def _k_eff(case: EvaluationCase, k: int) -> int:
    return min(k, len(case.chunks))


def _log2(n: float) -> float:
    import math

    return math.log2(n) if n > 0 else 0.0


# --- pure-code evaluators -------------------------------------------------------

def hit_rate_at_k(case: EvaluationCase) -> MetricOutcome:
    relevant_set, level = _resolve_relevant_ids(case)
    k = _k_eff(case, DEFAULT_K)
    retrieved = _retrieved_ids(case, level, k)
    hit = any(rid in relevant_set for rid in retrieved)
    return MetricOutcome(
        score=1.0 if hit else 0.0,
        passed=hit,
        reason=f"Top-{k} {'命中' if hit else '未命中'}相关{level}级标注。",
        evidence={
            "k": DEFAULT_K,
            "k_effective": k,
            "relevance_level": level,
            "hit": hit,
            "relevant_count": len(relevant_set),
            "retrieved_count": len(retrieved),
        },
    )


def recall_at_k(case: EvaluationCase) -> MetricOutcome:
    relevant_set, level = _resolve_relevant_ids(case)
    k = _k_eff(case, DEFAULT_K)
    retrieved = _retrieved_ids(case, level, k)
    found = relevant_set & set(retrieved)
    score = len(found) / len(relevant_set) if relevant_set else 0.0
    return MetricOutcome(
        score=score,
        reason=f"Top-{k} 召回了 {len(found)}/{len(relevant_set)} 个相关{level}级标注。",
        evidence={
            "k": DEFAULT_K,
            "k_effective": k,
            "relevance_level": level,
            "found": sorted(found),
            "missing": sorted(relevant_set - found),
            "relevant_count": len(relevant_set),
        },
    )


def precision_at_k(case: EvaluationCase) -> MetricOutcome:
    relevant_set, level = _resolve_relevant_ids(case)
    k = _k_eff(case, DEFAULT_K)
    retrieved = _retrieved_ids(case, level, k)
    relevant_hits = sum(1 for rid in retrieved if rid in relevant_set)
    score = relevant_hits / k if k else 0.0
    return MetricOutcome(
        score=score,
        reason=f"Top-{k} 中 {relevant_hits} 个相关，精确率 {score:.3f}。",
        evidence={
            "k": DEFAULT_K,
            "k_effective": k,
            "relevance_level": level,
            "relevant_hits": relevant_hits,
            "retrieved_count": len(retrieved),
        },
    )


def mrr(case: EvaluationCase) -> MetricOutcome:
    relevant_set, level = _resolve_relevant_ids(case)
    k = _k_eff(case, DEFAULT_K)
    first_rank: int | None = None
    for i, chunk in enumerate(case.chunks[:k]):
        rid = chunk.chunk_id if level == "chunk" else chunk.document_id
        if rid is not None and str(rid) in relevant_set:
            first_rank = chunk.rank if chunk.rank is not None else i + 1
            break
    score = 1.0 / first_rank if first_rank else 0.0
    return MetricOutcome(
        score=score,
        reason=f"首个相关结果排名为 {first_rank}，倒数排名 {score:.3f}。" if first_rank else "Top-K 未命中相关结果。",
        evidence={
            "k": DEFAULT_K,
            "k_effective": k,
            "relevance_level": level,
            "first_relevant_rank": first_rank,
        },
    )


def ndcg_at_k(case: EvaluationCase) -> MetricOutcome:
    relevant_set, level = _resolve_relevant_ids(case)
    k = _k_eff(case, DEFAULT_K)
    retrieved = _retrieved_ids(case, level, k)

    # DCG@K with binary relevance.
    dcg = sum(
        (1.0 if rid in relevant_set else 0.0) / _log2(i + 2)
        for i, rid in enumerate(retrieved)
    )
    # IDCG@K: ideal ranking places all relevant items first.
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / _log2(i + 2) for i in range(ideal_hits))
    score = dcg / idcg if idcg else 0.0
    return MetricOutcome(
        score=score,
        reason=f"DCG@{k}={dcg:.3f}，IDCG@{k}={idcg:.3f}，NDCG={score:.3f}。",
        evidence={
            "k": DEFAULT_K,
            "k_effective": k,
            "relevance_level": level,
            "dcg": dcg,
            "idcg": idcg,
        },
    )


def context_relevance(case: EvaluationCase) -> MetricOutcome:
    query_tokens = set(_tokens(case.query))
    context = " ".join(chunk.content for chunk in case.chunks)
    context_tokens = set(_tokens(context))
    if not query_tokens:
        return MetricOutcome(score=0.0, reason="问题无有效词元。")
    overlap = query_tokens & context_tokens
    score = len(overlap) / len(query_tokens)
    return MetricOutcome(
        score=score,
        reason=f"问题词元在检索上下文中的覆盖率为 {score:.3f}（{len(overlap)}/{len(query_tokens)}）。",
        evidence={
            "matched_tokens": sorted(overlap),
            "missing_tokens": sorted(query_tokens - context_tokens),
            "query_token_count": len(query_tokens),
            "level": "lexical",
        },
    )


# --- metric registry ------------------------------------------------------------

# ``any_of_fields`` lets a metric run when *either* chunk-level or doc-level
# relevance labels are present, so both golden-set formats are supported.
_RELEVANCE_ANY_OF: tuple[str, ...] = ("relevant_chunk_ids", "relevant_doc_ids")

METRICS = [
    MetricSpec(
        "hit_rate_at_k",
        "Hit Rate@K",
        "retrieval",
        f"Top-{DEFAULT_K} 是否命中相关片段或文档（二值）。",
        ("chunks.chunk_id",),
        any_of_fields=_RELEVANCE_ANY_OF,
        evaluator=hit_rate_at_k,
    ),
    MetricSpec(
        "recall_at_k",
        "Recall@K",
        "retrieval",
        f"Top-{DEFAULT_K} 召回的相关片段或文档占全部相关标注的比例。",
        ("chunks.chunk_id",),
        any_of_fields=_RELEVANCE_ANY_OF,
        evaluator=recall_at_k,
    ),
    MetricSpec(
        "precision_at_k",
        "Precision@K",
        "retrieval",
        f"Top-{DEFAULT_K} 中相关片段或文档的比例。",
        ("chunks.chunk_id",),
        any_of_fields=_RELEVANCE_ANY_OF,
        evaluator=precision_at_k,
    ),
    MetricSpec(
        "mrr",
        "MRR",
        "retrieval",
        "首个相关结果排名的倒数（Mean Reciprocal Rank）。",
        ("chunks.chunk_id", "chunks.rank"),
        any_of_fields=_RELEVANCE_ANY_OF,
        evaluator=mrr,
    ),
    MetricSpec(
        "ndcg_at_k",
        "NDCG@K",
        "retrieval",
        f"归一化折损累积增益，奖励相关结果排在更前位置（Top-{DEFAULT_K}，二值相关性）。",
        ("chunks.chunk_id", "chunks.rank"),
        any_of_fields=_RELEVANCE_ANY_OF,
        evaluator=ndcg_at_k,
    ),
    MetricSpec(
        "context_relevance",
        "上下文相关性",
        "retrieval",
        "问题词元在检索上下文中的覆盖率；词面基线，不替代语义判断。",
        ("query", "chunks.content"),
        evaluator=context_relevance,
    ),
    # --- embedding-based (contract declared, evaluator pending provider deployment) ---
    MetricSpec(
        "context_relevance_semantic",
        "上下文相关性（语义）",
        "retrieval",
        "问题与检索上下文的嵌入余弦相似度；需部署 EmbeddingProvider 后实现。",
        ("query", "chunks.content"),
    ),
    # --- LLM-as-judge (contract declared, evaluator pending provider deployment) ---
    MetricSpec(
        "context_precision",
        "上下文精确率（LLM）",
        "retrieval",
        "LLM 逐个判断检索片段是否与问题相关，加权累积精确率（RAGAS 风格）；需部署 LLMJudge 后实现。",
        ("query", "chunks.content"),
    ),
]
