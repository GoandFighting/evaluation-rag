from __future__ import annotations

import re
from collections import Counter

from app.evaluation.base import MetricOutcome, MetricSpec
from app.schemas import EvaluationCase


# --- shared lexical helpers (kept module-local for independence) -----------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*|[\u3400-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？\n]+|(?<=\.)\s+")
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

# Thresholds for lexical baselines.
_FAITHFULNESS_THRESHOLD = 0.5
_UTILIZATION_THRESHOLD = 0.2


def _tokens(text: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _split_sentences(text: str | None) -> list[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def _token_overlap(set_a: set[str], set_b: set[str]) -> float:
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


# --- pure-code evaluators -------------------------------------------------------

def faithfulness(case: EvaluationCase) -> MetricOutcome:
    """Lexical baseline: sentence-level token overlap of answer vs context.

    Each answer sentence is treated as a claim.  A claim is "supported" when
    its token overlap with the concatenated context exceeds a threshold.
    """
    sentences = _split_sentences(case.answer)
    context = " ".join(chunk.content for chunk in case.chunks)
    context_tokens = set(_tokens(context))

    if not sentences:
        return MetricOutcome(score=0.0, reason="回答无可分析的句子。")
    if not context_tokens:
        return MetricOutcome(score=0.0, reason="上下文为空，无法判定支持性。")

    supported: list[str] = []
    unsupported: list[str] = []
    for sentence in sentences:
        sent_tokens = set(_tokens(sentence))
        if not sent_tokens:
            continue
        overlap = _token_overlap(sent_tokens, context_tokens)
        if overlap >= _FAITHFULNESS_THRESHOLD:
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    total = len(supported) + len(unsupported)
    score = len(supported) / total if total else 0.0
    return MetricOutcome(
        score=score,
        reason=f"{len(supported)}/{total} 个陈述受到上下文支持（词面重叠 ≥ {_FAITHFULNESS_THRESHOLD}）。",
        evidence={
            "supported_claims": len(supported),
            "unsupported_claims": len(unsupported),
            "total_claims": total,
            "threshold": _FAITHFULNESS_THRESHOLD,
            "level": "lexical",
            "unsupported_samples": unsupported[:5],
        },
    )


def context_utilization(case: EvaluationCase) -> MetricOutcome:
    """Lexical baseline: per-chunk token overlap with the answer.

    A chunk is "utilized" when enough of its tokens appear in the answer.
    """
    answer_tokens = set(_tokens(case.answer))
    chunks = case.chunks
    if not chunks:
        return MetricOutcome(score=0.0, reason="无检索片段，无法计算利用率。")
    if not answer_tokens:
        return MetricOutcome(score=0.0, reason="回答无有效词元。")

    utilized = 0
    per_chunk: list[dict[str, object]] = []
    for chunk in chunks:
        chunk_tokens = set(_tokens(chunk.content))
        overlap = _token_overlap(chunk_tokens, answer_tokens)
        used = overlap >= _UTILIZATION_THRESHOLD
        if used:
            utilized += 1
        per_chunk.append({"chunk_id": chunk.chunk_id, "overlap": round(overlap, 3), "utilized": used})

    score = utilized / len(chunks)
    return MetricOutcome(
        score=score,
        reason=f"{utilized}/{len(chunks)} 个检索片段被回答利用（重叠 ≥ {_UTILIZATION_THRESHOLD}）。",
        evidence={
            "utilized_chunks": utilized,
            "total_chunks": len(chunks),
            "threshold": _UTILIZATION_THRESHOLD,
            "level": "lexical",
            "per_chunk": per_chunk,
        },
    )


def citation_correctness(case: EvaluationCase) -> MetricOutcome:
    """Structural validation: cited chunk references resolve to real chunks."""
    chunk_ids = {str(c.chunk_id) for c in case.chunks if c.chunk_id is not None}
    citations = case.citations or []

    # Collect all citation references.
    refs: list[dict[str, object]] = []

    # Inline markers [1], [2] — 1-indexed positions into chunks list.
    inline_markers = [int(m) for m in _CITATION_MARKER_RE.findall(case.answer or "")]
    for marker in inline_markers:
        valid = 1 <= marker <= len(case.chunks)
        refs.append({"type": "inline", "ref": marker, "valid": valid})

    # Citation list entries — each should reference a chunk_id.
    for cite in citations:
        if isinstance(cite, dict):
            cid = cite.get("chunk_id") or cite.get("chunkId")
            valid = str(cid) in chunk_ids if cid is not None else False
            refs.append({"type": "list", "ref": cid, "valid": valid})

    if not refs:
        return MetricOutcome(
            score=1.0,
            passed=True,
            reason="无引用可验证，正确率记为 1.0。",
            evidence={"total_citations": 0, "valid_citations": 0},
        )

    valid_count = sum(1 for r in refs if r["valid"])
    score = valid_count / len(refs)
    return MetricOutcome(
        score=score,
        passed=score == 1.0,
        reason=f"{valid_count}/{len(refs)} 个引用指向有效检索片段。",
        evidence={
            "total_citations": len(refs),
            "valid_citations": valid_count,
            "inline_markers": inline_markers,
            "details": refs,
        },
    )


def citation_completeness(case: EvaluationCase) -> MetricOutcome:
    """Coverage: fraction of answer sentences that carry at least one citation."""
    sentences = _split_sentences(case.answer)
    citations = case.citations or []

    if not sentences:
        return MetricOutcome(score=0.0, reason="回答无可分析的句子。")

    # Count sentences with inline [n] markers.
    cited = sum(
        1 for s in sentences if _CITATION_MARKER_RE.search(s)
    )

    # If no inline markers but a citation list exists, approximate coverage
    # by the ratio of citation entries to sentences.
    if cited == 0 and citations:
        cited = min(len(citations), len(sentences))

    score = cited / len(sentences)
    return MetricOutcome(
        score=score,
        reason=f"{cited}/{len(sentences)} 个句子附带引用。",
        evidence={
            "cited_sentences": cited,
            "total_sentences": len(sentences),
            "citation_count": len(citations),
        },
    )


def factual_consistency(case: EvaluationCase) -> MetricOutcome:
    """Lexical baseline: overall token-F1 alignment between answer and context."""
    answer_tokens = Counter(_tokens(case.answer))
    context = " ".join(chunk.content for chunk in case.chunks)
    context_tokens = Counter(_tokens(context))

    overlap = sum((answer_tokens & context_tokens).values())
    precision = overlap / sum(answer_tokens.values()) if answer_tokens else 0.0
    recall = overlap / sum(context_tokens.values()) if context_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return MetricOutcome(
        score=f1,
        reason=f"回答与上下文的词元 F1 为 {f1:.3f}（精确率 {precision:.3f}，召回率 {recall:.3f}）。",
        evidence={
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "level": "lexical",
        },
    )


# --- metric registry ------------------------------------------------------------

METRICS = [
    MetricSpec(
        "faithfulness",
        "忠实度",
        "generation",
        "回答中的陈述是否受到上下文支持（词面基线，后续接入 LLM Judge 做语义判定）。",
        ("answer", "chunks.content"),
        evaluator=faithfulness,
    ),
    MetricSpec(
        "context_utilization",
        "上下文利用率",
        "generation",
        "回答是否使用了检索上下文中的关键证据（词面基线）。",
        ("answer", "chunks.content"),
        evaluator=context_utilization,
    ),
    MetricSpec(
        "citation_correctness",
        "引用正确性",
        "generation",
        "引用标记或引用列表是否指向有效的检索片段（结构校验）。",
        ("answer", "citations", "chunks.chunk_id"),
        evaluator=citation_correctness,
    ),
    MetricSpec(
        "citation_completeness",
        "引用完整性",
        "generation",
        "回答中附带引用的句子比例。",
        ("answer", "citations"),
        evaluator=citation_completeness,
    ),
    MetricSpec(
        "factual_consistency",
        "事实一致性",
        "generation",
        "回答与上下文的整体词面对齐度（F1 基线，后续接入 LLM 做冲突检测）。",
        ("answer", "chunks.content"),
        evaluator=factual_consistency,
    ),
    # --- embedding-based (contract declared, evaluator pending provider deployment) ---
    MetricSpec(
        "faithfulness_semantic",
        "忠实度（语义）",
        "generation",
        "嵌入级声明支持检测：逐句计算回答与上下文的语义相似度；需部署 EmbeddingProvider 后实现。",
        ("answer", "chunks.content"),
        required_capabilities=("embedding",),
    ),
    # --- LLM-as-judge (contract declared, evaluator pending provider deployment) ---
    MetricSpec(
        "faithfulness_llm",
        "忠实度（LLM）",
        "generation",
        "LLM 逐句判断回答陈述是否受上下文支持（RAGAS/DeepEval 标准实现）；需部署 LLMJudge 后实现。",
        ("answer", "chunks.content"),
        required_capabilities=("llm_judge",),
    ),
    MetricSpec(
        "factual_correctness",
        "事实正确性（LLM）",
        "generation",
        "LLM 分解声明并做 NLI 判定 TP/FP/FN，对比参考答案；需部署 LLMJudge 后实现。",
        ("answer", "reference_answer"),
        required_capabilities=("llm_judge",),
    ),
]
