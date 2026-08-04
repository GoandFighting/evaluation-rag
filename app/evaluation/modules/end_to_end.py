from __future__ import annotations

import json
import re
from collections import Counter

from app.evaluation.base import MetricOutcome, MetricSpec
from app.schemas import EvaluationCase


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*|[\u3400-\u9fff]")
REFUSAL_MARKERS = ("无法回答", "不能回答", "无权", "权限", "不知道", "未找到", "抱歉")


def _tokens(text: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def answer_relevance(case: EvaluationCase) -> MetricOutcome:
    query = set(_tokens(case.query))
    answer = set(_tokens(case.answer))
    score = len(query & answer) / len(query) if query else 0.0
    return MetricOutcome(
        score=score,
        reason=f"回答覆盖了问题中 {len(query & answer)}/{len(query)} 个唯一词元。",
        evidence={"matched_tokens": sorted(query & answer)},
    )


def exact_match(case: EvaluationCase) -> MetricOutcome:
    actual = " ".join(_tokens(case.answer))
    expected = " ".join(_tokens(case.reference_answer))
    passed = actual == expected
    return MetricOutcome(
        score=1.0 if passed else 0.0,
        passed=passed,
        reason="回答与参考答案规范化后完全一致。" if passed else "回答与参考答案不完全一致。",
    )


def token_f1(case: EvaluationCase) -> MetricOutcome:
    actual = Counter(_tokens(case.answer))
    expected = Counter(_tokens(case.reference_answer))
    overlap = sum((actual & expected).values())
    precision = overlap / sum(actual.values()) if actual else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricOutcome(
        score=score,
        reason=f"词元精确率 {precision:.3f}，召回率 {recall:.3f}。",
        evidence={"precision": precision, "recall": recall},
    )


def key_point_completeness(case: EvaluationCase) -> MetricOutcome:
    answer = (case.answer or "").lower()
    points = case.key_points or []
    matched = [point for point in points if point.lower() in answer]
    score = len(matched) / len(points)
    return MetricOutcome(
        score=score,
        reason=f"回答覆盖了 {len(matched)}/{len(points)} 个关键点。",
        evidence={"matched": matched, "missing": [p for p in points if p not in matched]},
    )


def format_compliance(case: EvaluationCase) -> MetricOutcome:
    expected = case.expected_format
    answer = case.answer or ""
    passed = True
    reason = "回答满足格式要求。"
    if isinstance(expected, str) and expected.lower() == "json":
        try:
            json.loads(answer)
        except json.JSONDecodeError:
            passed = False
            reason = "回答不是合法 JSON。"
    elif isinstance(expected, dict):
        try:
            parsed = json.loads(answer)
            required = expected.get("required_fields", [])
            passed = isinstance(parsed, dict) and all(key in parsed for key in required)
            if not passed:
                reason = "JSON 缺少必需字段。"
        except json.JSONDecodeError:
            passed = False
            reason = "回答不是合法 JSON。"
    else:
        marker = str(expected).lower()
        if marker == "table":
            passed = "|" in answer
        elif marker:
            passed = marker in answer.lower()
        if not passed:
            reason = f"回答未满足格式标记：{expected}。"
    return MetricOutcome(score=1.0 if passed else 0.0, passed=passed, reason=reason)


def refusal_quality(case: EvaluationCase) -> MetricOutcome:
    refused = any(marker in (case.answer or "") for marker in REFUSAL_MARKERS)
    expected_refusal = case.expected_behavior == "refuse"
    passed = refused == expected_refusal
    return MetricOutcome(
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=f"预期{'拒答' if expected_refusal else '正常回答'}，实际{'拒答' if refused else '正常回答'}。",
        evidence={"expected_refusal": expected_refusal, "detected_refusal": refused},
    )


METRICS = [
    MetricSpec(
        "answer_relevance_lexical",
        "回答相关性（词面基线）",
        "end_to_end",
        "问题词元在回答中的覆盖率；用于低成本筛查，不替代语义或 LLM Judge。",
        ("query", "answer"),
        answer_relevance,
    ),
    MetricSpec(
        "exact_match",
        "完全匹配",
        "end_to_end",
        "回答与参考答案规范化后的完全匹配。",
        ("answer", "reference_answer"),
        exact_match,
    ),
    MetricSpec(
        "token_f1",
        "词元 F1",
        "end_to_end",
        "回答与参考答案之间的词元精确率、召回率及 F1。",
        ("answer", "reference_answer"),
        token_f1,
    ),
    MetricSpec(
        "key_point_completeness",
        "关键点完整性",
        "end_to_end",
        "回答覆盖人工标注关键点的比例。",
        ("answer", "key_points"),
        key_point_completeness,
    ),
    MetricSpec(
        "format_compliance",
        "格式合规性",
        "end_to_end",
        "检查 JSON、表格或指定格式标记。",
        ("answer", "expected_format"),
        format_compliance,
    ),
    MetricSpec(
        "refusal_quality",
        "拒答质量",
        "end_to_end",
        "判断回答/拒答行为是否符合测试标签。",
        ("answer", "expected_behavior"),
        refusal_quality,
    ),
]
