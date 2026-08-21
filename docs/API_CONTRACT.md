# 数据与指标接口契约

## 数据集记录

JSONL 每行是一条完整 JSON 对象。推荐字段：

- 标识：`schema_version`、`sample_id`、`question_id`
- 端到端：`query`、`answer`
- 检索/生成：`chunks[].chunk_id`、`content`、`document_id`、`source`、`rank`
- 标注：`ground_truth.reference_answer`、`relevant_doc_ids`、`relevant_chunk_ids`、`key_points`、`expected_behavior`
- 运行信息：`target`、`execution`、`tags`

`chunks: []` 合法。此时端到端指标仍可运行，但检索与生成指标必须标记为不可评测。

## 指标插件

每个指标使用 `MetricSpec`：

```python
MetricSpec(
    name="stable_machine_name",
    label="前端显示名称",
    dimension="end_to_end | retrieval | generation",
    description="指标含义和限制",
    required_fields=("query", "answer"),
    evaluator=evaluate,
    any_of_fields=("relevant_chunk_ids", "relevant_doc_ids"),  # 可选：至少满足其一
)
```

模型指标使用异步 evaluator，并声明运行能力：

```python
MetricSpec(
    name="semantic_similarity",
    label="语义相似度",
    dimension="end_to_end",
    description="回答与参考答案的语义相似度。",
    required_fields=("answer", "reference_answer"),
    async_evaluator=evaluate_semantic_similarity,
    required_capabilities=("embedding",),
)
```

异步 evaluator 接收 `(EvaluationCase, EvaluationContext)`。Context 提供
`embedding_provider`、`llm_judge`、并发上限和单指标超时；普通规则指标继续使用
只接收 `EvaluationCase` 的同步 `evaluator`。

名称一旦发布即视为 API。字段路径使用规范名称，例如 `chunks.content`，导入别名只在 `services/datasets.py` 处理。
指标名称在三个维度中全局唯一；Registry 会拒绝重复名称。一个指标只能声明同步
`evaluator` 或异步 `async_evaluator` 中的一种。

`any_of_fields` 为可选字段，声明后指标在 `required_fields` 全部满足且 `any_of_fields` 中至少一个字段存在时才可运行。用于支持同一指标兼容多种粒度的标注（如片段级 `relevant_chunk_ids` 与文档级 `relevant_doc_ids`）。

## HTTP API

### `POST /api/datasets/upload`

返回 `dataset_id`、样本数量、检测字段以及每个指标的 `eligible_samples`、`field_ready`、`implemented`、`configured`、`required_capabilities`、`missing_capabilities`、`any_of_fields` 和 `runnable`。

### `POST /api/evaluations/run`

请求：

```json
{"dataset_id":"uuid","metric_names":["token_f1"]}
```

响应包含指标汇总 `summary`、模块综合分 `module_scores` 和样本明细 `results`。端到端、检索和生成综合分都只使用 `success_count > 0` 且存在 `average` 的指标，并在本次成功指标之间重新归一化各维度默认权重。状态值为：

- `success`：指标成功执行；
- `not_applicable`：样本缺少指标字段；
- `not_configured`：指标已实现，但缺少 Embedding 或 Judge Provider；
- `not_implemented`：指标仅声明契约，尚无 evaluator；
- `failed`：指标执行异常或超时。

## 兼容性规则

- 新增可选字段、指标或 evidence 字段属于向后兼容变更。
- 删除/重命名字段、改变分数语义或状态值属于破坏性变更。
- 破坏性变更必须提升 `schema_version`，保留一段时间的导入兼容映射，并在 PR 中提供迁移示例。
