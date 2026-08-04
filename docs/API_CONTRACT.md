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
)
```

名称一旦发布即视为 API。字段路径使用规范名称，例如 `chunks.content`，导入别名只在 `services/datasets.py` 处理。

## HTTP API

### `POST /api/datasets/upload`

返回 `dataset_id`、样本数量、检测字段以及每个指标的 `eligible_samples`、`field_ready`、`implemented` 和 `runnable`。

### `POST /api/evaluations/run`

请求：

```json
{"dataset_id":"uuid","metric_names":["token_f1"]}
```

响应包含指标汇总 `summary` 和样本明细 `results`。状态值为 `success`、`not_applicable`、`not_implemented` 或 `failed`。

## 兼容性规则

- 新增可选字段、指标或 evidence 字段属于向后兼容变更。
- 删除/重命名字段、改变分数语义或状态值属于破坏性变更。
- 破坏性变更必须提升 `schema_version`，保留一段时间的导入兼容映射，并在 PR 中提供迁移示例。
