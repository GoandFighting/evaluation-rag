# 协作开发规范

## 模块所有权

| 范围 | 主要负责人 | 目录 |
|---|---|---|
| 前端与端到端回答评测 | 开发者 A | `app/static/`、`app/evaluation/modules/end_to_end.py` |
| 检索与生成评测 | 开发者 B | `app/evaluation/modules/retrieval.py`、`generation.py` |
| 公共契约与执行框架 | 共同维护 | `app/schemas.py`、`evaluation/base.py`、`registry.py`、`engine.py` |

模块负责人可以独立增加指标和测试。公共契约修改必须在 PR 中标记 `contract-change`，由另一位开发者审核后合并。

## 分支策略

`main` 始终保持可运行，不直接提交功能代码。每项工作从最新 `main` 创建短生命周期分支：

```text
feat/ui-<topic>
feat/e2e-<metric>
feat/retrieval-<metric>
feat/generation-<metric>
fix/<topic>
docs/<topic>
```

示例：`feat/e2e-answer-correctness`、`feat/retrieval-mrr`。提交信息使用简短祈使句，例如 `Add MRR evaluator`。一个分支只解决一个主题。

## 日常流程

1. 从 `main` 拉取最新代码并创建功能分支。
2. 只修改负责模块；涉及共享契约时先在 Issue/PR 中说明影响。
3. 增加或更新 `tests/`，本地运行 `pytest`。
4. 提交 Draft PR，说明字段需求、算法、输出证据和测试结果。
5. 至少一人审核；共享契约变更必须由另一模块负责人审核。
6. CI 通过后使用 Squash Merge，删除已合并分支。

## 指标开发规则

- 指标通过 `MetricSpec` 声明名称、维度、说明和 `required_fields`。
- evaluator 只接收 `EvaluationCase`，不得依赖 FastAPI、文件上传或前端状态。
- 缺少字段由引擎返回 `not_applicable`，不得把缺失输入记为 0 分。
- 分数统一在 `[0, 1]`，同时返回可复核的 `reason` 和 `evidence`。
- 修改指标名称或字段语义属于破坏性变更，应同步更新 `schema_version`、文档和示例。

## 避免冲突

- 两人不要在同一 PR 中顺手格式化对方模块。
- `registry.py` 只负责组合模块列表；新增普通指标不应修改它。
- 前端依据 `/api/metrics` 渲染，不硬编码新增指标。
- 大改公共文件前先拆出独立契约 PR，合并后双方再继续功能分支。
