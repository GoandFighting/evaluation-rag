# RAG Evaluation Workbench

一个字段驱动的 RAG 离线评测 MVP。用户上传 JSON/JSONL 后，后端按指标声明的必需字段计算样本覆盖率；前端只启用字段满足且已经实现的指标。

## 模块边界

```text
app/static/                         上传、指标选择、结果展示
app/main.py                         HTTP API
app/services/datasets.py            解析、别名兼容、临时数据集存储
app/evaluation/base.py              MetricSpec 插件契约
app/evaluation/registry.py          三维度统一注册与字段探测
app/evaluation/engine.py            逐样本执行与汇总
app/evaluation/modules/
  end_to_end.py                     端到端回答指标（已实现首批基线）
  retrieval.py                      检索指标契约（待实现）
  generation.py                     生成指标契约（待实现）
```

新增指标只需在对应模块中提供 `MetricSpec`。前端通过 API 读取指标元数据，不需要同步修改指标列表。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

访问 `http://127.0.0.1:8000`，可上传 `examples/sample_dataset.jsonl`。

不要直接双击 `app/static/index.html`：页面样式由 `/static` 路由提供，上传和评测依赖 `/api`，必须通过 Uvicorn 访问。

运行测试：

```powershell
pytest
```

## API

- `POST /api/datasets/upload`：上传并解析数据，返回字段和指标覆盖率。
- `POST /api/evaluations/run`：运行选定指标并返回汇总及样本明细。
- `GET /api/metrics`：获取三大维度的指标契约。

## 并行开发约定

- `app/evaluation/base.py` 和 `app/schemas.py` 是共享契约，修改前需要两位开发者确认。
- 两位开发者分别修改 `modules/end_to_end.py` 与 `modules/retrieval.py`、`modules/generation.py`，这些文件互不依赖。
- 指标通过模块内的 `METRICS` 列表注册，前端不硬编码指标名称。
- `chunks` 可以为空；此时仍可运行只需要 `query + answer` 的端到端指标，但检索和生成维度不可评测。
- JSONL 是“一行一个完整 JSON 对象”，示例对象需要压缩到单行后写入文件。

完整分支策略、模块所有权和接口变更流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，数据与指标契约见 [docs/API_CONTRACT.md](docs/API_CONTRACT.md)。


## 当前边界

- 数据集暂存在进程内，服务重启后需重新上传；后续可替换为数据库或对象存储。
- 评测同步执行，适合首批小数据集；大数据集需要任务队列。
- 回答相关性目前是可解释的词面基线，不代表语义正确性。
- 检索和生成指标已经声明字段契约，但 evaluator 留给相应模块负责人实现。
- LLM Judge、Embedding Provider 及人工校准集尚未接入。
