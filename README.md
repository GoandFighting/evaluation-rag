# RAG Evaluation Workbench

一个字段驱动的 RAG 评测 MVP。用户选择或拖入 JSON/JSONL 后，前端自动上传并检查字段；点击开始评测会默认提交全部已实现指标，字段不足的样本返回 `not_applicable`。

## 模块边界

```text
app/static/                         上传、指标选择、结果展示
app/main.py                         HTTP API
app/services/datasets.py            解析、别名兼容、临时数据集存储
app/services/invocations.py         批量调用目标 RAG、超时与单样本失败隔离
app/rag_adapters/                   外部 RAG 统一契约、注册表与具体 Adapter
app/evaluation/base.py              MetricSpec 插件契约
app/evaluation/registry.py          三维度统一注册与字段探测
app/evaluation/engine.py            同步/异步指标调度、并发限制与汇总
app/evaluation/modules/
  end_to_end.py                     端到端回答指标（已实现首批基线）
  retrieval.py                      检索指标（已实现纯编码基线 + 嵌入/LLM 契约预留）
  generation.py                     生成指标（已实现纯编码基线 + 嵌入/LLM 契约预留）
app/evaluation/providers.py          Provider 协议、EvaluationContext 与执行限制
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
- `GET /api/rag-adapters`：获取可选 RAG 目标及其输出能力。
- `POST /api/rag-adapters/{name}/healthcheck`：检查 Adapter 与上游服务连通性。
- `GET /api/model-providers`：查看 LLM Judge 与 Embedding Provider 配置状态。
- `POST /api/model-providers/{name}/healthcheck`：检查模型健康状态及模型列表，`name` 为 `llm_judge` 或 `embedding`。
- `POST /api/model-providers/{name}/probe`：实际调用一次模型核心接口，确认推理链路可用。

## 评测模型 Provider

系统通过 OpenAI 兼容 HTTP 接口连接 Qwen3-8B Judge 和 Qwen3 Embedding。Provider 只有在配置对应 `BASE_URL` 时才会注入 `EvaluationContext`；未配置时，依赖该能力的指标保持 `not_configured`。

同机部署可设置：

```bash
export EVAL_LLM_BASE_URL=http://127.0.0.1:8000
export EVAL_LLM_MODEL=qwen3-8b
export EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:8080
export EVAL_EMBEDDING_MODEL=qwen3-embedding-0.6b
```

启动评测系统后验证项目进程到模型服务的连接：

```bash
curl -X POST http://127.0.0.1:8002/api/model-providers/llm_judge/healthcheck
curl -X POST http://127.0.0.1:8002/api/model-providers/embedding/healthcheck
curl -X POST http://127.0.0.1:8002/api/model-providers/llm_judge/probe
curl -X POST http://127.0.0.1:8002/api/model-providers/embedding/probe
```

健康检查会依次访问模型服务的 `/health` 和 `/v1/models`，并确认配置的模型名已经加载；探针会实际请求 `/v1/chat/completions` 或 `/v1/embeddings`。具体超时、TLS、API Key、Judge 最大输出长度和评测并发参数见 `.env.example`。Qwen3 Judge 默认通过 `EVAL_LLM_ENABLE_THINKING=false` 关闭思考模式；结构化响应解析失败时自动重试一次。

端到端模块现已提供两类模型指标：Embedding 用于 `answer_relevance_semantic` 和 `semantic_similarity`；LLM Judge 用于 `answer_relevance_llm`、`answer_correctness`、`completeness_llm` 和 `refusal_quality_llm`。规则版相关性、词元 F1、关键点完整性、格式合规和拒答判断继续保留，便于对照和低成本筛查。评测响应还会返回 `module_scores`；端到端综合分只纳入成功产出分数的指标，并对这些指标的默认权重重新归一化。

前端全选时，每条字段齐全的样本最多触发 4 次 Judge 和 2 次 Embedding 请求。首次真实数据测试建议设置 `EVAL_MODEL_MAX_CONCURRENCY=2`，观察模型吞吐和超时后再逐步提高。

## Confluence KB Skill Adapter

前端选择 `Confluence KB Skill` 后，上传的黄金集只需包含 `query` 和相应人工标注字段（如 `reference_answer`）。运行评测时，系统逐条调用本机 `confluence-kb-query` Skill，并将检索结果标准化为 `answer`、`chunks` 和 `citations`。默认 Skill 目录为 `%USERPROFILE%\.zcode\skills\confluence-kb-query`。

可按需配置：

```powershell
$env:CONFLUENCE_SKILL_DIR="C:\path\to\confluence-kb-query"
$env:CONFLUENCE_AI_SERVER="https://internal-search.example"
$env:CONFLUENCE_SKILL_TOP_K="5"
$env:CONFLUENCE_SKILL_ALPHA="0.5"
```

该 Skill 当前只提供检索，不提供独立的 LLM 生成接口。因此 Adapter 的 `answer` 是检索结果的确定性 Markdown 整理；`chunks` 是真实检索片段。评测响应的 `invocations` 会返回标准化后的 `answer`、`chunks`、`citations` 和调用状态，但不会暴露 Skill 的原始响应。接入生成模型后，可以在不改变评测引擎的前提下替换该答案生成步骤。

## SmallRAG Adapter

SmallRAG 通过 HTTP 接入，是当前可用于“黄金集 query → 自动调用目标 RAG → 回填 answer/chunks → 执行评测”的完整目标系统。启动两个服务时使用不同端口：

```powershell
# 终端 1：目标 RAG
Set-Location C:\smallrag
.\.venv\Scripts\python.exe -m uvicorn smallrag.main:app --port 8001

# 终端 2：评测系统
Set-Location C:\evaluation-Rag
$env:SMALLRAG_BASE_URL="http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`，在“回答来源”中选择 `SmallRAG`，上传至少含 `query` 和人工标注字段（通常为 `reference_answer`）的黄金集，然后开始评测。Adapter 调用 `/v1/query`，优先将响应中的 `contexts` 映射为 `chunks`；因此端到端、检索和生成模块能共享同一批真实运行数据。

可通过 `.env.example` 中的 `SMALLRAG_TOP_K`、`SMALLRAG_ALPHA`、超时、TLS 和可选上下文上限调整连接。当前应用不会自动加载 `.env`，本地运行前请设置环境变量，或使用进程管理工具注入配置。

批量调用默认最多同时向目标 RAG 发送 2 个请求，可用
`TARGET_RAG_MAX_CONCURRENCY` 调整；`TARGET_RAG_TIMEOUT_SECONDS` 控制单条样本的外层超时。
建议目标服务容量不明确时保持并发为 1 或 2。SmallRAG 返回结构化 502 时，调用明细会显示
具体的 `knowledge_base_upstream_error` 或 `model_upstream_error`，以及对应 request ID。

## 并行开发约定

- `app/evaluation/base.py` 和 `app/schemas.py` 是共享契约，修改前需要两位开发者确认。
- 两位开发者分别修改 `modules/end_to_end.py` 与 `modules/retrieval.py`、`modules/generation.py`，这些文件互不依赖。
- 指标通过模块内的 `METRICS` 列表注册，前端不硬编码指标名称。
- `chunks` 可以为空；此时仍可运行只需要 `query + answer` 的端到端指标，但检索和生成维度不可评测。
- JSONL 是“一行一个完整 JSON 对象”，示例对象需要压缩到单行后写入文件。

完整分支策略、模块所有权和接口变更流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，数据与指标契约见 [docs/API_CONTRACT.md](docs/API_CONTRACT.md)。


## 当前边界

- 数据集暂存在进程内，服务重启后需重新上传；后续可替换为数据库或对象存储。
- 规则指标同步计算，模型指标通过异步 Provider 并发执行；当前仍是请求内评测，大数据集需要任务队列。
- 回答相关性目前是可解释的词面基线，不代表语义正确性。
- 检索和生成指标已实现纯编码（词面/结构）基线；语义类指标已声明能力契约，后续可直接调用已注入的 Embedding/LLM Judge Provider。
- LLM Judge、Embedding Provider 已接入，具体模型指标仍按各模块计划逐项实现。
