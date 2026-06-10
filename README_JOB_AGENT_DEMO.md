# Job Application Research Agent — 本地 Demo 指南

一个准生产级 **Agentic RAG** 服务：给定公司 / 岗位 / JD / CV，输出带引用的面试准备报告。
**本地无外部 API key 即可全程跑通**（deterministic embedder + 本地 Qdrant，无需 Docker / 模型下载）。

## 60 秒上手

```powershell
# Windows PowerShell（一条命令跑完 ingest -> prepare -> resume -> eval）
pwsh scripts/job_agent_demo.ps1
```

或手动：

```bash
uv run python -m job_agent.eval_dataset       # 生成 150 条评估集
uv run python -m job_agent.eval_dataset_v3    # 生成 100 条语义难例集
uv run python scripts/demo.py                 # 端到端演示
```

`scripts/demo.py` 会依次：

1. **ingest** 一个知识 manifest 进本地 Qdrant（持久化）；
2. **prepare** 一份带引用的报告（hybrid 检索），打印 intent / 检索器 / 节点轨迹 / 指标；
3. **resume** 同一 session（从 checkpoint 恢复）；
4. **eval** 跑 150 条评估并打印真实指标。

## 起服务 + curl

```bash
# 终端 A
uv run uvicorn job_agent.entrypoint:app --port 8000
```

```bash
# 终端 B —— 生成报告
curl -s -X POST http://localhost:8000/prepare \
  -H "Content-Type: application/json" \
  -d '{"company":"Tencent","role":"AI Agent Engineer","job_description":"Python RAG AI Agent FastAPI evaluation guardrails","cv_text":"Python, RAG, FastAPI, LangGraph","retrieval_mode":"hybrid"}'

# 流式（先 skeleton 再 result）
curl -N -X POST http://localhost:8000/prepare/stream \
  -H "Content-Type: application/json" \
  -d '{"company":"Tencent","role":"AI Agent Engineer","job_description":"Python RAG FastAPI","cv_text":"Python RAG FastAPI"}'

# 知识入库
curl -s -X POST http://localhost:8000/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"collection":"job_agent_knowledge","items":[{"text":"RAG with FastAPI and LangGraph","doc_type":"note","tenant_id":"acme","acl_roles":["tenant:acme"]}]}'

# 触发离线评估（优先 v2）
curl -s -X POST http://localhost:8000/eval/run

# 状态恢复 / 轨迹
curl -s -X POST http://localhost:8000/sessions/<id>/resume
curl -s http://localhost:8000/sessions/<id>/traces
```

## 评估与检索对比

```bash
# 150 条结构化评估（真实指标 + 失败 taxonomy + regression 候选）
uv run python -m job_agent.evaluation --eval-file data/eval/job_agent_eval_v2.jsonl --output-dir reports/eval_v2

# 100 条语义难例上的 lexical vs dense vs hybrid 三方对比
uv run python -m job_agent.evaluation --eval-file data/eval/semantic_hard_eval_v3.jsonl \
  --output-dir reports/eval_semantic --compare
# -> reports/eval_semantic/before_after.md
```

**150 条实测**（deterministic）：recall@k 1.0 · nDCG 0.998 · context_precision 0.868 ·
faithfulness 1.0 · citation 0.987 · intent 0.933 · task_success 0.98 · p50/p95 176/324ms。

## 切换真实模型（可选，非默认）

默认是 deterministic 离线降级；配置环境变量即可切真实后端（失败自动回退 hashing）：

```bash
# A) 本地 ONNX（需 Windows MSVC 运行库）
uv sync --extra embeddings
$env:JOB_AGENT_EMBEDDER="fastembed"; $env:JOB_AGENT_EMBED_MODEL="BAAI/bge-small-en-v1.5"

# B) API embedding（OpenAI 兼容；SiliconFlow GitHub/cloud 账号实测用 .com endpoint）
$env:JOB_AGENT_EMBEDDER="api"
$env:JOB_AGENT_EMBED_API_BASE="https://api.siliconflow.com/v1"
$env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
$env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"  # 速度/效果平衡
# 质量优先可切：Qwen/Qwen3-Embedding-4B；不建议默认 8B（延迟高且未超过 4B）

# 可选：真实 reranker / LLM routing
$env:JOB_AGENT_RERANKER="cross-encoder"
# llm_mode=auto 在请求体里传，配合 DASHSCOPE/OPENAI/SILICONFLOW key
```

> ⚠️ 本机当前缺少 MSVC 运行库，onnxruntime/torch 无法加载，因此 fastembed / sentence-transformers
> 本地基线未跑出；但 API embedding 已跑通。v3 语义难例上，Qwen3-Embedding-4B dense-only
> 当前质量最好，Qwen3-Embedding-0.6B 性价比最好，8B 不建议默认使用。见
> `reports/eval_semantic_qwen06b/`、`reports/eval_semantic_qwen4b/`、`reports/eval_semantic_qwen8b/`。

## 5 分钟讲稿（面试）

1. **它是什么**：把简历/JD 变成带引用的面试准备报告的 Agentic RAG 服务。
2. **架构**：多节点 LangGraph + 确定性 runner（checkpoint / resume），检索栈 = BM25 + dense(Qdrant) + RRF + rerank。
3. **工程治理**：多租户 ACL（Qdrant 原生 filter）、claim-level grounding 防伪造、tool executor（timeout/retry/fallback）、缓存/限流抽象、全链路 trace。
4. **怎么证明有效**：150 条标注评估集 + 真实指标（recall@k / nDCG / faithfulness / intent），失败 taxonomy → regression 候选；并展示 eval 驱动的真实改进（faithfulness 0.013→1.0、intent 0.70→0.933）。
5. **诚实的边界**：deterministic 默认可离线复现；真实 embedder / reranker / LLM / Redis 都是「抽象就绪 + 一键切换 + 降级」，未作默认依赖。
