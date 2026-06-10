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

# Hybrid 融合权重扫描（找让 hybrid 超过 dense-only 的权重）
uv run python scripts/fusion_sweep.py --metric recall_at_k          # 离线（hashing）验证流程
# 真实调优：先设 JOB_AGENT_EMBEDDER=api + key + 模型，再跑同一条命令
# -> reports/fusion_sweep/fusion_sweep.md（最优 env + 复现命令）
```

**实测结果（Qwen3-Embedding-4B，100 条 v3，目标 recall@k）**：最优 `BM25=0.5 / dense=2.0 / RRF k=30`
将 hybrid recall@k **0.478 → 0.575**、task_success **0.35 → 0.54**（vs 等权融合）。实证「等权 RRF 稀释
dense、hybrid 需调权」。注：最优配置按 recall 选出，其 MRR(0.352) 仍略低于等权(0.370) → rerank 是下一杠杆。

```bash
# Rerank A/B（固定最优 fusion，只换 reranker；解决 MRR/排序质量）
uv run python scripts/rerank_compare.py --strategies none heuristic --metric mrr   # 离线验证
# 真实 reranker（走 API key，不需 torch）：
$env:JOB_AGENT_EMBEDDER="api"
$env:JOB_AGENT_EMBED_API_BASE="https://api.siliconflow.com/v1"
$env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
$env:JOB_AGENT_RERANKER="api"          # SiliconFlow /rerank
$env:JOB_AGENT_RERANK_API_BASE="https://api.siliconflow.com/v1"
$env:JOB_AGENT_RERANK_MODEL="Qwen/Qwen3-Reranker-8B"
$env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
uv run python scripts/rerank_compare.py --strategies none api --metric mrr
# -> reports/rerank_compare/rerank_before_after.md
```

**真实 rerank 结果（Qwen3-Reranker-8B，固定最优 fusion）**：MRR **0.385 → 0.643**、
nDCG **0.4735 → 0.6491**，recall@k 保持 **0.575**，p50 latency 约 **1385ms → 2698ms**。
这说明正确证据已经在候选池里，reranker 主要把好证据排到更前面。

```bash
# Intent routing A/B（规则 vs LLM；解决 v3 上规则 routing ~0.1 的短板）
uv run python scripts/intent_compare.py --metric intent_accuracy        # 离线：off==auto（无 key 回退规则）
# 真实 LLM routing（走 chat key）：
$env:JOB_AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"; $env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
uv run python scripts/intent_compare.py --modes off auto --metric intent_accuracy
# -> reports/intent_compare/intent_before_after.md
```

**真实 intent A/B 结果（Qwen2.5-7B-Instruct + Qwen3-Embedding-4B，100 条 v3）**：
intent accuracy **0.10 → 0.27**，retriever selection **0.10 → 0.27**，recall@k
**0.4783 → 0.9817**，task_success **0.35 → 0.84**，p50 latency 约 **1395ms → 4090ms**。
这说明 LLM routing 有帮助，但更大的收益来自 query rewrite；生产上应对口语/缩写/规则低置信请求选择性启用。
**routing-only 消融**（只保留 LLM intent、禁用 rewrite）：intent **0.10 → 0.24**，但 recall@k
仍为 **0.4783**、task_success 仍为 **0.35**，说明当前召回提升主要来自 query rewrite；若要让
intent 本身影响召回，需要继续把 `selected_retrievers` 接入 source-level routing / metadata filter。

```bash
# 4-way 路由消融（off / routing-only / +source / +source+rewrite）—— Phase 5.3
# selected_retrievers 现已真正接入 source_type 过滤（含广播回退保护 recall）
uv run python scripts/routing_compare.py --metric recall_at_k --limit 40   # 离线即可测 source routing 收益
# 真实 Router v2（JSON 路由）：加 $env:JOB_AGENT_LLM_MODEL + key，去掉 --limit
# -> reports/routing_compare/routing_ablation.md
# run_eval 还会输出 intent_confusion.md（expected→predicted 混淆矩阵 + top 混淆对）
```

**离线 4-way 消融实测（100 条 v3，无 chat key）**：off / routing_only 的 recall 都是 **0.435**
（印证 routing-only 不接检索就不动召回）；**routing+source** 把 recall 提到 **0.4883**、nDCG
**0.3953→0.4499**、task_success **0.34→0.38**，且 evidence_sufficiency 仅 1.0→0.98——说明
**source-level routing 净收益为正、广播回退防止了过窄**。query rewrite 的额外收益要 chat key 才显现。

**线上 4-way 消融实测（Qwen2.5-7B-Instruct + Qwen3-Embedding-4B，40 条 v3）**：
`routing_only` 仍不改变 recall（0.3458），`routing+source` 把 recall 提到 **0.5583**、task_success
**0.30→0.55**；`routing+source+rewrite` 进一步把 recall 提到 **0.6208**、nDCG 提到 **0.3857**、
task_success 到 **0.60**，但 p50 latency 从约 **1249ms→6282ms**。100 条线上全量因 chat 调用较多超时，
因此这组是完整 40 条趋势样本，不夸大为全量最终结论。
注意：`routing+source+rewrite` 的 retriever accuracy 低于 `routing+source`，但 recall/task 更高；这说明
retriever accuracy 是路由贴标诊断指标，rewrite 主要提升 query 与证据的语义匹配，最终业务效果应看
recall、nDCG、task_success 和 faithfulness。

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

# 可选：真实 API reranker（质量优先；不需要本地 torch/MSVC）
$env:JOB_AGENT_RERANKER="api"
$env:JOB_AGENT_RERANK_API_BASE="https://api.siliconflow.com/v1"
$env:JOB_AGENT_RERANK_MODEL="Qwen/Qwen3-Reranker-8B"
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
