# Job Application Research Agent — 技术报告

一个准生产级 **Agentic RAG** 项目：给定公司 / 岗位 / JD / CV，输出带引用的面试准备报告。
强调工程治理（多租户、权限、状态恢复、可观测、评估闭环），且**本地无外部 API key 即可全程运行**。

> 严格区分「已实现并被测试证明」与「抽象就绪/后续计划」。指标均为本地 deterministic 模式实测。

## 1. 架构

```
StateGraph / Runner（共享同一批节点）：
entry_guard → parse_jd → parse_cv → intent_route → retrieve
            → match → evidence_check → generate_questions
            → write_report → grounding_check → persist
```

- **LangGraph `StateGraph`**（`graph.py`）：可视化 / 可编排，节点为 LangGraph-compatible。
- **Runner**（`runner.py`）：本地确定性执行 + 每节点 checkpoint + 失败 resume（生产恢复逻辑在此）。
- 节点函数共享，避免双实现漂移。

## 2. 检索栈（Agentic RAG）

| 层 | 实现 | 默认 | 可切换 |
|---|---|---|---|
| Embedding | `embeddings.py`（可插拔 + 降级） | deterministic feature-hashing（离线、可复现） | `JOB_AGENT_EMBEDDER=fastembed`(ONNX) / `api`(bge-m3) / `sentence-transformers` |
| 向量库 | `vectorstore.py` Qdrant 本地模式 | `:memory:` / 路径持久化 | Qdrant server |
| Sparse | `bm25.py` 无依赖 BM25 Okapi | 启用 | — |
| 融合 | `hybrid.py` RRF + metadata boost | `retrieval_mode=hybrid` | `deterministic`（纯 lexical）/ `dense`（纯向量） |
| Rerank | `rerank.py` 启发式 | 启用 | `JOB_AGENT_RERANKER=cross-encoder` |
| Agentic loop | `evidence_check` 节点 | 证据不足→query rewrite→补检索→重匹配（≤2 轮） | — |
| Routing | `llm.py` intent 分类 + retriever selection | 规则版 | `llm_mode=auto`（真实 LLM） |

**权限**：每个 chunk payload 带 `tenant_id` + `acl_roles`；Qdrant 原生 filter（must tenant + should MatchAny）+
lexical 端同等过滤，两路都强制隔离。

## 3. 工程治理

- **状态恢复**：SQLite `checkpoints` 表，节点失败记 `failed_node`，`/sessions/{id}/resume` 从首个未完成节点续跑。
- **Guardrail**：输入四分类 pass/sanitize/clarify/block + 敏感信息脱敏；输出启发式 claim-level grounding
  （仅扫 JD-CV match 区块，未支撑能力声明进 `unsupported_claims`，≥3 条 block）。结构上 matcher 只在有证据时标
  strong/partial，杜绝伪造。
- **缓存/限流**：`cache.py` 抽象（in-memory 默认 / `REDIS_URL` 切 Redis）；检索节点默认 `use_cache=True`。
- **韧性**：`tool_executor.py` timeout + retry + fallback。
- **可观测**：`NodeTrace`（status/latency/checkpoint）+ `ToolTrace` + `Metrics`（TTFT/latency/rounds/cache_hits）。

## 4. API

`/healthz` · `/prepare` · `/prepare/stream` · `/sessions/{id}` · `/sessions/{id}/traces` ·
`/sessions/{id}/resume` · `/sessions/{id}/feedback` · `/knowledge/ingest` · `/eval/run`

## 5. 评估闭环

- **150 条**标注 regression set（10 类，`data/eval/job_agent_eval_v2.jsonl`），标注 expected_intent /
  expected_retrievers / gold_evidence / must_answer_points / must_not_claim / guardrail_expected。
- **真实指标**：recall@k · MRR · nDCG@k · context precision；citation accuracy · faithfulness ·
  unsupported-claim rate；intent acc · retriever selection acc · tool-call validity · trajectory；
  TTFT · p50/p95 · cache hit rate · token cost。
- **失败闭环**：failure taxonomy + regression candidates + deterministic vs hybrid 的 `before_after.md`。

**实测 summary（150 条，deterministic）**：recall@k 1.0 · nDCG 0.998 · context_precision 0.868 ·
faithfulness 1.0 · citation 0.987 · intent 0.933 · retriever 0.933 · guardrail 0.987 ·
task_success 0.98 · p50/p95 176/324ms · cache_hit 0.137。

**eval 驱动的真实改进**：faithfulness 0.013→1.0（grounding 误报修正）；intent 0.70→0.933（路由 + 对抗样本计分修正）。

## 6. 测试与口径

- job_agent 相关测试 **44 全绿**；完整 `pytest tests/unit_tests` **121 passed**；`ruff check .` 通过。
- 全量 e2e 依赖外部服务/API key，未作本地验收条件。
- 一切默认 deterministic offline：无 Docker / API key / 模型下载即可跑 ingest / prepare / resume / eval。

## 6.1 Phase 3 增量

- **检索可切三模式**：`deterministic`（lexical）/ `dense`（纯向量）/ `hybrid`（RRF+rerank）；
  `evaluation --compare` 输出三方 `before_after.md`。
- **真实 embedder 三后端**（fastembed / api / sentence-transformers），统一协议 + 降级。
- **语义难例集 v3（100 条）**：刻意制造 query↔证据词汇 gap（口语/缩写/同义/多跳/无答案/injection/中英混合）。
- **source quality**：`encoding.py` 稳健解码（UTF-8 BOM / GB18030）+ mojibake 检测，chunk 写 `bad_encoding` 标志。
- **一键 demo**：`scripts/demo.py` + `job_agent_demo.ps1` + `README_JOB_AGENT_DEMO.md`。

## 6.2 Phase 3.1 真实 API Embedding 基线

在用户授权后，使用 SiliconFlow OpenAI-compatible `/embeddings` 接口，对
`data/eval/semantic_hard_eval_v3.jsonl` 的 100 条语义难例做真实 embedding 对比。
端点使用 `https://api.siliconflow.com/v1`，模型对比 `Qwen/Qwen3-Embedding-0.6B`、
`Qwen/Qwen3-Embedding-4B`、`Qwen/Qwen3-Embedding-8B`。API key 未写入仓库。

| 模型 | 推荐模式 | recall@k | nDCG@k | MRR | task_success | p50 latency | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-0.6B | hybrid / dense | dense 0.5375 / hybrid 0.5292 | dense 0.4573 / hybrid 0.4373 | dense 0.3333 / hybrid 0.3643 | dense 0.48 / hybrid 0.51 | ~2.5s | 性价比最好；比 hashing 有真实语义收益，延迟可接受 |
| Qwen3-Embedding-4B | dense | dense 0.5750 | dense 0.5265 | dense 0.4081 | dense 0.54 | ~4.0s | 当前指标最佳；适合作为“质量优先”展示基线 |
| Qwen3-Embedding-8B | dense | dense 0.5575 | dense 0.4624 | dense 0.3317 | dense 0.51 | ~7.2s | 不推荐；更慢但没有超过 4B |

关键发现：

- **更大模型不等于更好**：8B 在本任务上没有超过 4B，且 p50 延迟约 7.2s、p95 超过 25s，
  不适合作为默认线上方案。
- **当前瓶颈从 embedding 转到融合/排序**：4B 的 dense-only 最强，但 hybrid 被 BM25/RRF/启发式 rerank
  稀释，说明下一步应调 fusion weight、metadata boost 和 rerank，而不是继续盲目换更大 embedding。
- **缓存确实命中 retrieval 工具层**：8B run 中 retrieval cache hit rate 约 0.16，
  命中工具分别为 `dense_retrieval_tool`、`hybrid_retrieval_tool`、`rag_retrieval_tool`。
  但当前缓存的是“检索结果”，不是“embedding 向量计算”；`HybridRetriever(index)` 构建时仍可能触发远程 embedding，
  因此二次运行仍会很慢。后续应增加 `(model, text_hash) -> vector` 的 embedding cache 或持久化 Qdrant index。
- **intent routing 仍是主要短板**：语义难例集上规则 intent accuracy 仍约 0.1，
  需要接 LLM routing 或训练/规则增强的 intent classifier。

## 7. 诚实的局限

- **本地真实 embedder 仍受环境限制**：本机缺 MSVC 运行库，`onnxruntime` / `torch` DLL 无法加载；
  fastembed / sentence-transformers 本地基线未跑出。但 API embedding 已跑通，详见 `reports/eval_semantic_qwen06b/`、
  `reports/eval_semantic_qwen4b/`、`reports/eval_semantic_qwen8b/`。
- grounding / intent 为启发式，非语义级 NLI / LLM verifier。
- Redis、cross-encoder、真实 LLM 均为「抽象就绪 + 降级」，未跑真实基线。
- ACL 为打分前硬过滤，非独立 post-retrieval audit 阶段。

## 8. 简历可用句（中/英）

- 设计并实现准生产级 Agentic RAG 服务：多节点 LangGraph + 确定性 runner（checkpoint/resume），
  Qdrant 混合检索（BM25 + dense + RRF + rerank），多租户 ACL，claim-level grounding，
  150 条评估集（recall@k/nDCG/faithfulness/intent 等真实指标），全程本地可跑、无 key 降级。
- Built a production-style Agentic RAG service: multi-node LangGraph + deterministic runner with
  checkpoint/resume; hybrid retrieval (BM25 + dense + RRF + rerank over Qdrant) with multi-tenant ACL;
  claim-level grounding to prevent fabricated experience; a 150-case eval harness with real
  recall@k / nDCG / faithfulness / intent metrics and a failure-taxonomy feedback loop — all runnable
  offline with graceful degradation when no API key is present.
