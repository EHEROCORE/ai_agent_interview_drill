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
| Rerank | `rerank.py` 启发式 | 启用 | `JOB_AGENT_RERANKER=api` / `cross-encoder` |
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

## 4.1 链路接口与参数设计

这一节记录实际代码里的接口契约，重点不是“用了哪些模块”，而是每一层怎么交接、参数为什么这样设、失败时怎么降级。

### 4.1.1 请求与响应契约

`PrepareRequest` 是 `/prepare`、`/prepare/stream` 和 workflow 的统一入口。核心字段如下：

| 字段 | 类型 / 默认值 | 作用 | 设计原因 |
|---|---|---|---|
| `company` | `str`, 1-120 | 公司名 | 进入 query rewrite、报告标题、公司准备检索。长度限制防止异常长输入拖慢检索。 |
| `role` | `str`, 1-160 | 岗位名 | 用于 intent routing、问题生成和报告上下文。 |
| `job_description` | `str`, 20-30000 | JD 原文 | 最小长度避免空 JD，最大长度避免一次请求塞入过大上下文。 |
| `cv_text` | `str | None`, <=30000 | 简历文本 | 和 `cv_file_path` 二选一。当前主链路优先用 inline 文本，方便测试和 API demo。 |
| `cv_file_path` | `str | None`, <=500 | 简历文件路径 | 保留文件输入能力，但为了安全只在受控 workspace 读取。 |
| `target_interview_type` | 默认 `AI Agent / LLM application` | 面试方向 | 进入 query rewrite 和问题生成，避免生成泛泛的 HR 题。 |
| `tenant_id` | 默认 `default` | 租户隔离键 | 进入 Qdrant payload filter、lexical filter、post-retrieval ACL audit。 |
| `user_id` | 默认 `anonymous` | 用户身份 | 当前用于 trace/审计口径，后续可接 IAM。 |
| `user_roles` | 默认 `["public"]` | 权限角色 | 与 chunk 的 `acl_roles` 做交集判断。运行时还会补 `tenant:{tenant_id}`。 |
| `retrieval_mode` | `deterministic/dense/hybrid`, 默认 `deterministic` | 检索模式 | 默认可离线复现；`dense`/`hybrid` 用于真实 embedding 和 A/B。 |
| `llm_mode` | `off/auto`, 默认 `off` | 是否启用真实 LLM routing | 默认不用 key；`auto` 有 key 时走 LLM，失败回退规则。 |

`PrepareResponse` 返回 `report/citations/matches/interview_questions` 这些业务结果，同时暴露 `intent`、`selected_retrievers`、`metrics`、`tool_trace`、`node_trace`。这样面试和调试时不只看最终文本，还能回答“Agent 为什么这么做、检索了什么、哪一步慢、是否命中缓存、有没有 checkpoint”。

### 4.1.2 节点之间的 state 接口

多节点链路共享 `GraphState`，每个节点只读自己需要的字段，并返回 partial update。这样做的好处是：节点可以单测、可以 checkpoint、可以从中间恢复，也能映射到 LangGraph 的 `StateGraph`。

| 节点 | 主要输入 | 主要输出 | 失败 / 降级策略 |
|---|---|---|---|
| `entry_guard` | `request.company/role/job_description/cv_text` | `guardrail` | `block` 时短路，不继续调用工具；`sanitize` 由后续文本脱敏承接。 |
| `parse_jd` | `job_description` | `requirements: list[Requirement]` | 规则 parser，无 LLM 依赖；失败会记录 failed node 供 resume。 |
| `parse_cv` | `cv_text` | `cv_profile: CVProfile` | 用 `cv_text_hash` 进 trace，避免把完整 CV 放进工具参数日志。 |
| `intent_route` | `request + requirements` | `intent + selected_retrievers` | `llm_mode=off` 走规则；`auto` 失败回退 `interview_prep`。 |
| `retrieve` | `query + retrieval_mode + tenant/user_roles` | `citations + retrieval_latency_ms` | `hybrid/dense` 失败可通过 tool executor fallback 到空结果；后续 evidence_check 可补检索。 |
| `match` | `requirements + citations` | `matches` | 只在有证据时给 strong/partial，否则标 gap，避免伪造经历。 |
| `evidence_check` | `matches + citations` | 补充后的 `citations/matches/retrieval_rounds` | 最多 2 轮，防止证据不足时无限循环。 |
| `generate_questions` | `role + requirements + matches` | `interview_questions` | 规则生成，可离线运行。 |
| `write_report` | `company/role/matches/questions/citations` | `report` | 报告写出前做敏感信息 redaction。 |
| `grounding_check` | `report + citations` | `output_guardrail` | 未支撑 claim 进入 `unsupported_claims`，超过阈值 block。 |
| `persist` | `report + session_id` | `report_path` | 写入 `reports/job_agent/{session_id}.md`。 |

节点 trace 使用 `NodeTrace(node, status, latency_ms, checkpoint, detail)`；工具 trace 使用 `ToolTrace(tool_name, args_hash, status, latency_ms, observation)`。`args_hash` 来自稳定 JSON 序列化后的 `sha1(tool_name + normalized_args)`，所以相同 JD/CV/query 能复用结果，也能避免日志里直接暴露大段隐私文本。

### 4.1.3 检索链路接口

在线检索入口是 `_run_retrieval(state, store, query, extra_args)`，它把同一套 query 分发到三种模式：

| `retrieval_mode` | 底层实现 | 返回 | 使用场景 |
|---|---|---|---|
| `deterministic` | `HybridRAGIndex.query()`，lexical cosine + keyword overlap | `list[RetrievedChunk]` | 默认离线、CI、无模型环境。 |
| `dense` | `HybridRetriever.query_dense()`，Qdrant cosine | `list[RetrievedChunk]` | 对比真实 embedding 的纯语义收益。 |
| `hybrid` | BM25 + Qdrant dense + weighted RRF + metadata boost | `list[RetrievedChunk]` | 生产主路径和调优主路径。 |

固定参数：

- `RETRIEVAL_TOP_K = 10`：第一版上下文只喂 10 个 chunk，原因是面试准备报告更重视 citation 可读性和延迟，不是越多越好；如果 top-k 太大，rerank 和 grounding 都会变慢，且 context precision 下降。
- `MAX_RETRIEVAL_ROUNDS = 2`：第一轮按主 query 检索，第二轮只在 `gap` 存在时补检索。两轮上限是防死循环设计：证据不足可以补，但不能让 Agent 无限“再搜一次”。
- `user_roles = request.user_roles + ["tenant:{tenant_id}"]`：用户显式角色和租户角色合并，既支持公共资料，也支持租户私有资料。

`RetrievedChunk` 的接口是：

| 字段 | 含义 |
|---|---|
| `citation_id` | 当前 chunk 的引用 ID，报告里用它做 citation。 |
| `text` | chunk 正文。 |
| `source_path` | 来源文件或 inline 来源。 |
| `metadata` | `tenant_id/doc_type/section/topic/company/role/chunk_id/acl_roles/...`。 |
| `score` | 当前检索或融合后的分数，只用于排序解释，不直接等价于概率。 |

### 4.1.4 离线 ingestion 到在线检索的接口

`/knowledge/ingest` 接收 `IngestManifest`，每个 `ManifestItem` 支持 `source_path` 或 inline `text`。离线流程是：

`manifest -> decode -> clean -> section-aware chunk -> metadata enrichment -> embedding -> Qdrant upsert`

统一 payload：

| payload 字段 | 作用 |
|---|---|
| `tenant_id` | 检索前硬隔离，必须匹配请求租户。 |
| `acl_roles` | 角色权限，Qdrant filter 和 post audit 都会检查。 |
| `doc_type` | 区分 interview question、technical note、resume project、plan、company、memory 等逻辑来源。 |
| `source_path` | citation 和审计来源。 |
| `section/topic/company/role` | metadata boost、过滤和报告解释。 |
| `chunk_id/updated_at` | 去重、版本追踪。 |
| `encoding/bad_encoding` | 数据质量标记，避免乱码静默进入知识库。 |

这里的“parse”和“clean”不是一回事：parse 是把文件/文本读成可处理的结构和字段，clean 是去噪、解码、规范空白和标记乱码。当前 pipeline 先 `_load_text/decode_bytes`，再 `clean_text`，最后 chunk 和 enrich。

### 4.1.5 Hybrid fusion 参数

`FusionConfig` 是 Phase 4 的核心参数接口，可用 `JOB_AGENT_FUSION_*` 环境变量覆盖：

| 参数 | 默认值 | 实测调优值 | 作用 |
|---|---:|---:|---|
| `bm25_weight` | 1.0 | 0.5 | sparse/BM25 排名权重。降低它是因为 v3 中真实 dense 已经能抓语义改写，BM25 等权会稀释 dense。 |
| `dense_weight` | 1.0 | 2.0 | dense/Qdrant 排名权重。提高它让语义相关 chunk 更容易进最终前列。 |
| `rrf_k` | 60 | 30 | RRF 平滑项。更小的 k 会放大靠前名次差异，适合候选池已经较准但排序要更敏感的场景。 |
| `metadata_boost_weight` | 0.02 | 0.02 | metadata token 命中加分，防止公司/岗位/章节这类结构信息被正文语义冲掉。 |
| `sparse_top_k` | 30 | 30 | BM25 候选池大小。 |
| `dense_top_k` | 30 | 30 | dense 候选池大小。 |
| `final_top_k` | 10 | 10 | 融合后进入 rerank/context 的上限。 |

融合公式是 weighted RRF：每个候选在每一路按 `weight / (rrf_k + rank + 1)` 累加，再加 metadata boost。选择 RRF 而不是直接归一化分数，是因为 BM25 分数和 cosine 分数尺度不同，直接相加容易被某一路分数范围支配；RRF 只依赖排名，更稳。

### 4.1.6 Rerank 参数

`JOB_AGENT_RERANKER` 支持：

| 值 | 作用 | 说明 |
|---|---|---|
| `none` | 直通 fusion 原序 | A/B baseline，用来判断 reranker 是否真的改善排序。 |
| `heuristic` | 离线默认 | `0.5 * retrieval_score + 0.3 * query_coverage + 0.2 * overlap`，无模型可跑。 |
| `api` | SiliconFlow `/rerank` | 请求体为 `{model, query, documents, top_n}`；默认模型已改为 `Qwen/Qwen3-Reranker-8B`。 |
| `cross-encoder` | 本地 CrossEncoder | 需要 torch/MSVC，当前本机未作为验收基线。 |

rerank 只改变候选顺序，不新增候选。因此实验里 recall@k 和 task_success 不变是合理结果；MRR/nDCG 大幅提升才说明“好证据已经被召回，但原排序没排到前面”。

### 4.1.7 缓存与恢复接口

系统里有两层缓存：

| 缓存层 | key | value | 解决的问题 |
|---|---|---|---|
| tool result cache | `sha1({"tool": tool_name, "args": normalized_args})` | 工具 observation | 相同 retrieval/query/tool 参数直接复用结果，减少重复工具执行。 |
| embedding cache | `sha1(model_name + "\0" + text)` | embedding vector | 重建 retriever 或 sweep 多组参数时，不重复请求远程 embedding API。 |

Phase 3.1 暴露的问题是：retrieval result cache 命中后，`HybridRetriever(index)` 重建仍可能重新算 chunk embedding。Phase 4 的 `EmbeddingCache + CachingEmbedder` 修的是这一层，所以它和 tool cache 是互补关系，不是重复实现。

恢复链路由 SQLite `checkpoints` 表承接：每个节点结束后保存 `(session_id, node, status, state_json)`。`/sessions/{session_id}/resume` 读取已完成节点，从第一个未完成节点继续；如果中间对象不可序列化，比如内存里的 index/retriever，恢复时由 `_ensure_index()` 懒加载重建。

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

- job_agent 相关测试 **66 全绿**；完整 `pytest tests/unit_tests` **143 passed**；`ruff check .` 通过。
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

## 6.3 Phase 4 增量（针对 3.1 暴露的瓶颈）

- **Embedding / Vector Cache**：`EmbeddingCache`(SQLite, `model+text_hash → vector`) + `CachingEmbedder`，
  真实后端默认启用 → 重建 retriever / eval 多轮**只对 miss 调远程 API**，修掉「retrieval 缓存命中但
  embedding 仍重算」。
- **可调加权 Hybrid Fusion + 实证调优**：`FusionConfig` + 加权 RRF（`JOB_AGENT_FUSION_*` 可调）；
  `tuning.py`/`scripts/fusion_sweep.py` 做权重网格搜索。**用 Qwen3-Embedding-4B 实测调出最优
  `BM25=0.5/dense=2.0/RRF k=30`：hybrid recall@k 0.478→0.575、task_success 0.35→0.54**，
  实证「等权 RRF 稀释 dense、hybrid 需调权」。
- **独立 post-retrieval ACL audit**：`acl.py` 在检索+rerank 后二次校验 tenant/role，越权 chunk 剔除并
  在 trace 留 `denied` 原因（defense-in-depth）。
- **Rerank A/B harness + 真实 API rerank（Phase 5.1）**：`rerank.py` 支持
  `none/heuristic/api(SiliconFlow /rerank)/cross-encoder`；`run_rerank_compare` 固定最优 fusion、只换 reranker。
  在 Qwen3-Embedding-4B + 最优 fusion 下，`Qwen/Qwen3-Reranker-8B` 将 MRR 0.385→0.643、
  nDCG 0.4735→0.6491（p50 1385→2698ms），证明 rerank 修复了 fusion 后的排序短板。
- **LLM intent routing + query rewrite（Phase 5.2）**：统一 `_chat_completion` seam（OpenAI 兼容，
  默认 SiliconFlow chat）。`llm_mode=auto` 时把**单个决策节点**交给 LLM——判 8 类意图 + 把口语/缩写
  query 改写成专业检索词；非法/异常一律回退规则与 deterministic query；`intent_route` 每请求仅 1 次改写。
  `run_intent_compare`(off vs auto) A/B 对症 v3 规则 routing ~0.1。真实 Qwen2.5-7B-Instruct 跑出：
  intent 0.10→0.27、recall@k 0.4783→0.9817、task_success 0.35→0.84（p50 1395→4090ms）。
- **Intent Router v2 + source-level routing（Phase 5.3）**：把 5.2 消融暴露的「`selected_retrievers`
  只是可观测字段」补上。`route_intent` 让 LLM 输出严格 JSON（intent/confidence/retrievers/rewrite_needed），
  enum 校验、低置信→`multi_source` 广播、非法回退规则；`selected_retrievers`→`source_type` 过滤接入
  **两路检索**，并加 `MIN_SOURCE_EVIDENCE` 广播回退防过窄。新增 confusion matrix + `evidence_sufficiency`
  指标 + 4-way 消融 harness（`run_routing_compare`）。**离线即测出 source routing 净收益**：recall
  0.435→0.4883、nDCG 0.3953→0.4499、task_success 0.34→0.38（见 6.4）。

## 6.4 指标变化与原因推断

| 阶段 | 主要变化 | 指标变化 | 推测原因 | 下一步 |
|---|---|---|---|---|
| Phase 2 v2 deterministic | 150 条结构化 eval + 规则 RAG | recall@k 1.0、nDCG 0.998、task_success 0.98 | v2 集合偏结构化，词面与知识库高度重合，lexical 已足够。 | 需要更难的语义改写集，避免“demo 指标虚高”。 |
| Phase 3 v3 hashing | 100 条语义难例 | recall 降到约 0.47，intent accuracy 约 0.1 | v3 人为制造口语/缩写/同义词/多跳 gap；hashing embedder 不具备真正语义能力；规则 intent 不适合口语化输入。 | 接真实 embedding，后续接 LLM intent routing。 |
| Phase 3.1 Qwen3-Embedding-4B dense | 真实 embedding | lexical baseline recall@k 0.4758、task_success 0.37 → 4B dense recall@k 0.5750、task_success 0.54 | 真实 embedding 能把“首 token 延迟/TTFT”“RAG 证据不足/grounding”等语义近邻拉近。 | 不再盲目换更大 embedding，转向 fusion/rerank。 |
| Phase 3.1 Qwen3-Embedding-8B | 更大 embedding | recall@k 0.5575、MRR 0.3317、p50 ~7.2s，未超过 4B | 更大模型不一定匹配当前业务分布；延迟显著变高，且 candidate ordering 没明显改善。 | 默认不推荐 8B embedding。 |
| Phase 4 fusion sweep | `BM25=0.5/dense=2.0/RRF k=30` | equal-weight recall 0.478、task_success 0.35 → tuned recall 0.575、task_success 0.54 | 等权 RRF 把强 dense 的信号摊薄；调高 dense 后召回恢复，但保留 0.5 BM25 避免完全丢掉精确词匹配。 | MRR 仍未改善，说明排序是新瓶颈。 |
| Phase 5.1 Qwen3-Reranker-8B | 固定最优 fusion，只换 reranker | none MRR 0.385、nDCG 0.4735 → 8B MRR 0.643、nDCG 0.6491；recall 0.575 不变 | 好证据已经在候选池里，reranker 通过 query-document 交互式打分把它排到更前面。recall 不变证明它没有新增候选，只修排序。 | 下一优先级是 intent routing。 |
| Phase 5.2 LLM intent routing + query rewrite | 在 `intent_route` 节点接 LLM seam，只判意图和改写 query | intent 0.10→0.27、retriever selection 0.10→0.27、recall@k 0.4783→0.9817、task_success 0.35→0.84；p50 1395→4090ms | LLM 解决一部分规则意图误判；query rewrite 对召回影响更大，把口语/缩写翻成专业检索词后，证据更容易进入候选池。 | 延迟上升明显，生产上应做选择性启用；之后做语义级 grounding。 |
| Phase 5.2 routing-only ablation | 临时禁用 `llm_rewrite_query`，只保留 LLM intent 分类 | intent 0.10→0.24、retriever selection 0.10→0.24；recall@k 0.4783 不变、task_success 0.35 不变；p50 1224→2462ms | 证明 LLM 分类本身只改善标签判断；当前 retrieval path 还没有用 `selected_retrievers` 真正过滤 source，所以候选池不变。 | 后续若要让 intent 影响 recall，需要接 source-level retriever routing / metadata filter。 |
| Phase 5.3 source-level routing（离线 4-way 消融，100 条 v3） | `selected_retrievers`→`source_type` 过滤接入两路检索 + 广播回退 | off/routing_only recall 0.435 不变 → routing+source recall 0.4883、nDCG 0.3953→0.4499、task_success 0.34→0.38；evidence_sufficiency 1.0→0.98 | 接入 source 过滤后候选池真正改变：去掉跨来源噪声 chunk 提升了排序与召回；广播回退保证 CV-centric 语料不会被过窄打穿（仅 0.02 evidence 代价）。这是**无需 chat key 就能验证**的结论。 | 真实 Router v2 的 intent label 提升仍需 chat key；下一步语义级 grounding。 |
| Phase 5.3 online source+rewrite（线上 40 条 v3） | 真实 chat-key 下跑 4-way routing ablation | off recall 0.3458、task 0.30；routing_only recall 不变；routing+source recall 0.5583、task 0.55；routing+source+rewrite recall 0.6208、nDCG 0.3857、task 0.60；p50 1249→6282ms | 证明在线 LLM rewrite 在 source routing 之上仍有增益；但延迟代价明显。100 条全量因 chat 调用超时，40 条作为趋势样本，不夸大为全量最终结论。 | 做按需启用策略：低置信/口语缩写/evidence 不足时再启用 rewrite。 |

**关于 `retriever_selection_accuracy` 下降但 recall/task 上升的解释**：
`retriever_selection_accuracy` 衡量的是「模型选出的 retrievers 是否贴合 eval 标注的 expected_retrievers」，
本质是路由诊断指标，不是最终业务指标；而 `rewrite` 改善的是 query 与证据文本之间的语义匹配。
因此线上 40 条中出现 `routing+source` 的 retriever_acc 0.15 高于 `routing+source+rewrite` 的 0.075，
但后者 recall 0.5583→0.6208、nDCG 0.3263→0.3857、task_success 0.55→0.60。含义是：
LLM 选的 retriever 未必完全贴合人工标注，甚至可能因为多意图问题偏离 expected_retrievers；
但 rewrite 把口语/缩写翻成知识库更容易匹配的专业表达，仍然能让正确证据进入候选池。
另外，当前系统有 `MIN_SOURCE_EVIDENCE` 广播回退，错误或过窄 source filter 会被放宽，
所以 retriever_acc 下降不一定导致 recall 下降。报告口径上应把 retriever_acc 当作路由质量诊断，
最终效果仍以 recall@k、nDCG、task_success、faithfulness 为主。

这条指标链路的面试表达可以概括为：先用 v2 证明工程链路能跑，再用 v3 主动打破“词面匹配舒适区”；真实 embedding 解决“找得到一部分语义证据”，fusion sweep 解决“多路召回不要互相稀释”，rerank 解决“候选池里有好证据但排序靠后”，LLM routing 则只放在“规则最容易误判的意图/改写决策点”。每一步都隔离变量、保留 before/after，而不是只说“我加了一个模块”。

## 7. 诚实的局限

- **本地真实 embedder 仍受环境限制**：本机缺 MSVC 运行库，`onnxruntime` / `torch` DLL 无法加载；
  fastembed / sentence-transformers 本地基线未跑出。但 API embedding 已跑通，详见 `reports/eval_semantic_qwen06b/`、
  `reports/eval_semantic_qwen4b/`、`reports/eval_semantic_qwen8b/`。
- **rerank 修复排序质量，但带来延迟 trade-off**：Qwen3-Reranker-8B 将 MRR 0.385→0.643、nDCG 0.4735→0.6491，
  但 p50 latency 从约 1.4s 增至约 2.7s；recall/task_success 不变，因为 rerank 只重排候选池，不新增候选。
- grounding 仍为启发式 token-overlap，非语义级 NLI / LLM verifier。
- LLM intent routing + query rewrite 已跑出真实 chat-key A/B，但 p50 latency 从约 1.4s 升到约 4.1s；source+rewrite 线上 40 条进一步显示 p50 可到约 6.3s。默认 `llm_mode=off` 仍保持确定性离线，生产上建议只对口语/缩写/规则低置信/evidence 不足请求启用。
- Redis、cross-encoder、本地真实 LLM 均为「抽象就绪 + 降级」，未跑真实基线。

## 8. 端到端组件接口地图

这一节按真实执行链路列出每个部件的接口、输入输出、模型/工具依赖、失败策略和观测字段。面试深挖时可以按这张表从入口一路讲到评估。

| 链路层 | 代码入口 | 输入接口 | 输出接口 | 模型/工具依赖 | 用途 | 失败/回退 | 观测/指标 |
|---|---|---|---|---|---|---|---|
| API/Auth/Rate Limit | `entrypoint.py::_check_auth` | `X-API-Key`、请求身份 | identity 或 401/429 | `cache.get_cache()` | API key 校验、60s 窗口限流 | 未配置 `JOB_AGENT_API_KEY` 时允许 anonymous；超限返回 429 | rate-limit key、HTTP status |
| 请求校验 | `schemas.py::PrepareRequest` | `company/role/JD/CV/tenant/user_roles/retrieval_mode/llm_mode/source_routing` | Pydantic request | Pydantic | 统一 SaaS 请求契约 | 缺 CV 或字段越界直接 validation error | request schema |
| 输入护栏 | `nodes.entry_guard` / `guardrails.classify_input` | 公司、岗位、JD、CV | `GuardrailResult` | 规则 guardrail | prompt injection / 敏感内容初筛 | `block` 短路；`sanitize` 后续脱敏 | `guardrail.classification/risk/reasons` |
| JD 解析 | `nodes.parse_jd` -> `jd_parser_tool` | `job_description` | `list[Requirement]` | 规则 parser | 提取岗位要求与类别 | 节点失败写 checkpoint，resume 可续跑 | `ToolTrace(jd_parser_tool)` |
| CV 解析 | `nodes.parse_cv` -> `cv_parser_tool` | `cv_text` | `CVProfile` | 规则 parser | 提取技能、项目、经历、教育 | 工具参数只记录 `cv_text_hash`，避免日志泄露全文 | `ToolTrace(cv_parser_tool)` |
| Intent Router v2 | `nodes.intent_route` -> `llm.route_intent` | request + requirements | `intent/selected_retrievers/intent_confidence/base_query?` | 规则 + 可选 LLM `Qwen/Qwen2.5-7B-Instruct` | 判断意图、选择 retriever、可选 query rewrite | 非法 JSON/label、异常、无 key 均回退规则；低置信 `<0.5` 走 `multi_source` | intent accuracy、retriever selection、confusion matrix |
| Query Rewrite | `llm.llm_rewrite_query` | request context | `base_query | None` | LLM chat | 把口语/缩写改成专业检索词 | `JOB_AGENT_LLM_REWRITE=off` 或无 key 返回 None | recall/task before-after、latency |
| Source Routing | `nodes._source_types_for_state` | `selected_retrievers + confidence + source_routing` | `source_types | None` | `RETRIEVER_SOURCE_TYPES` 映射 | 将逻辑 retriever 转成 `source_type` filter | `source_routing=False` 或低置信则 broad retrieval | source routing fallback count |
| Index 构建 | `tools.retrieval.build_default_index` | project root、CV、JD、tenant | `HybridRAGIndex` | 本地文件 + inline CV/JD | 构建默认本地知识池 | resume 时通过 `_ensure_index()` 懒加载重建 | chunk count、source_type |
| 离线入库 | `ingestion.ingest` / `/knowledge/ingest` | `IngestManifest` | Qdrant points | `decode_bytes/clean_text/chunk_text_by_section/get_embedder/QdrantVectorStore` | manifest 到 Qdrant collection | 只允许 `.txt/.md` 和 workspace 内路径；乱码打标 | chunks_upserted、bad_encoding |
| Lexical Retrieval | `HybridRAGIndex.query` | query、tenant、roles、`source_types` | `list[RetrievedChunk]` | lexical cosine + overlap | 离线/默认召回 | tenant/ACL/source_type 先过滤再打分 | recall@k、context precision |
| Dense Retrieval | `HybridRetriever.query_dense` | query、tenant、roles、`source_types` | `list[RetrievedChunk]` | Qdrant cosine + embedder | 真实 embedding 语义召回 | source filter 后候选不足时可由上层 broad fallback | recall@k、latency |
| Hybrid Retrieval | `HybridRetriever.query` | query、tenant、roles、`source_types` | fused chunks | BM25 + dense + weighted RRF + metadata boost | 生产主检索路径 | BM25/dense 两路都做 ACL/source filter | nDCG、MRR、task_success |
| Rerank | `rerank.rerank` | query + candidate chunks | reranked chunks | `none/heuristic/api/cross-encoder`；线上 `Qwen/Qwen3-Reranker-8B` | 修复候选排序 | API/cross-encoder 失败回退 heuristic | MRR/nDCG/latency |
| Post ACL Audit | `acl.audit` | reranked chunks、tenant、roles | allowed + denied | 规则 audit | 防御纵深，防止越权 chunk 泄露 | 越权 chunk 剔除并写 denied reason | `acl_audit` trace |
| Evidence Check | `nodes.evidence_check` | matches + citations | 补充 citations/matches | rewrite + retrieval loop | 证据不足时补检索 | `MAX_RETRIEVAL_ROUNDS=2` 防循环 | retrieval_rounds、evidence_sufficiency |
| Match | `cv_match_tool` | requirements + citations | `list[MatchItem]` | 规则 matcher | 判断 strong/partial/gap | 无证据只标 gap，不伪造 strong | match_level、gap |
| Report | `report_writer_tool` | company/role/matches/questions/citations | markdown report | 模板 writer | 生成带 citation 的准备报告 | 写出前 `redact_sensitive` | report_path、citation_count |
| Grounding | `check_claim_grounding` | report + citations | `GuardrailResult` | 启发式 token overlap | 防止未支撑能力声明 | unsupported claims 超阈值 block | faithfulness、unsupported_claim_rate |
| Persistence/Resume | `SessionStore` + `runner` | state + node | sessions/checkpoints/traces/cache | SQLite | session、checkpoint、tool cache、feedback | 失败节点保存 `failed_node`，`/resume` 从未完成节点继续 | node_trace、tool_trace |
| Evaluation | `evaluation.run_eval` | eval JSONL | metrics + failure files | evaluation metrics | 回归评估与数据飞轮 | 输出 failure taxonomy/regression candidates | recall/MRR/nDCG/intent/tool/ops |
| Tuning Harness | `tuning.py` + scripts | eval set + env knobs | before/after markdown/json | fusion/rerank/intent/routing compare | 做变量隔离实验 | 每个配置独立 SQLite store，避免缓存串味 | fusion_sweep/rerank/intent/routing reports |

端到端执行顺序可以简化成：

```text
PrepareRequest
-> API/Auth/Rate Limit
-> Guardrail
-> JD/CV Parse
-> Intent Router v2 + Query Rewrite
-> Source Routing
-> Retrieval (lexical/dense/hybrid)
-> Rerank
-> Post ACL Audit
-> Match + Evidence Check Loop
-> Interview Questions + Report
-> Claim Grounding
-> Persist + Trace + Eval
```

## 9. 模型与后端配置矩阵

| 类型 | 默认/候选 | 触发配置 | 用途 | 实测结论 | 选择原因 / 不选原因 |
|---|---|---|---|---|---|
| Offline embedder | `HashingEmbedder(dim=256)` | 默认 `JOB_AGENT_EMBEDDER=hashing` | 无 key/无模型时跑通 dense/Qdrant 流程 | v3 上不能提供真实语义收益 | 适合 CI、demo、单测；不用于证明语义能力 |
| API embedding | `Qwen/Qwen3-Embedding-0.6B` | `JOB_AGENT_EMBEDDER=api` + `JOB_AGENT_EMBED_MODEL` | 低成本真实 embedding | dense recall 0.5375，hybrid task 0.51，p50 ~2.5s | 性价比最好 |
| API embedding | `Qwen/Qwen3-Embedding-4B` | 同上 | 质量优先 embedding | dense recall 0.5750，nDCG 0.5265，task 0.54 | 当前默认展示质量基线；用于 fusion/routing 在线实验 |
| API embedding | `Qwen/Qwen3-Embedding-8B` | 同上 | 更大 embedding 对比 | recall 0.5575，MRR 0.3317，p50 ~7.2s | 更慢且未超过 4B，不推荐默认 |
| Local embedding | `fastembed` | `JOB_AGENT_EMBEDDER=fastembed` | 本地 ONNX embedding | 本机缺 MSVC，未跑真实基线 | 接口就绪，环境满足后可跑 |
| Local embedding | `sentence-transformers` | `JOB_AGENT_EMBEDDER=sentence-transformers` | 本地 torch embedding | 本机缺 MSVC/torch DLL，未跑 | 接口就绪，非本地验收条件 |
| Vector DB | Qdrant local | `QDRANT_PATH` 或 `:memory:` | dense index、payload filter | local 持久化重启复用有测试 | 无 Docker 可跑，面试展示成本低 |
| Sparse retrieval | BM25 Okapi | 内置 `bm25.py` | 精确词/术语召回 | 过度下调 BM25 会伤 recall | 与 dense 互补；保留 0.5 权重最优 |
| Fusion | weighted RRF | `JOB_AGENT_FUSION_*` | 融合 BM25 + dense | 最优 `BM25=0.5/dense=2.0/RRF k=30`，recall 0.478→0.575 | 分数尺度不同，用 RRF 比直接加权分数稳 |
| Reranker | heuristic | 默认 `JOB_AGENT_RERANKER=heuristic` | 离线排序 | offline 可测，但真实排序提升有限 | 无 key fallback |
| Reranker | `Qwen/Qwen3-Reranker-8B` | `JOB_AGENT_RERANKER=api` + `JOB_AGENT_RERANK_MODEL` | query-document 交互式排序 | MRR 0.385→0.643，nDCG 0.4735→0.6491，p50 1.4s→2.7s | 排序质量收益大，质量优先推荐 |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | 同上 | 轻量 rerank 对比 | MRR 0.4074，p50 ~2.55s | 提升有限且延迟接近 8B，不作为默认推荐 |
| Chat LLM | `Qwen/Qwen2.5-7B-Instruct` | `JOB_AGENT_LLM_MODEL` + `SILICONFLOW_API_KEY` | intent routing、query rewrite | intent 0.10→0.27；rewrite 后 recall 0.4783→0.9817；source+rewrite 40 条 recall 0.6208 | 高杠杆用于路由/改写，不替代整条链路 |
| Cache | SQLite embedding cache | `JOB_AGENT_EMBED_CACHE_PATH` | `(model,text_hash)->vector` | 跨实例 0 重算单测通过 | 修复 retrieval cache 命中但 embedding 重算问题 |
| Cache/Rate limit | in-memory / Redis | `REDIS_URL` | cache、rate limit | Redis 抽象就绪，未跑真实 Redis 基线 | 本地默认轻量；生产可切 Redis |
| Storage | SQLite SessionStore | `JOB_AGENT_DB_PATH` | session/checkpoint/tool_cache/feedback | resume、traces、cache 有测试 | 本地可复现，适合 portfolio |

关键环境变量：

| 变量 | 示例 | 作用 |
|---|---|---|
| `JOB_AGENT_EMBEDDER` | `hashing` / `api` / `fastembed` / `sentence-transformers` | 选择 embedding 后端 |
| `JOB_AGENT_EMBED_API_BASE` | `https://api.siliconflow.com/v1` | API embedding endpoint |
| `JOB_AGENT_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | API embedding 模型 |
| `JOB_AGENT_RERANKER` | `heuristic` / `api` / `cross-encoder` / `none` | 选择 rerank 后端 |
| `JOB_AGENT_RERANK_MODEL` | `Qwen/Qwen3-Reranker-8B` | API rerank 模型 |
| `JOB_AGENT_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | chat LLM 模型 |
| `JOB_AGENT_LLM_REWRITE` | `on` / `off` | 消融 query rewrite |
| `JOB_AGENT_FUSION_BM25_WEIGHT` | `0.5` | BM25 权重 |
| `JOB_AGENT_FUSION_DENSE_WEIGHT` | `2.0` | dense 权重 |
| `JOB_AGENT_FUSION_RRF_K` | `30` | RRF 平滑项 |
| `QDRANT_PATH` | `reports/qdrant` | Qdrant local 持久化路径 |
| `REDIS_URL` | `redis://...` | Redis cache/rate limit 后端 |

## 10. 工程更迭记录

| 阶段 | 原问题 | 核心改动 | 接口/参数变化 | 实验结果 | 结论 |
|---|---|---|---|---|---|
| Phase 1 | 单节点 wrapper 不利于 trace/resume | 多节点 runner + StateGraph | `entry_guard -> ... -> persist`，`NodeTrace`，checkpoint | job_agent 20 tests，unit 101 passed | 先把 Agent 从 demo 变成可观测 workflow |
| Phase 2 | 检索仍偏本地 lexical，评估不真实 | Qdrant、BM25+dense、RRF、150 eval | `/knowledge/ingest`、`retrieval_mode`、`/eval/run` | 150 v2：recall 1.0、task 0.98 | 工程链路跑通，但 v2 过于结构化 |
| Phase 3 | v2 指标虚高，缺语义难例 | v3 100 hard cases、dense mode、真实 embedder 接口 | `eval_dataset_v3`、`retrieval_mode=dense` | hashing 下 recall ~0.47、intent ~0.1 | 难例暴露真实短板 |
| Phase 3.1 | 需要真实 embedding baseline | SiliconFlow Qwen3 embedding 对比 | `JOB_AGENT_EMBEDDER=api` | 4B dense recall 0.575；8B 更慢且未超过 4B | 不盲目追大模型，转向融合/排序 |
| Phase 4 | 等权 hybrid 稀释 dense，embedding 重算慢 | embedding cache、weighted RRF、ACL audit、fusion sweep | `EmbeddingCache`、`FusionConfig`、`acl.audit` | tuned hybrid recall 0.478→0.575，task 0.35→0.54 | 调权比堆模型更有效 |
| Phase 5.1 | fusion 提升 recall 但 MRR 低 | rerank harness + API reranker | `JOB_AGENT_RERANKER=api` | 8B reranker MRR 0.385→0.643，nDCG 0.4735→0.6491 | 候选池已有好证据，瓶颈转为排序 |
| Phase 5.2 | 规则 intent routing 难例上崩 | LLM intent + query rewrite + A/B | `llm_mode=auto`、`JOB_AGENT_LLM_MODEL` | intent 0.10→0.27，recall 0.4783→0.9817，task 0.35→0.84 | 最大收益来自 query rewrite |
| Phase 5.2 消融 | LLM routing 和 rewrite 贡献混在一起 | 禁用 rewrite，只保留 LLM routing | monkeypatch / `JOB_AGENT_LLM_REWRITE=off` | intent 0.10→0.24，recall/task 不变 | intent label 不接检索不会影响候选池 |
| Phase 5.3 | `selected_retrievers` 只是观测字段 | source-level routing + fallback + confusion matrix | `source_routing`、`source_types`、`MIN_SOURCE_EVIDENCE=3` | offline recall 0.435→0.4883，task 0.34→0.38 | routing 真正接入候选池后有净收益 |
| Phase 5.3 online | 想验证 source+rewrite 在线叠加 | 40 条线上 4-way ablation | `routing_compare.py --limit 40` + chat key | source+rewrite recall 0.6208，task 0.60，p50 6.28s | 有增益但延迟高，需按需启用 |

## 11. 代码文件到能力映射

| 文件 | 关键接口 | 负责能力 |
|---|---|---|
| `schemas.py` | `PrepareRequest/PrepareResponse/NodeTrace/Metrics` | API 和 workflow 数据契约 |
| `entrypoint.py` | `/prepare`、`/prepare/stream`、`/sessions/{id}/resume`、`/knowledge/ingest`、`/eval/run` | FastAPI 服务入口 |
| `nodes.py` | `PIPELINE`、`intent_route`、`_run_retrieval`、`evidence_check` | 多节点 Agent 执行链路 |
| `runner.py` | `run_pipeline`、`resume_application` | checkpoint、resume、response assemble |
| `graph.py` | `StateGraph` wrapper | LangGraph 可视化/编排 |
| `llm.py` | `route_intent`、`llm_rewrite_query`、`source_types_for`、`_chat_completion` | Intent Router v2、query rewrite、source routing 映射 |
| `rag.py` | `HybridRAGIndex.query`、`chunk_text_by_section` | 离线 lexical RAG 和 chunking |
| `hybrid.py` | `HybridRetriever.query/query_dense`、`FusionConfig` | BM25+dense+RRF 混合检索 |
| `vectorstore.py` | `QdrantVectorStore` | Qdrant local/server dense store |
| `embeddings.py` | `get_embedder`、`APIEmbedder`、`EmbeddingCache` | embedding 后端和向量缓存 |
| `rerank.py` | `rerank`、`_api_rerank` | heuristic/API/cross-encoder rerank |
| `acl.py` | `audit` | post-retrieval ACL defense-in-depth |
| `guardrails.py` | `classify_input`、`check_claim_grounding` | input/output guardrails |
| `session.py` | `SessionStore` | session、trace、checkpoint、tool cache、feedback |
| `ingestion.py` | `IngestManifest`、`build_chunks`、`ingest` | manifest 到 Qdrant 入库 |
| `evaluation.py` | `run_eval`、`evaluate_case`、`intent_confusion` | 指标、失败分类、混淆矩阵 |
| `tuning.py` | `run_fusion_sweep`、`run_rerank_compare`、`run_intent_compare`、`run_routing_compare` | A/B 和消融实验 |
| `scripts/*.py` | `fusion_sweep.py`、`rerank_compare.py`、`intent_compare.py`、`routing_compare.py` | 可复现实验 CLI |

## 12. 简历可用句（中/英）

- 设计并实现准生产级 Agentic RAG 服务：多节点 LangGraph + 确定性 runner（checkpoint/resume），
  Qdrant 混合检索（BM25 + dense + RRF + rerank），多租户 ACL，claim-level grounding，
  150 条评估集（recall@k/nDCG/faithfulness/intent 等真实指标），全程本地可跑、无 key 降级。
- Built a production-style Agentic RAG service: multi-node LangGraph + deterministic runner with
  checkpoint/resume; hybrid retrieval (BM25 + dense + RRF + rerank over Qdrant) with multi-tenant ACL;
  claim-level grounding to prevent fabricated experience; a 150-case eval harness with real
  recall@k / nDCG / faithfulness / intent metrics and a failure-taxonomy feedback loop — all runnable
  offline with graceful degradation when no API key is present.
- 用 100 条语义难例 + 权重网格搜索，把混合检索从「等权融合」调优到 `BM25=0.5/dense=2.0/RRF k=30`，
  在 Qwen3-Embedding-4B 下将 recall@k 0.478→0.575、task_success 0.35→0.54，并定位 MRR 未提升→
  rerank 为下一优化点（数据驱动、可复现、含成本/延迟权衡）。
- Tuned hybrid retrieval from naive equal-weight RRF to `BM25=0.5/dense=2.0/RRF k=30` via a weight
  grid-search over 100 hard semantic cases, lifting recall@k 0.478→0.575 and task-success 0.35→0.54
  on Qwen3-Embedding-4B, and identified flat MRR as the next (reranking) lever — a reproducible,
  metric-driven retrieval-tuning story with explicit latency/cost trade-offs.
- 在固定最优 fusion 后接入 SiliconFlow `Qwen/Qwen3-Reranker-8B`，将排序指标 MRR 0.385→0.643、
  nDCG 0.4735→0.6491，同时保持 recall@k 0.575 不变，证明正确证据已在候选池中、瓶颈转为排序。
- Added an API rerank stage after tuned fusion using `Qwen/Qwen3-Reranker-8B`, improving MRR
  0.385→0.643 and nDCG 0.4735→0.6491 while keeping recall@k at 0.575, showing that the evidence
  was already retrieved and the bottleneck had shifted to ordering.
