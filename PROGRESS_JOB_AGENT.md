# Job Agent — 工业级增强进度日志

> 准生产 SaaS 升级计划的执行记录。区分「已实现」与「待实现」，所有「已实现」项均有测试支撑。

## Phase 1 — 已完成（2026-06-09）

本轮在**不引入重型外部依赖**（Qdrant / Redis / cross-encoder）的前提下，完成了计划中
可本地运行、可被测试证明的核心骨架。

**测试口径（精确）**：
- 基线 10 个 → 现 **20 个 job_agent 相关测试全绿**。
- **完整 unit tests 全绿**（`pytest tests/unit_tests -q` → 101 passed），`ruff check .` 通过。
- ⚠️ 全量 `pytest -q` 含 e2e（118 passed / 11 failed / 8 errors）：失败/报错来自需要外部服务、
  API key 或 LangGraph dev server 的 e2e 用例，**不属于 job_agent Phase 1 范围**，
  未作为本地验收条件。

### 1. 架构：单节点 wrapper → 显式多节点图 ✅
- 新增 `src/job_agent/nodes.py`：显式节点
  `entry_guard → parse_jd → parse_cv → retrieve → match → evidence_check
  → generate_questions → write_report → grounding_check → persist`。
- 新增 `src/job_agent/runner.py`：顺序编排器，每个节点写入结构化 `NodeTrace`
  （status / latency_ms / checkpoint），并在每个节点后落 checkpoint。
- `src/job_agent/graph.py` 重写为**真正的多节点 `StateGraph`**（LangGraph Studio 可视），
  与 runner 复用同一批节点函数；`entry_guard` 条件路由到 `__end__`（hard block 时）。
- `workflow.prepare_application` 改为委托 runner，**对外契约不变**（旧测试继续通过）。
- 保留 deterministic：无 LLM key 即可全程跑通（规则路由 / 模板 query / 启发式 match）。
- ⚠️ **LangGraph 与 runner 的分工（面试口径）**：节点抽象为 LangGraph-compatible node，
  `graph.py` 是真实可视化 / 可编排的 `StateGraph`；但 **API 实际执行、checkpoint 与 resume
  的生产恢复逻辑在 `runner.py`**。这样既能展示图编排，又能保证无外部服务时确定性可测。

### 2. 状态恢复 / Checkpoint / Resume ✅
- `SessionStore` 新增 `checkpoints` 表 + `save_checkpoint` / `get_latest_checkpoint`
  / `get_completed_nodes`。
- 节点失败时记录 `failed_node`、`status=failed` 并落 checkpoint；
  `resume_application()` 从「最近未完成节点」继续（index 等不可序列化对象会按需重建）。
- 集成测试：注入瞬时失败 → `/resume` 从失败节点续跑直至 `completed`。

### 3. 权限隔离（多租户 + ACL）✅
- `PrepareRequest` 新增 `tenant_id` / `user_id` / `user_roles` / `retrieval_mode`
  / `llm_mode`（全部带默认值，**向后兼容**旧请求）。
- `HybridRAGIndex` 的 chunk payload 写入 `tenant_id` + `acl_roles`；
  `query()` 在**打分前强制执行 tenant 隔离 + ACL 硬过滤**（pre-scoring hard filter）。
- ⚠️ 这是打分前硬过滤，**不是独立的 post-retrieval ACL audit 阶段**；后者留待 Phase 2。
- 请求自带的 CV/JD 文档限定 `tenant:{tenant_id}` 角色；workspace 公共知识为 `public`。

### 4. Guardrail 升级 ✅
- 输入 guardrail 从 binary blocked → **四分类 `pass / sanitize / clarify / block`**。
- 新增 `redact_sensitive()`：API key / token / system prompt / 内部路径脱敏。
- 输出 grounding 从「是否含 citation」→ **启发式 claim-level**：对报告中的能力声明行，
  用 citation 标记 / token overlap 判断是否被证据支持，未支撑的进入 `unsupported_claims`；
  ≥3 条触发 block。⚠️ 这是启发式版本，**非语义级 NLI / LLM verifier**（Phase 2 升级）。
- 结构上防伪造：matcher 只在有证据时标 strong/partial，否则标 gap。

### 5. Agentic Retrieval（轻量闭环）✅
- 已实现**轻量 Agentic Retrieval 闭环**：证据不足检测 → 规则式补充查询（追加 gap 需求词）
  → 补充检索 → 去重合并 → 重新匹配；受 `MAX_RETRIEVAL_ROUNDS` 限制（默认 2 轮）。
- ⚠️ 尚**未**拆出独立的 intent routing / query rewrite / retriever selection 节点，
  也未接 Qdrant / BM25 / dense / RRF / cross-encoder（均为 Phase 2）。
- `Metrics` 新增 `retrieval_rounds` / `cache_hits`。

### 6. 工具缓存抽象 ✅ / 节点默认启用 ⚠️待补
- SQLite tool cache 抽象已实现：`_trace_tool` 支持 `tool_name + args_hash → cached observation`，
  命中时写 trace 并记 `cache_hits`。
- ⚠️ Phase 1 **关键工具节点尚未默认传 `use_cache=True`**，因此实际运行中 `cache_hits` 基本不增长。
- Phase 2：在 retrieval / tool 节点接入缓存策略，并加 Redis 后端。

### 7. API 扩展 ✅（旧路由全部保留）
- `GET  /sessions/{id}/traces` — tool traces + 已完成节点。
- `POST /sessions/{id}/resume` — 从 checkpoint 恢复。
- `POST /eval/run` — 触发离线评估，返回 summary。

### 测试
- 新增 `tests/unit_tests/test_job_agent_governance.py`（分类 / 脱敏 / claim grounding
  / 租户 ACL / 节点 trace+checkpoint / block 短路 / resume）。
- 新增 `tests/integration_tests/test_job_agent_resume.py`（新路由注册 / 失败→resume）。

---

## Phase 2 — 已完成（2026-06-09）

**测试口径（精确）**：job_agent 相关测试 20 → **38 全绿**（Phase 1 的 20 + Phase 2 的
14 unit + 4 API）。完整 `pytest tests/unit_tests -q` → **115 passed**；`ruff check .` 通过。
全量 e2e 仍依赖外部服务/API key，未作本地验收条件。所有 Phase 2 能力默认 **deterministic
offline**：用 feature-hashing embedder + 本地 Qdrant，无需 Docker / API key / 模型下载。

### 1. Qdrant 向量库 + ingestion pipeline ✅
- `vectorstore.py`：`QdrantVectorStore`，默认 Qdrant 本地模式（`:memory:` 测试 / 路径持久化）。
  tenant 隔离 + ACL 用原生 Qdrant payload filter（`must` tenant + `should` MatchAny(acl_roles)）。
- `ingestion.py`：manifest → parse/clean → structure-aware chunk → metadata enrichment →
  embedding → upsert。统一 payload（tenant_id/doc_type/source_path/section/topic/company/role/
  chunk_id/updated_at/acl_roles）。`/knowledge/ingest` 端点。
- ✅ 持久化重启复用：测试 `test_ingestion_persists_and_reloads`（写入→重开 collection→点数一致）。

### 2. BM25 + dense + RRF 混合检索 ✅
- `bm25.py`：无依赖 BM25 Okapi。`embeddings.py`：deterministic `HashingEmbedder`（可插拔，
  `JOB_AGENT_EMBEDDER` 可切真实 sentence-transformers）。
- `hybrid.py`：BM25(sparse) + Qdrant(dense) 用 **RRF** 融合 + metadata boost，两路都强制 tenant/ACL。
- `retrieval_mode="hybrid"` 已在 retrieve / evidence_check 节点接通；`deterministic` 仍为默认且向后兼容。

### 3. Cross-encoder rerank（两层）✅
- `rerank.py`：默认启发式 rerank（score+overlap+coverage 混合），`JOB_AGENT_RERANKER=cross-encoder`
  可切真实 cross-encoder（失败自动降级）。检索后统一过一层 rerank。

### 4. Agentic routing：intent + retriever selection ✅
- `llm.py`：deterministic intent 分类（8 类）+ retriever selection（6 个 logical retriever）+
  query rewrite。新增 `intent_route` 节点；response 暴露 `intent` / `selected_retrievers`。
- `llm_mode="auto"` + 配置 key 时可切真实 LLM routing（失败降级）。

### 5. 工具缓存真正启用 + Redis 抽象 ✅
- `cache.py`：`Cache` 抽象（in-memory 默认 + `REDIS_URL` 切 Redis），用于限流 + 结果缓存。
- 检索节点 `use_cache=True` **已默认启用**；序列化修正后 `cache_hit_rate` 真实增长（eval 实测 ~0.14）。

### 6. 服务端 tool executor：timeout + retry + fallback ✅
- `tool_executor.py`：线程级 timeout + bounded retry + fallback。检索调用接 `retries=1, fallback=[]`。

### 7. `/prepare/stream` ✅
- 先发 report skeleton（sections）再发完整 result（NDJSON）；block 输入直接发 blocked 事件。

### 8. 150 条 regression set + 真实指标 + 失败闭环 ✅
- `eval_dataset.py` 生成 **150 条**标注用例（10 类，按计划分布），标注 `expected_intent /
  expected_retrievers / gold_evidence / must_answer_points / must_not_claim / guardrail_expected /
  relevance_grade`。文件：`data/eval/job_agent_eval_v2.jsonl`。
- `evaluation.py` 重写为真实指标：
  retrieval（recall@k / MRR / nDCG@k / context precision / source coverage）、
  answer（citation accuracy / faithfulness / unsupported-claim rate / answer coverage / must-not 违例）、
  agent（intent acc / retriever selection acc / tool-call validity / trajectory correctness）、
  ops（TTFT / p50 / p95 / cache hit rate / token cost）。
- 失败 taxonomy + regression candidates + `before_after.md`（deterministic vs hybrid A/B）。

**150 条实测 summary**（deterministic hashing embedder）：

| 指标 | 值 | 指标 | 值 |
|---|---|---|---|
| recall@k | 1.0 | faithfulness | 1.0 |
| MRR | 1.0 | citation_accuracy | 0.987 |
| nDCG@k | 0.998 | intent_accuracy | 0.933 |
| context_precision | 0.868 | retriever_selection_acc | 0.933 |
| task_success_rate | 0.98 | guardrail_correct | 0.987 |
| latency_p50 / p95 (ms) | 176 / 324 | cache_hit_rate | 0.137 |

### eval 驱动的真实 before/after 改进（本轮）
- **faithfulness 0.013 → 1.0**：150 条 eval 暴露 claim-grounding 误报（把面试题 prompt 当能力声明），
  修正为只扫 JD-CV match 区块。
- **intent_accuracy 0.70 → 0.933**：eval 暴露 `required/requirements` 未触发 jd_cv_match，且对抗样本
  不应计 intent，修正后提升。
- ⚠️ deterministic vs hybrid A/B：本合成集上 lexical 已 recall=1.0，hybrid 仅 nDCG +0.002 且增延迟；
  hashing embedder 非语义，dense 优势要等**真实 embedder**（Phase 3）才显现——已如实写入 `before_after.md`。

### 测试
- 新增 `tests/unit_tests/test_job_agent_phase2.py`（embedder / BM25 / Qdrant ACL / RRF / hybrid /
  rerank / ingestion 持久化 / tool executor / cache / intent routing / eval 指标 / 150 条跑通）。
- 新增 `tests/integration_tests/test_job_agent_phase2_api.py`（stream / knowledge ingest / eval v2 路由）。

---

## Phase 3 — 已完成（2026-06-09 续）

**测试口径**：新增 `tests/unit_tests/test_job_agent_phase3.py`（6 项）；job_agent 相关测试
38 → **44 全绿**；完整 `pytest tests/unit_tests`（含全项目）→ **121 passed**（+eval smoke = 122）；
`ruff check .` 通过。所有新增能力默认仍 deterministic offline。

### 1. 真实 embedder 后端（3 路）+ 降级 ✅
- `embeddings.py` 三个可插拔后端，统一 `Embedder` 协议，失败自动回退 hashing：
  - `fastembed`（ONNX，无需 torch；Windows 需 MSVC 运行库）
  - `api`（OpenAI 兼容 `/embeddings`，如 SiliconFlow **bge-m3**，配 key 即用）
  - `sentence-transformers`（本地 torch）
- ⚠️ **本机环境限制**：缺 MSVC 运行库，`onnxruntime` / `torch` DLL 无法加载，**真实 embedder
  本地基线未能跑出**；不伪造语义提升。已在 README 写明启用方式（装 MSVC 运行库 或 配 API key）。

### 2. dense-only 检索模式 + 三方对比 ✅
- `retrieval_mode` 增加 `dense`（纯向量）；`hybrid.py` 加 `query_dense()`。
- `evaluation.py` 的 compare 升级为 **lexical vs dense vs hybrid 三方**，输出 `before_after.md`。
- 单测用 `_SynonymEmbedder` 桩**证明机制**：语义 embedder 下 dense 能召回改写证据、lexical 不能。
- **v3 三方实测（hashing embedder）**：recall@k lexical 0.476 / dense 0.44 / hybrid 0.435；
  MRR 0.394 / 0.426 / 0.339；intent_accuracy 全 **0.1**。诚实结论：① 难例确实把 recall 从 v2 的 1.0
  打到 ~0.47（集子是真难）；② hashing 下 dense≈lexical（符合预期，语义增益需真实 embedder）；
  ③ 规则 intent 路由在口语输入上**崩到 0.1**——是接真实 LLM routing 的明确目标。已写入 `before_after.md`。

### 3. 语义难例集 v3（100 条）✅
- `eval_dataset_v3.py` → `data/eval/semantic_hard_eval_v3.jsonl`：刻意制造 query↔证据**词汇 gap**
  （口语 vs 专业、TTFT vs 首token延迟、同义、多跳、无答案、injection、中英混合）。
- ⚠️ 用 hashing embedder 跑时 dense 仍非语义，三方差异有限；该集的价值在**配真实 embedder 后**才完全体现。

### 4. ingestion 编码清洗 + source quality ✅
- `encoding.py`：UTF-8 BOM / GB18030 稳健解码 + mojibake 检测（U+FFFD / CP1252 伪字符密度 + 无 CJK）。
- ingestion 对疑似乱码打 warning，chunk metadata 写入 `encoding` + `bad_encoding` 标志（eval 可消费）。

### 5. 一键 Demo ✅
- `scripts/demo.py`（ingest→prepare→resume→eval）、`scripts/job_agent_demo.ps1`、
  `README_JOB_AGENT_DEMO.md`（含 curl 示例 + 5 分钟讲稿 + 真实模型切换说明）。

---

## Phase 3.1 — 真实 API Embedding 基线（2026-06-10）

**数据集**：`data/eval/semantic_hard_eval_v3.jsonl`（100 条语义难例）。
**服务**：SiliconFlow OpenAI-compatible `/embeddings`，`https://api.siliconflow.com/v1`。
**说明**：API key 仅通过 PowerShell 环境变量传入，未写入仓库。

### 1. Qwen3 Embedding 三模型对比 ✅

| 模型 | 最佳模式 | recall@k | nDCG@k | MRR | task_success | p50 latency | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-0.6B | hybrid / dense | dense 0.5375 / hybrid 0.5292 | dense 0.4573 / hybrid 0.4373 | dense 0.3333 / hybrid 0.3643 | dense 0.48 / hybrid 0.51 | ~2.5s | 性价比最好 |
| Qwen3-Embedding-4B | dense | 0.5750 | 0.5265 | 0.4081 | 0.54 | ~4.0s | 当前质量最佳 |
| Qwen3-Embedding-8B | dense | 0.5575 | 0.4624 | 0.3317 | 0.51 | ~7.2s | 不推荐默认使用 |

### 2. 关键工程结论

- **真实 embedding 有收益**：相比 lexical baseline（recall@k 0.4758、task_success 0.37），
  4B dense 将 recall@k 提升到 0.5750，task_success 提升到 0.54。
- **模型越大不一定越好**：8B 比 4B 慢很多，但指标没有超过 4B。
- **hybrid 当前不是最终最优**：强 dense 模型接入后，BM25/RRF/启发式 rerank 反而会稀释 dense 结果。
  下一步应调 fusion weight、metadata boost、rerank 位置和 top-k，而不是盲目换更大 embedding。
- **缓存结论已核对**：8B run 中 retrieval tool cache hit rate 约 0.16，命中工具为
  `dense_retrieval_tool` / `hybrid_retrieval_tool` / `rag_retrieval_tool`。
  但该缓存位于 retrieval result 层，不缓存 embedding vector；`HybridRetriever(index)` 构建阶段仍可能发起远程 embedding。
  因此后续要补 `(model, text_hash) -> vector` 的 embedding cache，或持久化每个模型的 Qdrant index。
- **下一短板**：v3 语义难例的规则 intent routing 仍约 0.1，需要接 LLM routing 或更强 intent classifier。

---

## Phase 4 — 待实现

| 计划项 | 状态 | 说明 |
|---|---|---|
| 真实 embedder 本地基线 | ⚠️ 抽象就绪/环境受阻 | 代码三路就绪；本机缺 MSVC 运行库无法跑 ONNX/torch，需装运行库或用 API key。 |
| 语义级 faithfulness（NLI / LLM verifier） | ❌ | 现 grounding 为启发式 token-overlap。 |
| 独立 post-retrieval ACL audit 阶段 | ❌ | 现为打分前硬过滤（Qdrant filter）。 |
| Redis 实跑基线 / 分布式限流压测 | ⚠️ 抽象就绪 | 代码路径已就位，未起 Redis 实测。 |
| 简历 bullet 同步 | ⚠️ | 见 `TECH_REPORT_JOB_AGENT.md`「简历可用句」，docx 需手动落。 |

## 验收对照（累计）
- ✅ 现有核心测试继续通过（10 → 44）。
- ✅ 本地无外部 key 可跑通 ingest / prepare / resume / eval（全 deterministic）。
- ✅ Qdrant local 持久化重启复用（测试覆盖）。
- ✅ 150 条 eval 一键运行，输出 metrics.csv / failure_cases.md / regression_set.jsonl /
  summary.json / before_after.md，指标真实对应。
- ✅ eval 报告展示真实 before/after 改进（faithfulness、intent）。
- ✅ session 中断后 `/resume` 从失败节点继续。
