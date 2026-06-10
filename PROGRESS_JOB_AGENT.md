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

## Phase 4 — 已完成（针对 3.1 暴露的瓶颈，2026-06-10）

**测试口径**：新增 `tests/unit_tests/test_job_agent_phase4.py`（4 项）+ `test_job_agent_tuning.py`（1 项）；
job_agent 相关测试 44 → **49 全绿**；完整 `pytest tests/unit_tests` → **126 passed**；`ruff check .` 通过。
全部**离线可测**，不依赖 API key。

### 1. Embedding / Vector Cache ✅（直接修 3.1 暴露的 bug）
- `embeddings.py` 新增 `EmbeddingCache`（SQLite，key=`sha1(model_name + text)` → vector）+
  `CachingEmbedder` 包装层；`get_embedder()` 对**真实后端默认启用缓存**（hashing 默认不缓存，可
  `JOB_AGENT_EMBED_CACHE=on` 强开），路径 `JOB_AGENT_EMBED_CACHE_PATH`。
- 每个 embedder 带稳定 `name`（`api:Qwen3-Embedding-4B` 等）做缓存键，避免跨模型串味。
- 效果：`HybridRetriever(index)` 重建 / eval 多轮**只对 cache miss 调用远程 embedding**——
  正是 3.1「retrieval cache 命中但 embedding 仍重算」的修复。测试证明跨实例命中 0 次重算。

### 2. 可调加权 Hybrid Fusion ✅（针对 dense-only > hybrid）
- `hybrid.py` 新增 `FusionConfig`（`bm25_weight` / `dense_weight` / `rrf_k` /
  `metadata_boost_weight` / `sparse_top_k` / `dense_top_k` / `final_top_k`），全部可经
  `JOB_AGENT_FUSION_*` 环境变量覆盖；`reciprocal_rank_fusion` 升级为**加权 RRF**。
- 默认仍为平衡权重（保持旧行为）。诚实口径：**让 hybrid 真正超过 dense-only 的最优权重需要你用
  真实 embedding + v3 调出来**——本轮提供的是「可调的旋钮 + 加权融合机制」，不是已调好的结论。

### 3. 独立 post-retrieval ACL audit ✅
- `acl.py`：检索 + rerank 后**再独立审计一遍** tenant/role，越权 chunk 被剔除并产出结构化
  `denied` 记录（`tenant_mismatch` / `role_denied`）写入 trace（`acl_audit`）。
- 与打分前硬过滤形成 defense-in-depth：正常流程是 no-op，但任何过滤 bug 导致的跨租户泄漏都会被
  这层拦下并留痕。测试用混合 chunk 证明能抓到两类越权。

### 4. Fusion 权重扫描工具 ✅（让 4.2 的旋钮可被实证调出）
- `tuning.py` + `scripts/fusion_sweep.py`：对 `JOB_AGENT_FUSION_*` 做网格搜索，每组在 v3 上跑
  **hybrid** 模式，按目标指标（默认 recall@k）排序，输出 `fusion_sweep.md`（含最优 env + 复现命令）。
- 工程细节：**每组独立 retrieval-result store**（避免跨配置缓存串味）+ **共享 embedding cache**
  （向量只算一次）+ 跑完**还原 env**。离线可跑（hashing），配 `JOB_AGENT_EMBEDDER=api` 即真实调优。
- **离线信号**：平衡权重在小样本上排名垫底，印证 3.1「等权融合稀释 dense」的判断。

### 4.1 真实调优结果（Qwen3-Embedding-4B，2026-06-10）✅
- 100 条 v3、6 组配置、目标 recall@k（报告：`reports/fusion_sweep_qwen4b_real/fusion_sweep.md`）。
- **最优 = `BM25=0.5 / dense=2.0 / RRF k=30`**，相对 equal-weight：

  | 配置 | recall@k | nDCG | MRR | task_success | p50 |
  |---|---:|---:|---:|---:|---:|
  | 调权 hybrid（最优） | **0.575** | 0.456 | 0.352 | **0.54** | 1276ms |
  | equal-weight（bm25=dense=1.0） | 0.478 | 0.446 | 0.370 | 0.35 | 1655ms |

- **结论**：真实 embedding 下，调权把 hybrid recall@k 0.478 → **0.575**、task_success 0.35 → **0.54**，
  **实证「hybrid 需要调权才超过等权融合」**（不是一融合就更好）。
- ⚠️ **诚实 nuance**：① 这是按 recall@k 排名；最优 hybrid 的 **MRR(0.352) 仍略低于等权(0.370)**，
  说明**排序质量/rerank 是下一杠杆**（Phase 5）。② 过度下调 BM25（`bm25=0.3`）recall 反跌到 0.407，
  甜点是「偏向 dense 但不杀死 BM25」。
- embedding cache 生效：`reports/embedding_cache_qwen4b.sqlite`（188 rows）跨配置复用向量。
- 产物治理：大 sqlite（sweep_*.sqlite 各 ~50MB、cache ~11MB）已被 `.gitignore` 排除，仓库只留结论。

---

## Phase 5.1 — Rerank Eval Harness（2026-06-10）✅

**动机**：fusion tune 把 recall 拉上来了（0.478→0.575），但 MRR 反降（0.370→0.352）——问题从
「找不找得到」变成「能不能排到前面」。rerank 正对症。

- **rerank 后端扩展**（`rerank.py`）：`JOB_AGENT_RERANKER` ∈ `none`（直通=fusion 原序，A/B 基线）/
  `heuristic`（默认离线）/ **`api`**（SiliconFlow `/rerank`，如 `Qwen/Qwen3-Reranker-0.6B/8B`，**走 API key、
  不需 torch/MSVC**）/ `cross-encoder`（本地 torch）。全部失败回退 heuristic。
- **Rerank A/B 工具**（`tuning.run_rerank_compare` + `scripts/rerank_compare.py`）：**固定最优 fusion**
  （`BM25=0.5/dense=2.0/RRF k=30`），只换 reranker，输出 `rerank_before_after.md/json`，按 MRR 排名，
  报告 MRR/nDCG/context_precision/recall/task_success/p50。工程同 fusion sweep：每策略独立 store、
  共享 embedding cache、跑完还原 env。
- **离线信号**（hashing）：`heuristic` MRR 0.211 > `none` 0.186——证明 harness 测的是排序质量，方向正确。
- **真实 rerank 结果（SiliconFlow，Qwen3-Embedding-4B + 最优 fusion，100 条 v3）**：

  | reranker | MRR | nDCG | recall@k | task_success | p50 |
  |---|---:|---:|---:|---:|---:|
  | none（fusion 原序） | 0.385 | 0.4735 | 0.575 | 0.54 | 1385ms |
  | Qwen3-Reranker-0.6B | 0.4074 | 0.4809 | 0.575 | 0.54 | 2549ms |
  | Qwen3-Reranker-8B | **0.643** | **0.6491** | 0.575 | 0.54 | 2698ms |

- **结论**：8B reranker 明显修复排序质量，MRR 0.385→0.643、nDCG 0.4735→0.6491；
  recall/task_success 不变符合预期，因为 rerank 只重排候选池，不新增候选。0.6B 提升有限且延迟接近 8B，
  因此当前质量优先默认推荐 `Qwen/Qwen3-Reranker-8B`。

### 变更历史与参数原因（Phase 3.1 → 5.1）

| 时间点 | 发现的问题 | 改动 | 参数 / 接口 | 为什么这样改 | 验证结果 |
|---|---|---|---|---|---|
| Phase 3.1 | hashing 不能证明语义收益 | 接 SiliconFlow API embedding | `JOB_AGENT_EMBEDDER=api`、`.com /v1` endpoint、`Qwen3-Embedding-0.6B/4B/8B` | 用真实 embedding 验证 v3 语义难例，而不是拿 MTEB 或假向量讲故事。 | 4B dense recall@k 0.575、task_success 0.54；8B 更慢且没超过 4B。 |
| Phase 3.1 | retrieval cache 命中但重建 retriever 仍重算 embedding | 增加 embedding cache | `sha1(model_name + "\0" + text) -> vector`，路径 `JOB_AGENT_EMBED_CACHE_PATH` | retrieval result cache 和 embedding vector cache 是两层；前者缓存工具结果，后者减少远程 embedding 调用。 | Phase 4 单测证明跨实例 0 次重算。 |
| Phase 4 | 等权 hybrid 被 dense 稀释 | 加权 RRF + sweep | `bm25_weight/dense_weight/rrf_k/top_k/meta_boost` | BM25 分数和 cosine 分数不可直接相加；RRF 用排名融合，但真实业务仍要调权。 | 最优 `BM25=0.5/dense=2.0/RRF k=30`，recall 0.478→0.575。 |
| Phase 4 | ACL 只靠检索前过滤，不够防御纵深 | post-retrieval ACL audit | `audit(chunks, tenant_id, user_roles) -> allowed, denied` | 如果上游 filter 写错，最终结果层仍能剔除越权 chunk 并留下 denied reason。 | 单测覆盖 `tenant_mismatch` 和 `role_denied`。 |
| Phase 5.1 | fusion 提升 recall，但 MRR 下降 | 增加 rerank A/B harness | 固定最优 fusion，只改变 `JOB_AGENT_RERANKER` | 隔离变量，判断排序器是否真的提升 MRR/nDCG，而不是被召回变化干扰。 | Qwen3-Reranker-8B：MRR 0.385→0.643、nDCG 0.4735→0.6491。 |

面试口径：这个项目不是“一次性堆模块”，而是先用评估集暴露瓶颈，再只改一个变量做 before/after。embedding 解决语义召回，fusion 解决多路召回融合，rerank 解决候选排序，ACL/cache/checkpoint 解决 SaaS 工程治理。

---

## Phase 5.2 — LLM Intent Routing + Query Rewrite（2026-06-10）✅

**动机**：v3 难例上规则 intent routing ≈0.1（口语/缩写/同义把关键词匹配打穿）。把**单个决策节点**
交给 LLM 是模型最该放的位置——不是盲目把整条链路换大模型。

- **LLM 决策接入**（`llm.py`）：统一 OpenAI 兼容 chat seam `_chat_completion`（默认 SiliconFlow
  `.com /v1`，`JOB_AGENT_LLM_MODEL` 如 `Qwen/Qwen2.5-7B-Instruct`）。`llm_mode=auto` + key 时：
  - `classify_intent`：LLM 判 8 类意图，非法 label / 异常 → **回退规则分类**；
  - `llm_rewrite_query`：把口语/缩写 query 改写成专业英文检索词（TTFT→time to first token），
    失败 → 回退 deterministic query。
- **管线接入**：`intent_route` 节点一次性算出 intent + （LLM 模式下）`base_query`，存入 state；
  `retrieve` / `evidence_check` 用 `_compose_query` 复用该 base_query（**每请求仅 1 次 LLM 改写**）。
  `llm_mode=off`（默认）行为与之前完全一致。
- **Intent A/B 工具**（`tuning.run_intent_compare` + `scripts/intent_compare.py`）：`off`(规则) vs
  `auto`(LLM) 两臂，报告 intent_accuracy / retriever_selection_accuracy / recall / task_success / p50，
  输出 `intent_before_after.md/json`。
- **离线信号**：无 chat key 时 `auto` 自动回退规则 → 两臂相同（诚实）；配 key 才显真实提升。
- **真实 LLM A/B 结果（SiliconFlow，Qwen2.5-7B-Instruct + Qwen3-Embedding-4B，100 条 v3）**：

  | llm_mode | intent_accuracy | retriever_selection_accuracy | recall@k | task_success | p50 |
  |---|---:|---:|---:|---:|---:|
  | off（规则） | 0.10 | 0.10 | 0.4783 | 0.35 | 1395ms |
  | auto（LLM routing + rewrite） | **0.27** | **0.27** | **0.9817** | **0.84** | 4090ms |

- **结论**：LLM intent routing 把 intent/retriever selection 从 0.10 提到 0.27；更大的收益来自
  `llm_rewrite_query` 把口语/缩写改成专业检索词，使 recall@k 0.4783→0.9817、task_success 0.35→0.84。
  代价是 p50 latency 约 1.4s→4.1s，因此生产口径应按请求复杂度/难例检测选择性启用，而不是所有请求默认开。
- **routing-only 消融实验**（临时禁用 `llm_rewrite_query`，只保留 LLM intent 分类）：

  | llm_mode | intent_accuracy | retriever_selection_accuracy | recall@k | task_success | p50 |
  |---|---:|---:|---:|---:|---:|
  | off（规则） | 0.10 | 0.10 | 0.4783 | 0.35 | 1224ms |
  | auto（仅 LLM routing） | **0.24** | **0.24** | 0.4783 | 0.35 | 2462ms |

- **消融结论**：LLM 分类本身能提升 intent/retriever selection，但当前 retrieval path 尚未真正用
  `selected_retrievers` 做 source-level filtering，所以候选池不变，recall/task_success 也不变。
  上一组 `auto` 的大幅 recall 提升主要来自 query rewrite，而不是 intent label 本身。

---

## Phase 5.3 — Intent Router v2 + Source-level Routing（2026-06-10）✅

**动机**：5.2 消融证明 `selected_retrievers` 只是「可观测字段」，没接进检索链路——所以 routing-only
不动 recall，收益来自 rewrite。本轮把 routing **真正接进候选池**，并升级路由器质量。

- **Intent Router v2**（`llm.py route_intent`）：LLM 输出**严格 JSON**（primary_intent / secondary_intents
  / confidence / retrievers / rewrite_needed / reason），enum 校验，非法 JSON/label → 回退规则；
  **低置信(<0.5) → 广播 `multi_source`**（宁可放宽不要误窄）；few-shot 专攻易混对
  （concept_qa↔interview_prep、concept_qa↔project_deepdive、jd_cv_match↔evidence_lookup、multi_source）。
- **Source-level routing**（rag/hybrid/nodes）：新增 `source_routing` 请求字段；`selected_retrievers`
  → `source_type` 过滤，**两路检索都生效**。安全护栏：① 低置信不过滤；② 过滤后证据 < `MIN_SOURCE_EVIDENCE=3`
  自动**广播回退**到无过滤检索（`source_routing_fallbacks` 计数），防止过窄打穿 recall。
- **Confusion matrix**（`evaluation.intent_confusion` + `write_confusion`）：run_eval 输出
  `intent_confusion.md/json`（expected→predicted 计数 + top 混淆对 + 例子）。新增 `evidence_sufficiency` 指标。
- **4-way 路由消融**（`tuning.run_routing_compare` + `scripts/routing_compare.py`）：
  off / routing-only / +source / +source+rewrite，报告 intent/retriever acc、recall、MRR/nDCG、
  context_precision、evidence_sufficiency、p50/p95。rewrite 用 `JOB_AGENT_LLM_REWRITE` env 门控做消融。
- **离线 4-way 消融实测（100 条 v3，无 chat key，rule intent + source 过滤真实生效）**：

  | arm | recall@k | nDCG | task_success | context_prec | evidence_suff | p50 |
  |---|---:|---:|---:|---:|---:|---:|
  | off | 0.435 | 0.3953 | 0.34 | 0.241 | 1.00 | 163ms |
  | routing_only | 0.435 | 0.3953 | 0.34 | 0.241 | 1.00 | 169ms |
  | **routing+source** | **0.4883** | **0.4499** | **0.38** | 0.247 | 0.98 | 178ms |
  | routing+source+rewrite | 0.4883 | 0.4499 | 0.38 | 0.247 | 0.98 | 166ms |

- **结论（直接回答消融问题）**：① **routing-only 不动 recall**（0.435=0.435）——印证「intent label 不接检索
  就不影响候选池」；② **source-level routing 真正接入后 recall +0.053、nDCG +0.055、task_success +0.04**——
  说明按 source_type 过滤候选池**净收益为正**，不是单纯过窄；③ **没有打穿 recall**，因为广播回退兜底
  （evidence_sufficiency 仅 1.0→0.98 的小代价）；④ 离线 +rewrite 与 +source 相同（无 key→rewrite 退化），
  rewrite 的额外收益要 chat key 才显现。
- ⚠️ **诚实口径**：intent label 准确率提升仍需 chat key 跑 Router v2；但「source-level routing 在 CV-centric
  语料上是否值得」**本轮离线已给出肯定答案（+0.053 recall）**，且证明了广播回退能防止过窄。

### 5.3.1 线上 chat-key 4-way ablation（40 条 v3）

100 条全量在线 4-way 因 chat 调用较多在 20 分钟处超时，已停止残留进程以避免继续消耗额度；随后跑 40 条完整样本得到稳定趋势。

配置：`Qwen/Qwen3-Embedding-4B` + `Qwen/Qwen2.5-7B-Instruct`，`scripts/routing_compare.py --limit 40`。

| arm | intent_acc | retriever_acc | recall@k | MRR | nDCG | context_precision | task_success | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0.00 | 0.00 | 0.3458 | 0.1596 | 0.2844 | 0.0775 | 0.30 | 1249ms |
| routing_only | 0.575 | 0.075 | 0.3458 | 0.1596 | 0.2844 | 0.0775 | 0.30 | 4401ms |
| routing+source | 0.50 | 0.15 | 0.5583 | 0.1872 | 0.3263 | 0.1500 | 0.55 | 4507ms |
| routing+source+rewrite | **0.575** | 0.075 | **0.6208** | **0.2644** | **0.3857** | **0.1650** | **0.60** | 6282ms |

结论：
- 只做 LLM routing 仍不改变 recall/task_success，说明 intent label 本身不是召回提升来源。
- `routing+source` 让 recall 0.3458→0.5583、task_success 0.30→0.55，说明 source filter 真正改变候选池。
- `routing+source+rewrite` 进一步把 recall 拉到 0.6208、nDCG 拉到 0.3857、task_success 到 0.60，证明在线 LLM rewrite 在 source routing 之上仍有增益。
- `routing+source+rewrite` 的 retriever_acc 低于 `routing+source`（0.075 vs 0.15）但 recall/task 更高，
  说明 retriever_acc 只是「是否贴合 expected_retrievers 标注」的路由诊断指标；rewrite 改善的是 query-证据语义匹配，
  加上 evidence fallback 兜底，最终效果仍应看 recall/nDCG/task_success，而不是单独追 retriever_acc。
- 代价是 p50 从 1.25s→6.28s，生产上应做按需启用：规则低置信、口语/缩写、或 evidence sufficiency 不足时再开 rewrite。

---

## Phase 5 — 待实现（需要你的 key / GPU 环境）

| 计划项 | 状态 | 说明 |
|---|---|---|
| 用真实 embedding 调出 hybrid > dense | ✅ 已完成 | Qwen3-4B 实测最优 `BM25=0.5/dense=2.0/RRF k=30`，recall 0.478→0.575、task_success 0.35→0.54。 |
| Rerank A/B harness | ✅ 已完成 | `none/heuristic/api/cross-encoder` + `run_rerank_compare`；离线 heuristic>none，真实 Qwen3-Reranker-8B 已验证。 |
| 真实 rerank 跑出 MRR 提升 | ✅ 已完成 | Qwen3-Reranker-8B：MRR 0.385→0.643、nDCG 0.4735→0.6491，p50 1385→2698ms。 |
| LLM intent routing + query rewrite | ✅ 已完成 | 真实 LLM 决策路径 + intent A/B（`run_intent_compare`）就绪，含 mock 单测与规则回退。 |
| 真实 LLM routing 跑出 intent 提升 | ✅ 已完成 | Qwen2.5-7B：intent 0.10→0.27，recall 0.4783→0.9817，task_success 0.35→0.84，p50 1395→4090ms。 |
| Intent Router v2 + source-level routing | ✅ 已完成 | JSON 路由器 + enum 校验 + 低置信广播；`selected_retrievers`→source_type 过滤接入两路检索，含广播回退；confusion matrix + 4-way 消融 harness。 |
| Router v2 真实 intent 提升 / source routing 收益 | ✅ 部分完成 | 40 条线上 4-way 已跑：routing+source recall 0.3458→0.5583，+rewrite 到 0.6208；100 条全量因 chat 调用超时，未作为最终结论。 |
| 语义级 faithfulness（NLI / LLM verifier） | ❌·下一优先级 | 现 grounding 为启发式 token-overlap；可复用 `_chat_completion` seam 做 claim-evidence 核查。 |
| Redis 实跑基线 / 分布式限流压测 | ⚠️ 抽象就绪 | 代码路径就位，未起 Redis 实测。 |
| 更真实业务分布评测集 | ❌ | 在 v2/v3 之外加真实流量分布（多轮追问等）。 |
| 简历 bullet 同步 | ⚠️ | 见 `TECH_REPORT_JOB_AGENT.md`「简历可用句」，docx 需手动落。 |

## 验收对照（累计）
- ✅ 现有核心测试继续通过（10 → **66**，job_agent 相关；完整 unit **143 passed**；`ruff check .` 通过）。
- ✅ 本地无外部 key 可跑通 ingest / prepare / resume / eval（全 deterministic）。
- ✅ Qdrant local 持久化重启复用（测试覆盖）。
- ✅ 150 条 eval 一键运行，输出 metrics.csv / failure_cases.md / regression_set.jsonl /
  summary.json / before_after.md，指标真实对应。
- ✅ eval 报告展示真实 before/after 改进（faithfulness、intent）。
- ✅ session 中断后 `/resume` 从失败节点继续。
