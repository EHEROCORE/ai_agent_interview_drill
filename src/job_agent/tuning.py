"""Hybrid-fusion weight sweep.

Grid-searches the ``JOB_AGENT_FUSION_*`` knobs (bm25/dense weights, RRF k, metadata
boost, per-arm top-k) against an eval set in **hybrid** mode and reports the configs
ranked by a target metric, so the optimal fusion weights can be found empirically.

Offline-safe: runs with the deterministic hashing embedder by default; set
``JOB_AGENT_EMBEDDER=api`` (+ key) to sweep against real semantic embeddings. The
embedding cache is shared across configs (vectors computed once), while each config gets
a fresh retrieval-result store so fusion configs never read each other's cached results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from job_agent.evaluation import _summarize, evaluate_case
from job_agent.session import SessionStore

# Knob name -> the env var FusionConfig.from_env() reads.
_ENV = {
    "bm25_weight": "JOB_AGENT_FUSION_BM25_WEIGHT",
    "dense_weight": "JOB_AGENT_FUSION_DENSE_WEIGHT",
    "rrf_k": "JOB_AGENT_FUSION_RRF_K",
    "metadata_boost_weight": "JOB_AGENT_FUSION_META_BOOST",
    "sparse_top_k": "JOB_AGENT_FUSION_SPARSE_TOPK",
    "dense_top_k": "JOB_AGENT_FUSION_DENSE_TOPK",
    "final_top_k": "JOB_AGENT_FUSION_FINAL_TOPK",
}

# A focused default grid (kept small so a local run is tractable). Override via the CLI.
DEFAULT_GRID: list[dict[str, float | int]] = [
    {"bm25_weight": 1.0, "dense_weight": 1.0},   # balanced baseline
    {"bm25_weight": 0.5, "dense_weight": 1.0},   # lean dense
    {"bm25_weight": 0.3, "dense_weight": 1.0},
    {"bm25_weight": 1.0, "dense_weight": 2.0},   # favour dense
    {"bm25_weight": 1.0, "dense_weight": 3.0},
    {"bm25_weight": 0.5, "dense_weight": 2.0, "rrf_k": 30},
]

REPORT_METRICS = [
    "recall_at_k", "ndcg_at_k", "mrr", "context_precision",
    "task_success_rate", "latency_p50_ms",
]


def _apply_env(combo: dict[str, float | int]) -> dict[str, str | None]:
    """Set the fusion env vars for ``combo``; return prior values for restoration."""
    prior: dict[str, str | None] = {}
    for key, value in combo.items():
        env = _ENV[key]
        prior[env] = os.environ.get(env)
        os.environ[env] = str(value)
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for env, value in prior.items():
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value


def run_fusion_sweep(
    cases: list[dict[str, Any]],
    output_dir: Path,
    grid: list[dict[str, float | int]] | None = None,
    metric: str = "recall_at_k",
) -> dict[str, Any]:
    """Run the sweep and write ``fusion_sweep.md``; return ranked results + best combo."""
    grid = grid or DEFAULT_GRID
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for i, combo in enumerate(grid):
        prior = _apply_env(combo)
        try:
            store = SessionStore(output_dir / f"sweep_{i}.sqlite")
            rows = []
            for case in cases:
                patched = dict(case)
                # Fusion only affects hybrid retrieval; force it for an apples-to-apples sweep.
                patched["request"] = {**case["request"], "retrieval_mode": "hybrid"}
                rows.append(evaluate_case(patched, store))
            summary = _summarize(rows)
        finally:
            _restore_env(prior)
        results.append({"combo": combo, "summary": summary})

    higher_is_better = metric != "latency_p50_ms"
    results.sort(key=lambda r: r["summary"].get(metric, 0.0), reverse=higher_is_better)
    best = results[0]

    embedder = type(_active_embedder()).__name__
    lines = [
        "# Hybrid Fusion Weight Sweep", "",
        f"_Target metric: **{metric}** · embedder: **{embedder}** · {len(cases)} cases · "
        f"{len(grid)} configs._", "",
        "| rank | config | " + " | ".join(REPORT_METRICS) + " |",
        "|---|---|" + "---|" * len(REPORT_METRICS),
    ]
    for rank, r in enumerate(results, start=1):
        combo_str = ", ".join(f"{k}={v}" for k, v in r["combo"].items())
        vals = " | ".join(str(r["summary"].get(m)) for m in REPORT_METRICS)
        lines.append(f"| {rank} | {combo_str} | {vals} |")

    best_env = " ".join(f'{_ENV[k]}={v}' for k, v in best["combo"].items())
    lines += [
        "", f"**Best ({metric})**: `{best_env or 'defaults'}` "
        f"→ {metric}={best['summary'].get(metric)}", "",
        "> Reproduce (PowerShell): "
        + "; ".join(f'$env:{_ENV[k]}="{v}"' for k, v in best["combo"].items()),
        "",
        "> Note: with the hashing embedder the dense arm is not semantic, so the sweep mainly",
        "> validates the harness. Run with `JOB_AGENT_EMBEDDER=api` (+ key) for a meaningful tune.",
    ]
    (output_dir / "fusion_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "fusion_sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"results": results, "best": best, "metric": metric}


def _active_embedder() -> object:
    from job_agent.embeddings import get_embedder

    return get_embedder()


# Best fusion config from the Phase-4 sweep (Qwen3-Embedding-4B); reranking is layered
# on top of this fixed retrieval so the only variable in the A/B is the reranker.
BEST_FUSION: dict[str, float | int] = {"bm25_weight": 0.5, "dense_weight": 2.0, "rrf_k": 30}
RERANK_STRATEGIES = ["none", "heuristic", "api", "cross-encoder"]
RERANK_REPORT_METRICS = [
    "mrr", "ndcg_at_k", "context_precision", "recall_at_k",
    "task_success_rate", "latency_p50_ms",
]


def run_rerank_compare(
    cases: list[dict[str, Any]],
    output_dir: Path,
    strategies: list[str] | None = None,
    fusion: dict[str, float | int] | None = None,
    metric: str = "mrr",
) -> dict[str, Any]:
    """Compare rerank strategies on a *fixed* best-fusion retrieval; write before/after.

    Fusion is pinned (default :data:`BEST_FUSION`) so the A/B isolates the reranker's
    effect on ordering quality (MRR / nDCG / context precision). Offline only ``none`` and
    ``heuristic`` run; ``api`` / ``cross-encoder`` need a key / model and otherwise fall
    back to the heuristic (still recorded, clearly labelled).
    """
    strategies = strategies or RERANK_STRATEGIES
    fusion = fusion or BEST_FUSION
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_prior = _apply_env(fusion)  # pin fusion for the whole comparison
    results: list[dict[str, Any]] = []
    try:
        for i, strategy in enumerate(strategies):
            prior = os.environ.get("JOB_AGENT_RERANKER")
            os.environ["JOB_AGENT_RERANKER"] = strategy
            try:
                store = SessionStore(output_dir / f"rerank_{strategy}.sqlite")
                rows = []
                for case in cases:
                    patched = dict(case)
                    patched["request"] = {**case["request"], "retrieval_mode": "hybrid"}
                    rows.append(evaluate_case(patched, store))
                summary = _summarize(rows)
            finally:
                if prior is None:
                    os.environ.pop("JOB_AGENT_RERANKER", None)
                else:
                    os.environ["JOB_AGENT_RERANKER"] = prior
            results.append({"strategy": strategy, "summary": summary})
    finally:
        _restore_env(fusion_prior)

    higher_is_better = metric != "latency_p50_ms"
    results.sort(key=lambda r: r["summary"].get(metric, 0.0), reverse=higher_is_better)
    baseline = next((r for r in results if r["strategy"] == "none"), results[-1])

    embedder = type(_active_embedder()).__name__
    fusion_str = ", ".join(f"{k}={v}" for k, v in fusion.items())
    lines = [
        "# Rerank Before/After", "",
        f"_Target: **{metric}** · embedder: **{embedder}** · fusion fixed: `{fusion_str}` · "
        f"{len(cases)} cases._", "",
        "| rank | reranker | " + " | ".join(RERANK_REPORT_METRICS) + " | Δ" + metric + " vs none |",
        "|---|---|" + "---|" * (len(RERANK_REPORT_METRICS) + 1),
    ]
    base_val = baseline["summary"].get(metric, 0.0)
    for rank, r in enumerate(results, start=1):
        vals = " | ".join(str(r["summary"].get(m)) for m in RERANK_REPORT_METRICS)
        delta = round(r["summary"].get(metric, 0.0) - base_val, 4)
        lines.append(f"| {rank} | {r['strategy']} | {vals} | {delta:+} |")
    lines += [
        "", "> Baseline `none` = raw fusion order. A reranker is worth it only if it lifts",
        "> MRR / nDCG / context precision without blowing up p50 latency.",
        "> `api` / `cross-encoder` require a key / model; offline they fall back to heuristic.",
    ]
    (output_dir / "rerank_before_after.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "rerank_before_after.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"results": results, "metric": metric, "baseline": baseline}


INTENT_REPORT_METRICS = [
    "intent_accuracy", "retriever_selection_accuracy", "recall_at_k",
    "task_success_rate", "latency_p50_ms",
]


def run_intent_compare(
    cases: list[dict[str, Any]],
    output_dir: Path,
    modes: list[str] | None = None,
    metric: str = "intent_accuracy",
) -> dict[str, Any]:
    """A/B the request ``llm_mode`` (rule routing vs LLM routing) on the eval set.

    Offline (no key) ``auto`` degrades to the rule classifier, so both arms match; with a
    chat key the ``auto`` arm shows the real intent-accuracy lift on hard/colloquial cases.
    """
    modes = modes or ["off", "auto"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mode in modes:
        store = SessionStore(output_dir / f"intent_{mode}.sqlite")
        rows = []
        for case in cases:
            patched = dict(case)
            patched["request"] = {**case["request"], "llm_mode": mode}
            rows.append(evaluate_case(patched, store))
        results.append({"llm_mode": mode, "summary": _summarize(rows)})

    base = next((r for r in results if r["llm_mode"] == "off"), results[0])
    lines = [
        "# Intent Routing Before/After (rule vs LLM)", "",
        f"_Target: **{metric}** · {len(cases)} cases · modes: {', '.join(modes)}._", "",
        "| llm_mode | " + " | ".join(INTENT_REPORT_METRICS) + f" | Δ{metric} vs off |",
        "|---|" + "---|" * (len(INTENT_REPORT_METRICS) + 1),
    ]
    base_val = base["summary"].get(metric, 0.0)
    for r in results:
        vals = " | ".join(str(r["summary"].get(m)) for m in INTENT_REPORT_METRICS)
        delta = round(r["summary"].get(metric, 0.0) - base_val, 4)
        lines.append(f"| {r['llm_mode']} | {vals} | {delta:+} |")
    lines += [
        "", "> Offline (no chat key) `auto` falls back to rule routing → arms match.",
        "> Set a chat key + `JOB_AGENT_LLM_MODEL` for the real LLM-routing lift.",
    ]
    (output_dir / "intent_before_after.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "intent_before_after.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"results": results, "metric": metric}


# 4-way routing ablation: isolate routing vs source routing vs query rewrite.
ROUTING_ARMS: list[dict[str, Any]] = [
    {"name": "off", "llm_mode": "off", "source_routing": False, "rewrite": True},
    {"name": "routing_only", "llm_mode": "auto", "source_routing": False, "rewrite": False},
    {"name": "routing+source", "llm_mode": "auto", "source_routing": True, "rewrite": False},
    {"name": "routing+source+rewrite", "llm_mode": "auto", "source_routing": True, "rewrite": True},
]
ROUTING_REPORT_METRICS = [
    "intent_accuracy", "retriever_selection_accuracy", "recall_at_k", "mrr", "ndcg_at_k",
    "context_precision", "evidence_sufficiency", "task_success_rate",
    "latency_p50_ms", "latency_p95_ms",
]


def run_routing_compare(
    cases: list[dict[str, Any]],
    output_dir: Path,
    arms: list[dict[str, Any]] | None = None,
    metric: str = "recall_at_k",
) -> dict[str, Any]:
    """4-way ablation: off / routing-only / +source / +source+rewrite.

    Answers: does LLM routing alone move recall (it shouldn't until retrievers are wired)?
    does source routing help or over-filter? rewrite vs source — which contributes more?
    Offline (no chat key) every ``auto`` arm degrades to rule routing / no rewrite, but
    ``source_routing=True`` still changes the candidate pool. A chat key is only needed
    to measure the incremental LLM-intent / rewrite lift.
    """
    arms = arms or ROUTING_ARMS
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for arm in arms:
        prior = os.environ.get("JOB_AGENT_LLM_REWRITE")
        os.environ["JOB_AGENT_LLM_REWRITE"] = "on" if arm["rewrite"] else "off"
        try:
            store = SessionStore(output_dir / f"routing_{arm['name']}.sqlite")
            rows = []
            for case in cases:
                patched = dict(case)
                patched["request"] = {**case["request"], "llm_mode": arm["llm_mode"],
                                      "source_routing": arm["source_routing"],
                                      "retrieval_mode": case["request"].get("retrieval_mode", "hybrid")}
                rows.append(evaluate_case(patched, store))
            summary = _summarize(rows)
        finally:
            if prior is None:
                os.environ.pop("JOB_AGENT_LLM_REWRITE", None)
            else:
                os.environ["JOB_AGENT_LLM_REWRITE"] = prior
        results.append({"arm": arm["name"], "summary": summary})

    base = results[0]["summary"]
    lines = [
        "# Routing Ablation (off / routing-only / +source / +source+rewrite)", "",
        f"_Target Δ vs off: **{metric}** · {len(cases)} cases._", "",
        "| arm | " + " | ".join(ROUTING_REPORT_METRICS) + f" | Δ{metric} |",
        "|---|" + "---|" * (len(ROUTING_REPORT_METRICS) + 1),
    ]
    for r in results:
        vals = " | ".join(str(r["summary"].get(m)) for m in ROUTING_REPORT_METRICS)
        delta = round(r["summary"].get(metric, 0.0) - base.get(metric, 0.0), 4)
        lines.append(f"| {r['arm']} | {vals} | {delta:+} |")
    lines += [
        "", "> Reads: routing-only moving recall ⇒ retrievers truly affect the pool;",
        "> +source under +rewrite ⇒ source filter's marginal value; watch recall for over-filtering.",
        "> Offline (no chat key), LLM routing/rewrite fall back to rules, but",
        "> `source_routing=True` still changes the candidate pool via source filters.",
    ]
    (output_dir / "routing_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "routing_ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"results": results, "metric": metric}
