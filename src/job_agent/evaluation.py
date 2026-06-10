"""Evaluation runner with real retrieval / answer / agent / ops metrics.

Computes, per case and in aggregate:

* retrieval: recall@k, MRR, nDCG@k, context precision, source coverage
* answer: citation accuracy, faithfulness, unsupported-claim rate, answer coverage,
  report completeness, must-not-claim violations
* agent: intent accuracy, retriever-selection accuracy, tool-call validity,
  trajectory correctness
* ops: TTFT, p50/p95 latency, cache-hit rate, token cost

It also emits a failure taxonomy, regression candidates, and supports a before/after
comparison (e.g. deterministic vs hybrid retrieval).

Backward compatible: the v1 dataset (``expected_requirements`` only) still produces the
original ``tool_call_validity`` / ``report_completeness`` summary keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application, resume_application

K = 10


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #


def _gold(case: dict[str, Any]) -> list[str]:
    gold = case.get("gold_evidence") or case.get("expected_requirements") or []
    return [str(g).lower() for g in gold]


def _is_relevant(text: str, gold: list[str]) -> bool:
    low = text.lower()
    return any(g in low for g in gold)


def recall_at_k(citations: list[Any], gold: list[str]) -> float:
    """Fraction of gold-evidence keywords found in any retrieved citation."""
    if not gold:
        return 1.0
    found = {g for g in gold if any(g in c.text.lower() for c in citations)}
    return len(found) / len(gold)


def mrr(citations: list[Any], gold: list[str]) -> float:
    """Reciprocal rank of the first relevant citation."""
    if not gold:
        return 1.0
    for rank, chunk in enumerate(citations, start=1):
        if _is_relevant(chunk.text, gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(citations: list[Any], gold: list[str], k: int = K) -> float:
    """Return normalized discounted cumulative gain over binary relevance at k."""
    if not gold:
        return 1.0
    rels = [1.0 if _is_relevant(c.text, gold) else 0.0 for c in citations[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(citations: list[Any], gold: list[str]) -> float:
    """Fraction of retrieved citations that are relevant."""
    if not citations or not gold:
        return 1.0 if not gold else 0.0
    relevant = sum(1 for c in citations if _is_relevant(c.text, gold))
    return relevant / len(citations)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[idx], 3)


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #


def _trajectory_ok(response: Any, expect_block: bool) -> float:
    nodes = [t.node for t in response.node_trace]
    if expect_block:
        return 1.0 if ("entry_guard" in nodes and "persist" not in nodes) else 0.0
    expected = ["entry_guard", "parse_jd", "parse_cv", "intent_route", "retrieve",
                "match", "generate_questions", "write_report", "persist"]
    pos = -1
    for name in expected:
        if name not in nodes:
            return 0.0
        idx = nodes.index(name)
        if idx < pos:
            return 0.0
        pos = idx
    return 1.0


def _must_not_violation(response: Any, must_not: list[str]) -> int:
    violations = 0
    for term in must_not:
        low = term.lower()
        for match in response.matches:
            if low in match.requirement.lower() and match.match_level == "strong":
                violations += 1
    return violations


def evaluate_case(case: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    """Run one case and compute its full metric row."""
    request = PrepareRequest.model_validate(case["request"])
    gold = _gold(case)
    guardrail_expected = case.get("guardrail_expected", "pass")
    expect_block = guardrail_expected == "block"

    session_id = str(uuid.uuid4())
    response = prepare_application(request, store=store, session_id=session_id)

    # Resilience cases: prove resume re-completes from checkpoints.
    resilience_ok = 1.0
    if case.get("inject_failure"):
        try:
            resumed = resume_application(session_id, store=store)
            resilience_ok = 1.0 if resumed.report else 0.0
        except Exception:
            resilience_ok = 0.0

    report_lower = response.report.lower()
    citations = response.citations
    og = response.output_guardrail
    unsupported = len(og.unsupported_claims) if og else 0

    must_answer = [str(m).lower() for m in case.get("must_answer_points", [])]
    answer_coverage = (
        sum(1 for m in must_answer if m in report_lower) / len(must_answer)
        if must_answer else 1.0
    )
    must_not = case.get("must_not_claim", [])
    must_not_violations = _must_not_violation(response, must_not)

    # Intent / retriever scoring is not applicable to adversarial inputs (their
    # meaningful behaviour is the guardrail action, scored separately).
    intent_applicable = guardrail_expected == "pass"
    expected_intent = case.get("expected_intent")
    if intent_applicable and expected_intent is not None:
        intent_acc = 1.0 if response.intent == expected_intent else 0.0
    else:
        intent_acc = 1.0
    expected_retrievers = set(case.get("expected_retrievers", []))
    got_retrievers = set(response.selected_retrievers)
    if intent_applicable and expected_retrievers:
        jaccard = len(expected_retrievers & got_retrievers) / len(expected_retrievers | got_retrievers)
        retriever_acc = 1.0 if expected_retrievers == got_retrievers else 0.0
    else:
        jaccard, retriever_acc = 1.0, 1.0

    guardrail_correct = 1.0 if response.guardrail.classification == guardrail_expected else 0.0
    tool_validity = 1.0 if all(t.status == "ok" for t in response.tool_trace) else 0.0
    citation_accuracy = 1.0 if (citations and "[C" in response.report) else 0.0
    report_completeness = (
        1.0 if "JD-CV Match Table" in response.report and "Interview Questions" in response.report
        else 0.0
    )
    faithfulness = 1.0 if unsupported == 0 and must_not_violations == 0 else 0.0

    recall = recall_at_k(citations, gold)
    # Task success reflects genuine task outcome (retrieval, grounding, guardrail,
    # no fabrication). Intent-label match is reported separately as an agent metric.
    success = 1.0 if (
        recall >= 0.5 and citation_accuracy
        and guardrail_correct and must_not_violations == 0
    ) else 0.0

    return {
        "id": case["id"],
        "task_type": case.get("task_type", "unknown"),
        # retrieval
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr(citations, gold), 4),
        "ndcg_at_k": round(ndcg_at_k(citations, gold), 4),
        "context_precision": round(context_precision(citations, gold), 4),
        "source_coverage": len({c.source_path for c in citations}),
        # answer
        "citation_accuracy": citation_accuracy,
        "faithfulness": faithfulness,
        "unsupported_claims": unsupported,
        "must_not_violations": must_not_violations,
        "answer_coverage": round(answer_coverage, 4),
        "report_completeness": report_completeness,
        # agent
        "intent_accuracy": intent_acc,
        "retriever_selection_accuracy": retriever_acc,
        "retriever_jaccard": round(jaccard, 4),
        "tool_call_validity": tool_validity,
        "trajectory_correctness": _trajectory_ok(response, expect_block),
        "guardrail_correct": guardrail_correct,
        "resilience_ok": resilience_ok,
        # ops
        "ttft_ms": response.metrics.ttft_ms,
        "total_latency_ms": response.metrics.total_latency_ms,
        "retrieval_rounds": response.metrics.retrieval_rounds,
        "cache_hits": response.metrics.cache_hits,
        "tool_call_count": response.metrics.tool_call_count,
        "task_success_rate": success,
    }


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


def classify_failure(row: dict[str, Any]) -> list[str]:
    """Tag a metric row with failure categories."""
    tags: list[str] = []
    if row["recall_at_k"] < 0.5:
        tags.append("retrieval_miss")
    if not row["citation_accuracy"]:
        tags.append("citation_missing")
    if not row["intent_accuracy"]:
        tags.append("intent_wrong")
    if not row["retriever_selection_accuracy"]:
        tags.append("retriever_wrong")
    if row["unsupported_claims"] or row["must_not_violations"]:
        tags.append("unsupported_claim")
    if not row["guardrail_correct"]:
        tags.append("guardrail_miss")
    if not row["tool_call_validity"]:
        tags.append("tool_failure")
    if not row["trajectory_correctness"]:
        tags.append("trajectory_broken")
    if not row["resilience_ok"]:
        tags.append("recovery_failed")
    return tags


# --------------------------------------------------------------------------- #
# Aggregation + runner
# --------------------------------------------------------------------------- #

_AVG_KEYS = [
    "recall_at_k", "mrr", "ndcg_at_k", "context_precision",
    "citation_accuracy", "faithfulness", "answer_coverage", "report_completeness",
    "intent_accuracy", "retriever_selection_accuracy", "tool_call_validity",
    "trajectory_correctness", "guardrail_correct", "resilience_ok",
    "task_success_rate",
]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in _AVG_KEYS:
        summary[key] = round(sum(float(r[key]) for r in rows) / len(rows), 4)
    latencies = [float(r["total_latency_ms"]) for r in rows]
    summary["latency_p50_ms"] = _percentile(latencies, 50)
    summary["latency_p95_ms"] = _percentile(latencies, 95)
    summary["ttft_p50_ms"] = _percentile([float(r["ttft_ms"]) for r in rows], 50)
    total_tool_calls = sum(int(r["tool_call_count"]) for r in rows)
    total_cache_hits = sum(int(r["cache_hits"]) for r in rows)
    summary["cache_hit_rate"] = round(total_cache_hits / total_tool_calls, 4) if total_tool_calls else 0.0
    summary["unsupported_claim_rate"] = round(
        sum(int(r["unsupported_claims"]) for r in rows) / len(rows), 4
    )
    summary["token_cost"] = 0.0  # deterministic mode uses no LLM tokens
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(eval_file: Path, output_dir: Path) -> dict[str, float]:
    """Run all cases, write metrics/failure/regression artifacts, return the summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    store = SessionStore(output_dir / "job_agent_eval.sqlite")
    rows: list[dict[str, Any]] = []
    regression: list[dict[str, Any]] = []
    taxonomy: dict[str, int] = {}

    cases = _load_jsonl(eval_file)
    for case in cases:
        row = evaluate_case(case, store)
        rows.append(row)
        if row["task_success_rate"] < 1.0:
            tags = classify_failure(row)
            for tag in tags:
                taxonomy[tag] = taxonomy.get(tag, 0) + 1
            regression.append({"id": case["id"], "task_type": row["task_type"],
                               "failure_tags": tags, "case": case})

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)

    failure_lines = ["# Job Agent Failure Cases", "", f"Total cases: {len(rows)}",
                     f"Failures: {len(regression)}", "", "## Failure taxonomy", ""]
    for tag, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
        failure_lines.append(f"- {tag}: {count}")
    failure_lines += ["", "## Top failure examples", ""]
    for item in regression[:15]:
        failure_lines.append(f"- {item['id']} ({item['task_type']}): {', '.join(item['failure_tags'])}")
    (output_dir / "failure_cases.md").write_text("\n".join(failure_lines), encoding="utf-8")

    (output_dir / "regression_set.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in regression), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


COMPARE_MODES = ["deterministic", "dense", "hybrid"]


def run_compare(eval_file: Path, output_dir: Path, modes: list[str] | None = None) -> dict[str, Any]:
    """A/B/C comparison across retrieval modes (lexical / dense / hybrid)."""
    cases = _load_jsonl(eval_file)
    modes = modes or COMPARE_MODES

    def _run(mode: str) -> dict[str, float]:
        store = SessionStore(output_dir / f"compare_{mode}.sqlite")
        rows = []
        for case in cases:
            patched = dict(case)
            patched["request"] = {**case["request"], "retrieval_mode": mode}
            rows.append(evaluate_case(patched, store))
        return _summarize(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {mode: _run(mode) for mode in modes}
    baseline = modes[0]
    keys = sorted(results[baseline])

    header = "| metric | " + " | ".join(f"{m} ({'lexical' if m == 'deterministic' else m})" for m in modes)
    header += f" | delta vs {baseline} (last) |"
    lines = [f"# Retrieval A/B/C: {' vs '.join(modes)}", "",
             "> Lexical (BM25-style) excels at exact keywords; dense excels at paraphrase/synonymy;",
             "> hybrid (RRF + rerank) trades a little latency for more robust ranking.", "",
             header, "|" + "---|" * (len(modes) + 2)]
    for key in keys:
        row_vals = " | ".join(str(results[m][key]) for m in modes)
        delta = round(results[modes[-1]][key] - results[baseline][key], 4)
        lines.append(f"| {key} | {row_vals} | {delta:+} |")

    from job_agent.embeddings import get_embedder

    embedder = type(get_embedder()).__name__
    if embedder == "HashingEmbedder":
        interpretation = [
            "> Interpretation: with the deterministic `HashingEmbedder` the dense arm is *not*",
            "> semantic, so dense ≈ lexical and hybrid does not win on these hard cases. The",
            "> semantic gain (dense beating lexical on paraphrase/synonymy) requires a real",
            "> embedder — set `JOB_AGENT_EMBEDDER=fastembed` or `=api` and re-run. Reported as-is,",
            "> no semantic win is claimed under the hashing embedder.",
        ]
    else:
        dense = results.get("dense", {})
        hybrid = results.get("hybrid", {})
        base = results[baseline]
        interpretation = [
            f"> Interpretation: `{embedder}` is a real embedding backend. On this hard set,",
            f"> dense recall@k changes from {base.get('recall_at_k')} to {dense.get('recall_at_k')},",
            f"> and hybrid recall@k changes to {hybrid.get('recall_at_k')}.",
            f"> Hybrid task success changes from {base.get('task_success_rate')} to {hybrid.get('task_success_rate')},",
            f"> while p50 latency increases from {base.get('latency_p50_ms')}ms to {hybrid.get('latency_p50_ms')}ms.",
            "> This is the trade-off to discuss: semantic retrieval improves coverage/task success",
            "> on paraphrase-heavy cases, but remote embedding adds latency and may still need",
            "> stronger reranking and LLM-based intent routing.",
        ]
    lines += [
        "",
        f"_Embedder backend: **{embedder}**._",
        "",
        *interpretation,
    ]
    (output_dir / "before_after.md").write_text("\n".join(lines), encoding="utf-8")

    deltas = {k: round(results[modes[-1]][k] - results[baseline][k], 4) for k in keys}
    return {"modes": results, "deltas": deltas}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument("--compare", action="store_true", help="Run deterministic vs hybrid comparison.")
    args = parser.parse_args()
    if args.compare:
        result = run_compare(args.eval_file, args.output_dir)
        sys.stdout.write(json.dumps(result["deltas"], indent=2, ensure_ascii=False) + "\n")
    else:
        summary = run_eval(args.eval_file, args.output_dir)
        sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
