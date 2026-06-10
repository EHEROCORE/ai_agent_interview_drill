"""CLI: compare rerank strategies on a fixed best-fusion retrieval.

Layered directly on the Phase-4 fusion result: fusion fixed → vary the reranker → see
whether ordering quality (MRR / nDCG / context precision) improves.

Examples:
  # offline: none vs heuristic (validates the harness)
  uv run python scripts/rerank_compare.py --strategies none heuristic

  # real reranker via SiliconFlow (no torch needed):
  $env:JOB_AGENT_EMBEDDER="api"; $env:JOB_AGENT_EMBED_API_BASE="https://api.siliconflow.com/v1"
  $env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
  $env:JOB_AGENT_RERANKER="api"; $env:JOB_AGENT_RERANK_API_BASE="https://api.siliconflow.com/v1"
  $env:JOB_AGENT_RERANK_MODEL="BAAI/bge-reranker-v2-m3"
  $env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
  uv run python scripts/rerank_compare.py --strategies none heuristic api --metric mrr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_agent.tuning import RERANK_STRATEGIES, run_rerank_compare


def main() -> None:
    """Parse args and run the rerank comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path,
                        default=Path("data") / "eval" / "semantic_hard_eval_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "rerank_compare")
    parser.add_argument("--metric", default="mrr")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cases (0 = all).")
    parser.add_argument("--strategies", nargs="+", default=RERANK_STRATEGIES,
                        help=f"Subset of {RERANK_STRATEGIES}.")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]

    result = run_rerank_compare(cases, args.output_dir, strategies=args.strategies, metric=args.metric)
    table = {r["strategy"]: r["summary"].get(args.metric) for r in result["results"]}
    print(json.dumps({args.metric: table}, ensure_ascii=False, indent=2))  # noqa: T201
    print(f"Report: {args.output_dir / 'rerank_before_after.md'}")  # noqa: T201


if __name__ == "__main__":
    main()
