"""CLI: 4-way routing ablation — off / routing-only / +source / +source+rewrite.

Isolates the three routing levers so you can answer: does LLM routing alone move recall
(needs `selected_retrievers` wired into retrieval), does source-level filtering help or
over-narrow, and does query rewrite or source routing contribute more.

Examples:
  # offline: all `auto` arms collapse to rule routing (validates the harness)
  uv run python scripts/routing_compare.py --limit 20

  # real LLM routing via SiliconFlow chat:
  $env:JOB_AGENT_EMBEDDER="api"; $env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
  $env:JOB_AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"; $env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
  uv run python scripts/routing_compare.py --metric recall_at_k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_agent.tuning import run_routing_compare


def main() -> None:
    """Parse args and run the 4-way routing ablation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path,
                        default=Path("data") / "eval" / "semantic_hard_eval_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "routing_compare")
    parser.add_argument("--metric", default="recall_at_k")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cases (0 = all).")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]

    result = run_routing_compare(cases, args.output_dir, metric=args.metric)
    table = {r["arm"]: r["summary"].get(args.metric) for r in result["results"]}
    print(json.dumps({args.metric: table}, ensure_ascii=False, indent=2))  # noqa: T201
    print(f"Report: {args.output_dir / 'routing_ablation.md'}")  # noqa: T201


if __name__ == "__main__":
    main()
