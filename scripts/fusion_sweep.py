"""CLI: sweep hybrid-fusion weights against an eval set and report the best config.

Examples:
  # offline (hashing embedder) — validates the harness
  uv run python scripts/fusion_sweep.py

  # real semantic tune (SiliconFlow Qwen3) — find weights that beat dense-only
  $env:JOB_AGENT_EMBEDDER="api"; $env:SILICONFLOW_API_KEY="sk-..."
  $env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
  uv run python scripts/fusion_sweep.py --metric recall_at_k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_agent.tuning import run_fusion_sweep


def main() -> None:
    """Parse args and run the fusion sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path,
                        default=Path("data") / "eval" / "semantic_hard_eval_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "fusion_sweep")
    parser.add_argument("--metric", default="recall_at_k")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cases (0 = all).")
    parser.add_argument("--grid", type=Path, default=None,
                        help="Optional JSON file: a list of fusion-config dicts.")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]
    grid = json.loads(args.grid.read_text(encoding="utf-8")) if args.grid else None

    result = run_fusion_sweep(cases, args.output_dir, grid=grid, metric=args.metric)
    best = result["best"]
    print(json.dumps({"best_combo": best["combo"],  # noqa: T201
                      args.metric: best["summary"].get(args.metric)}, ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'fusion_sweep.md'}")  # noqa: T201


if __name__ == "__main__":
    main()
