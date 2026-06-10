"""CLI: A/B the intent router (rule vs LLM) on an eval set.

Attacks the measured short board: rule-based intent routing scores ~0.1 on the v3 hard
(colloquial / abbreviated) cases. Routing this one decision through an LLM is the
highest-leverage use of a model.

Examples:
  # offline: both arms = rule routing (no chat key) — validates the harness
  uv run python scripts/intent_compare.py

  # real LLM routing via SiliconFlow chat:
  $env:JOB_AGENT_EMBEDDER="api"; $env:JOB_AGENT_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
  $env:JOB_AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"; $env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
  uv run python scripts/intent_compare.py --metric intent_accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_agent.tuning import run_intent_compare


def main() -> None:
    """Parse args and run the intent-routing comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path,
                        default=Path("data") / "eval" / "semantic_hard_eval_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "intent_compare")
    parser.add_argument("--metric", default="intent_accuracy")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cases (0 = all).")
    parser.add_argument("--modes", nargs="+", default=["off", "auto"])
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]

    result = run_intent_compare(cases, args.output_dir, modes=args.modes, metric=args.metric)
    table = {r["llm_mode"]: r["summary"].get(args.metric) for r in result["results"]}
    print(json.dumps({args.metric: table}, ensure_ascii=False, indent=2))  # noqa: T201
    print(f"Report: {args.output_dir / 'intent_before_after.md'}")  # noqa: T201


if __name__ == "__main__":
    main()
