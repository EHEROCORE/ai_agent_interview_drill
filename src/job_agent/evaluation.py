"""Deterministic evaluation runner for the job application agent."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_eval(eval_file: Path, output_dir: Path) -> dict[str, float]:
    """Run eval cases and write metrics/failure reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    store = SessionStore(output_dir / "job_agent_eval.sqlite")
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for case in _load_jsonl(eval_file):
        request = PrepareRequest.model_validate(case["request"])
        expected_requirements = [str(item).lower() for item in case.get("expected_requirements", [])]
        response = prepare_application(request, store=store)
        report_lower = response.report.lower()
        requirement_hits = sum(
            1 for requirement in expected_requirements if requirement in report_lower
        )
        requirement_recall = (
            requirement_hits / len(expected_requirements) if expected_requirements else 1.0
        )
        citation_accuracy = 1.0 if response.citations and "[C" in response.report else 0.0
        report_complete = 1.0 if "JD-CV Match Table" in response.report and "Interview Questions" in response.report else 0.0
        guardrail_pass = 0.0 if response.guardrail.blocked else 1.0
        task_success = 1.0 if requirement_recall >= 0.5 and citation_accuracy else 0.0

        row = {
            "id": case["id"],
            "task_type": case["task_type"],
            "tool_call_validity": 1.0 if all(trace.status == "ok" for trace in response.tool_trace) else 0.0,
            "retrieval_recall_at_k": round(requirement_recall, 4),
            "citation_accuracy": citation_accuracy,
            "report_completeness": report_complete,
            "guardrail_pass_rate": guardrail_pass,
            "task_success_rate": task_success,
            "ttft_ms": response.metrics.ttft_ms,
            "total_latency_ms": response.metrics.total_latency_ms,
            "tool_call_count": response.metrics.tool_call_count,
        }
        rows.append(row)
        if not task_success:
            failures.append(
                f"- {case['id']} ({case['task_type']}): recall={requirement_recall:.2f}, citations={citation_accuracy}"
            )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failure_path = output_dir / "failure_cases.md"
    failure_path.write_text(
        "# Job Agent Failure Cases\n\n" + ("\n".join(failures) if failures else "No failures."),
        encoding="utf-8",
    )

    regression_path = output_dir / "regression_set.jsonl"
    regression_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows if row["task_success_rate"] < 1.0),
        encoding="utf-8",
    )

    summary: dict[str, float] = {}
    for key in [
        "tool_call_validity",
        "retrieval_recall_at_k",
        "citation_accuracy",
        "report_completeness",
        "guardrail_pass_rate",
        "task_success_rate",
        "ttft_ms",
        "total_latency_ms",
    ]:
        summary[key] = round(sum(float(row[key]) for row in rows) / len(rows), 4)
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eval"))
    args = parser.parse_args()
    summary = run_eval(args.eval_file, args.output_dir)
    sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
