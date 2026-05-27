"""Smoke tests for the deterministic job-agent evaluation runner."""

from pathlib import Path

from job_agent.evaluation import run_eval


def test_job_agent_eval_outputs_metrics(tmp_path: Path) -> None:
    """Evaluation should write metrics and failure artifacts."""
    eval_file = Path("data/eval/job_agent_eval.jsonl")
    summary = run_eval(eval_file, tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "failure_cases.md").exists()
    assert (tmp_path / "regression_set.jsonl").exists()
    assert summary["tool_call_validity"] == 1.0
    assert summary["report_completeness"] == 1.0
