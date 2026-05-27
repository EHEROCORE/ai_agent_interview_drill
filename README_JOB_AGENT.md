# Job Application Research Agent

This project extends the original LangGraph ReAct template into a portfolio-grade AI application project:

```text
Entry Gateway -> Agent Orchestration -> Tool/RAG -> Guardrails -> Trace/Logging -> Evaluation -> Feedback Loop
```

## What It Does

Given a company, target role, job description, and CV text, the agent:

1. validates the request and checks prompt-injection risks
2. builds a local RAG index over the CV, JD, interview question bank, project notes, and planning notes
3. extracts JD requirements
4. retrieves cited CV/project evidence
5. builds a JD-CV match table
6. generates targeted interview questions
7. writes a cited preparation report
8. stores session state, tool traces, metrics, and feedback

The implementation is deterministic by default, so it can run locally without an external LLM API key. The original LangGraph ReAct agent remains available as `agent`; the new graph is registered as `job_agent`.

## Key Files

```text
src/job_agent/
  entrypoint.py      FastAPI app: /healthz, /prepare, /sessions, /feedback
  graph.py           LangGraph wrapper for the job application workflow
  workflow.py        End-to-end orchestration
  rag.py             Local hybrid RAG index and chunking
  guardrails.py      Prompt injection, tool and output checks
  session.py         SQLite-backed state, trace, cache and feedback store
  evaluation.py      Deterministic evaluation runner
  tools/             JD parser, CV parser, retrieval, matcher, interview, report

data/eval/job_agent_eval.jsonl
reports/eval/
tests/unit_tests/test_job_agent_core.py
tests/integration_tests/test_job_agent_workflow.py
tests/evaluations/test_job_agent_eval.py
```

## Run Tests

```powershell
$env:UV_CACHE_DIR="F:\wu_sm2\study_llm\agent_project_base\.uv-cache"
$env:UV_PYTHON="D:\anaconda3\python.exe"
uv run python -m pytest tests/unit_tests/test_job_agent_core.py tests/integration_tests/test_job_agent_workflow.py tests/evaluations/test_job_agent_eval.py -q
```

## Run Evaluation

```powershell
uv run python -m job_agent.evaluation --eval-file data/eval/job_agent_eval.jsonl --output-dir reports/eval
```

Current evaluation smoke output:

```json
{
  "tool_call_validity": 1.0,
  "retrieval_recall_at_k": 0.9222,
  "citation_accuracy": 1.0,
  "report_completeness": 1.0,
  "guardrail_pass_rate": 1.0,
  "task_success_rate": 1.0
}
```

Generated artifacts:

```text
reports/eval/metrics.csv
reports/eval/failure_cases.md
reports/eval/regression_set.jsonl
reports/eval/traces/*.jsonl
```

## Run API

```powershell
uv run uvicorn job_agent.entrypoint:app --reload
```

Endpoints:

```text
GET  /healthz
POST /prepare
GET  /sessions/{session_id}
POST /sessions/{session_id}/feedback
```

Set `JOB_AGENT_API_KEY` to enable API-key authentication.

## Interview Mapping

This project is designed to answer production Agent questions:

- Prompt injection: `guardrails.py`
- RAG metrics and low-recall optimization: `rag.py`, `evaluation.py`
- Agent quality evaluation: `data/eval/job_agent_eval.jsonl`, `reports/eval/metrics.csv`
- Entry management: `entrypoint.py`
- Auth/rate limit/parameter validation: FastAPI dependency + Pydantic schemas
- State recovery: `session.py`
- TTFT/latency metrics: `workflow.py`, `Metrics`
- Data flywheel: feedback endpoint + failure/regression artifacts

## Attribution

The base ReAct agent scaffold is built on the MIT-licensed [`webup/langgraph-up-react`](https://github.com/webup/langgraph-up-react) template (originally derived from LangChain's react-agent). The `job_agent` application layer in this repository is my own work. See `LICENSE` for the original copyright notice.
