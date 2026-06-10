# Job Application Research Agent - one-command local demo (Windows PowerShell).
# Runs fully offline: ingest -> prepare -> resume -> eval. No API key required.
#
#   pwsh scripts/job_agent_demo.ps1
#
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "== Job Agent demo: ingest -> prepare -> resume -> eval ==" -ForegroundColor Cyan

# Ensure the eval datasets exist.
uv run python -m job_agent.eval_dataset
uv run python -m job_agent.eval_dataset_v3

# Run the end-to-end demo driver.
uv run python scripts/demo.py

Write-Host "`n== Optional: run the API and try curl ==" -ForegroundColor Cyan
Write-Host @'
# Terminal A - start the API:
uv run uvicorn job_agent.entrypoint:app --port 8000

# Terminal B - call it:
curl -s -X POST http://localhost:8000/prepare -H "Content-Type: application/json" -d "{\"company\":\"Tencent\",\"role\":\"AI Agent Engineer\",\"job_description\":\"Python RAG AI Agent FastAPI evaluation guardrails\",\"cv_text\":\"Python, RAG, FastAPI, LangGraph\",\"retrieval_mode\":\"hybrid\"}"

# Streaming (skeleton first, then result):
curl -N -X POST http://localhost:8000/prepare/stream -H "Content-Type: application/json" -d "{\"company\":\"Tencent\",\"role\":\"AI Agent Engineer\",\"job_description\":\"Python RAG FastAPI\",\"cv_text\":\"Python RAG FastAPI\"}"

# Trigger the offline eval suite:
curl -s -X POST http://localhost:8000/eval/run
'@
