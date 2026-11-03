# One-click local run for Windows PowerShell:
#   .\scripts\run_all.ps1
# (activate the virtual environment first:  .\.venv\Scripts\Activate.ps1)

$ErrorActionPreference = "Stop"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:MLFLOW_DISABLE_TELEMETRY = "1"

Write-Host "`n=== 1/5 Lint ===" -ForegroundColor Cyan
flake8 src app tests scripts

Write-Host "`n=== 2/5 DVC pipeline: collect -> preprocess -> train -> evaluate ===" -ForegroundColor Cyan
dvc repro

Write-Host "`n=== 3/5 Tests ===" -ForegroundColor Cyan
pytest -v

Write-Host "`n=== 4/5 Drift check (healthy data, then shifted data) ===" -ForegroundColor Cyan
python -m src.monitor --simulate none
python -m src.monitor --simulate drift

Write-Host "`n=== 5/5 Metrics ===" -ForegroundColor Cyan
dvc metrics show

Write-Host "`nAll steps finished. Next: 'uvicorn app.main:app --reload' and 'mlflow ui --backend-store-uri sqlite:///mlflow.db'" -ForegroundColor Green
