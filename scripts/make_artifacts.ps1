# One-click renderer for figures and table source data.
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File scripts/make_artifacts.ps1
#
# Reads output/analysis/numbers.json and writes figures/ and tables/. It does not
# run any experiment.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
$Logs = Join-Path $Root "output\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

python -m t2v_probe_audit artifacts 2>&1 |
    Tee-Object -FilePath (Join-Path $Logs "artifacts.log")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output ""
Write-Output "figures: $Root\figures"
Write-Output "tables:  $Root\tables"
Write-Output "To refresh the manuscript tokens: python scripts/fill_manuscript.py"
