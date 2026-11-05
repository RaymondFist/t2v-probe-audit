param(
    [int]$N = 60,
    [int]$Seed = 2026,
    [int]$Workers = 0,
    [switch]$ForceManifest,
    [switch]$SkipManifest
)
# Run the audit end to end: manifest, experiments, analysis.
#
# Usage (from any directory):
#     powershell -ExecutionPolicy Bypass -File scripts/run_experiments.ps1
#     powershell -ExecutionPolicy Bypass -File scripts/run_experiments.ps1 -N 10
#
# Logs are written to output/logs.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
$Logs = Join-Path $Root "output\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

if (-not $SkipManifest) {
    if ($ForceManifest) {
        python -m t2v_probe_audit manifest --force 2>&1 |
            Tee-Object -FilePath (Join-Path $Logs "manifest.log")
    } else {
        python -m t2v_probe_audit manifest 2>&1 |
            Tee-Object -FilePath (Join-Path $Logs "manifest.log")
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$RunArgs = @("run", "--n", $N, "--seed", $Seed)
if ($Workers -gt 0) { $RunArgs += @("--workers", $Workers) }
python -m t2v_probe_audit @RunArgs 2>&1 |
    Tee-Object -FilePath (Join-Path $Logs "run.log")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m t2v_probe_audit analyze 2>&1 |
    Tee-Object -FilePath (Join-Path $Logs "analyze.log")
exit $LASTEXITCODE
