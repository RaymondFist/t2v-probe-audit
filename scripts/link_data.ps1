param(
    [Parameter(Mandatory = $true)][string]$Source,
    [switch]$Force
)
# Point data/raw at an existing local copy of the ECTV corpus.
#
# The corpus is third-party material and is not redistributed with this repository.
# Copy it here if it is already on the machine, or download the public release:
#     huggingface-cli download RaphaelLiu/EvalCrafter_T2V_Dataset --local-dir data/raw
#
# A directory junction is used; no video is copied or modified.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $Root "data\raw"

if (-not (Test-Path $Source)) {
    Write-Error "source not found: $Source"
    exit 1
}

if (Test-Path $Target) {
    if (-not $Force) {
        Write-Output "data/raw already exists; use -Force to replace the junction"
        exit 0
    }
    $item = Get-Item $Target -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        [System.IO.Directory]::Delete($item.FullName, $false)
    } else {
        Write-Error "data/raw is a real directory, refusing to delete it"
        exit 1
    }
}

try {
    New-Item -ItemType Junction -Path $Target -Target $Source -ErrorAction Stop | Out-Null
} catch {
    Write-Output "could not create a junction: $($_.Exception.Message)"
    Write-Output "set the corpus location for each run instead:"
    Write-Output ('  $env:TPA_DATA_ROOT = "{0}"' -f $Source)
    exit 1
}
Write-Output "linked data/raw -> $Source"
Write-Output "generators:"
Get-ChildItem $Target -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -Filter *.mp4 | Measure-Object).Count
    Write-Output ("  {0}: {1} clips" -f $_.Name, $count)
}
