#!/usr/bin/env bash
# One-click renderer for figures and table source data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
mkdir -p "$ROOT/output/logs"

python -m t2v_probe_audit artifacts 2>&1 | tee "$ROOT/output/logs/artifacts.log"

echo
echo "figures: $ROOT/figures"
echo "tables:  $ROOT/tables"
echo "To refresh the manuscript tokens: python scripts/fill_manuscript.py"
