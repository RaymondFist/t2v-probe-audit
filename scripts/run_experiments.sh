#!/usr/bin/env bash
# Run the audit end to end: manifest, experiments, analysis.
# Usage: scripts/run_experiments.sh [N] [SEED] [WORKERS]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
mkdir -p "$ROOT/output/logs"

N="${1:-60}"
SEED="${2:-2026}"
WORKERS="${3:-0}"

python -m t2v_probe_audit manifest 2>&1 | tee "$ROOT/output/logs/manifest.log"

ARGS=(run --n "$N" --seed "$SEED")
if [ "$WORKERS" -gt 0 ]; then
  ARGS+=(--workers "$WORKERS")
fi
python -m t2v_probe_audit "${ARGS[@]}" 2>&1 | tee "$ROOT/output/logs/run.log"

python -m t2v_probe_audit analyze 2>&1 | tee "$ROOT/output/logs/analyze.log"
