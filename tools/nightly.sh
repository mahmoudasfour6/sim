#!/usr/bin/env bash
# Nightly local regression: fast tier, slow tier (all solvers), and the workflow. Logs to results/nightly/.
# Registered as a Windows scheduled task by tools/register_nightly.ps1 (runs inside WSL).
set -uo pipefail
cd "$(dirname "$0")/.."
E=~/miniforge3/envs
STAMP=$(date +%Y-%m-%d_%H%M)
LOG=results/nightly/$STAMP.log
mkdir -p results/nightly
{
  echo "== nightly $STAMP on $(hostname), commit $(git rev-parse --short HEAD)"
  echo "== fast tier";  $E/workflow/bin/pytest -q 2>&1 | tail -3;            fast=${PIPESTATUS[0]}
  echo "== slow tier";  $E/workflow/bin/pytest -m slow -q 2>&1 | tail -15;   slow=${PIPESTATUS[0]}
  CORES=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['cores'])" 2>/dev/null || echo 4)
  MEM=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['mem_mb'])" 2>/dev/null || echo 16000)
  echo "== workflow (cores=$CORES mem_mb=$MEM)"
  $E/workflow/bin/snakemake all studies --cores $CORES --resources mem_mb=$MEM -q 2>&1 | tail -5; wf=${PIPESTATUS[0]}
  echo "== results"; $E/workflow/bin/python tools/summarize_results.py
  if git remote | grep -qx offsite; then
    echo "== off-laptop backup"; git push -q offsite --all 2>&1 | tail -2; echo "  pushed $(git rev-parse --short HEAD)"
  fi
  echo "== summary: fast=$fast slow=$slow workflow=$wf (0 = pass)"
} > "$LOG" 2>&1
tail -1 "$LOG" > results/nightly/LATEST.txt
cp "$LOG" "/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/sim_backup/nightly_latest.log" 2>/dev/null || true
