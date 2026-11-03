#!/usr/bin/env bash
# Resume the simws 0.9a install after an interruption in step 7 (steps 1-6 are done: results/ is complete).
#   bash tools/resume_v9a.sh            stops with the failure details if a slow test fails; rerun after a fix
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
step(){ echo; echo "== $*"; }
DFTENV=$($E/workflow/bin/python -c "import yaml; print((yaml.safe_load(open('config/config.yaml')).get('dft') or {}).get('env', 'materials'))")
ENVS="workflow tcad quantum_iii_v process_rad $DFTENV"

step "R1 Charon cold-diode cross-check"
CHARON_OK=skipped
if bash tools/run_charon_xcheck.sh > logs_charon.txt 2>&1; then CHARON_OK=pass; else
  rc=$?; [ $rc -eq 2 ] && CHARON_OK="skipped (Charon not found)" || CHARON_OK="FAIL (logs_charon.txt)"; fi
tail -25 logs_charon.txt; echo "  Charon cross-check: $CHARON_OK"
if [[ $CHARON_OK == FAIL* ]]; then           # hand the failing deck and its log over for debugging
  WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
  DBG=/mnt/c/Users/$WINUSER/sim_backup/charon_dbg; mkdir -p "$DBG"
  cp -f logs_charon.txt "$DBG/" 2>/dev/null || true
  for f in runs/charon_xcheck/*.log; do d=${f%.log}; cp -f "$f" "$d".inp "$d".inp.xml "$d".interp.log "$DBG/" 2>/dev/null || true; done
  echo "  copied the Charon logs and decks to $DBG"
fi

step "R2 slow tier: the failed 0.8 reproduction test, with details (last-failed first; all if no record)"
if ! $E/workflow/bin/pytest -m slow -q -k "simws and not cryo_gate and not charon" --lf -rf --tb=short > logs_slow.txt 2>&1; then
  grep -E "^(FAILED|ERROR)|Error|assert|golden|rel|!=" logs_slow.txt | head -60
  tail -3 logs_slow.txt
  echo; echo "STOPPED: a 0.8 result does not reproduce. Send logs_slow.txt (in ~/sim_workspace)."; exit 1
fi
tail -1 logs_slow.txt

step "R3 record and confirm the 0.9a golden values"
$E/workflow/bin/pytest -m slow -q -k cryo_gate --update-golden 2>&1 | tail -1
$E/workflow/bin/pytest -m slow -q -k "cryo_gate or charon" -rs 2>&1 | tail -3
$E/workflow/bin/pytest -q 2>&1 | tail -1

step "R4 re-lock envs (conda + pip) and relink"
for env in $ENVS; do
  ~/miniforge3/bin/conda env export -n $env > envs/$env.yml 2>/dev/null
  ~/miniforge3/bin/conda list -n $env --explicit --md5 > envs/$env.explicit.txt
  $E/$env/bin/python -m pip freeze --all 2>/dev/null > envs/$env.pip.txt || true
done
PYTHONPATH=. $E/tcad/bin/python -m simws.hub.relink --canaries 2 2>&1 | grep "^relink" || true

step "R5 commit"
rm -f logs_relink.txt logs_slow.txt
git add -A simws config workflow pyproject.toml tests pipeline tools docs notes notebooks pytest.ini .gitignore \
  envs/*.yml envs/*.explicit.txt envs/*.pip.txt
git commit -q -m "simws 0.9a: hub fixes (12-digit keys, lean physics-only env lock, physics_version, batched ensemble members, store location check, sketch mode, coupling registry rule, canary relinking), cryogenic core (Altermatt freeze-out, Fermi-Dirac via Nilsson with generalized Einstein relation, Klaassen mobility with damped outer iteration, FD-consistent BGN and recombination, extended precision below 200 K), paper-fidelity mesh error bars (GCI) for the HBT adapter, Einstein/Onsager checks, flagship study A v2 77-400 K with the neutral-base-edge H1 follow-up, Charon cold-diode cross-check (Charon: $CHARON_OK)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj"
git push -q && git log --oneline -1
git remote | grep -qx offsite && git push -q offsite --all && echo "  pushed to offsite" || echo "  (no off-site remote yet: see tools/setup_offsite_backup.sh)"
echo; echo "DONE (Charon cross-check: $CHARON_OK)"
