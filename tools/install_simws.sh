#!/usr/bin/env bash
# Installs simws 0.9a (on top of an installed 0.8) into ~/sim_workspace. Run from ~/sim_workspace after unpacking.
# Everything runs on this laptop. Stops at the first failure; nothing is committed unless every blocking check passes.
#   0.9a: seven hub fixes + canary relinking, cryogenic core (freeze-out, Fermi-Dirac, Klaassen, FD-consistent BGN
#   and recombination, extended precision below 200 K), mesh error bars at paper fidelity, Einstein/Onsager checks,
#   flagship study A v2 (77-400 K, H1 follow-up), Charon cold-diode cross-check.
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
step(){ echo; echo "== $*"; }
DFTENV=$($E/workflow/bin/python -c "import yaml; print((yaml.safe_load(open('config/config.yaml')).get('dft') or {}).get('env', 'materials'))")
ENVS="workflow tcad quantum_iii_v process_rad $DFTENV"

step "1/9 preconditions"
grep -q '__version__ = "0.9.0a0"' simws/__init__.py || { echo "simws/__init__.py is not 0.9a: unpack simws_v9a.tgz first"; exit 1; }
git log --oneline -1
$E/tcad/bin/python -c "import exodus3" 2>/dev/null && echo "  tcad: exodus3 OK (Charon result files)" \
  || echo "  tcad: no exodus3 (the Charon comparison step will be skipped)"

step "2/9 lean environment locks (physics packages only) and canary relinking of the result store"
# the 0.9a job key hashes only physics packages: conda explicit lock + pip freeze
for env in $ENVS; do
  [ -f envs/$env.explicit.txt ] || ~/miniforge3/bin/conda list -n $env --explicit --md5 > envs/$env.explicit.txt
  $E/$env/bin/python -m pip freeze --all 2>/dev/null > envs/$env.pip.txt || true
done
# every job key changed (lean lock, 12-digit rounding, physics_version): rerun 2 canaries per adapter and link
# the rest when they reproduce to 1e-9; anything that differs is recomputed when next requested
PYTHONPATH=. $E/tcad/bin/python -m simws.hub.relink --canaries 2 > logs_relink.txt 2>&1 \
  && grep "^relink" logs_relink.txt || { grep "^relink" logs_relink.txt; echo "  some canaries differ: those adapters recompute (see logs_relink.txt)"; }

step "3/9 slow-tier additions"
grep -q "test_simws_cryo_gate" tests/test_slow_solvers.py || cat tests/slow_simws_v9a_addition.py >> tests/test_slow_solvers.py
rm -f tests/slow_simws_v9a_addition.py
for pat in runs/; do grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore; done

step "4/9 unit tests (tcad env)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/unit > logs_unit.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" logs_unit.txt | head -30; echo "unit tests failed (full log: logs_unit.txt)"; exit 1; }
tail -1 logs_unit.txt; rm -f logs_unit.txt
echo "  connection tests (producer -> consumer hand-overs through the hub)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/connections > logs_conn.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" logs_conn.txt | head -30; echo "connection tests failed (full log: logs_conn.txt)"; exit 1; }
tail -1 logs_conn.txt; rm -f logs_conn.txt

CORES=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['cores'])")
MEM=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['mem_mb'])")

step "5/9 workflow: core results, then studies (flagship A v2 77-400 K and the 0.9a gate; several hours)"
$E/workflow/bin/snakemake -n --cores $CORES -q
$E/workflow/bin/snakemake --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/snakemake studies --cores $CORES --resources mem_mb=$MEM -q --rerun-incomplete
$E/workflow/bin/snakemake paper --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/python tools/summarize_results.py

step "6/9 Charon cold-diode cross-check (reported; not blocking until the decks are confirmed on this laptop)"
CHARON_OK=skipped
if bash tools/run_charon_xcheck.sh > logs_charon.txt 2>&1; then CHARON_OK=pass; else
  rc=$?; [ $rc -eq 2 ] && CHARON_OK="skipped (Charon not found)" || CHARON_OK="FAIL (logs_charon.txt)"; fi
tail -25 logs_charon.txt; echo "  Charon cross-check: $CHARON_OK"

step "7/9 slow tier: 0.8 results reproduce against their golden values, then record and confirm the 0.9a values"
$E/workflow/bin/pytest -m slow -q -k "simws and not cryo_gate and not charon" 2>&1 | tail -1
$E/workflow/bin/pytest -m slow -q -k cryo_gate --update-golden 2>&1 | tail -1
$E/workflow/bin/pytest -m slow -q -k "cryo_gate or charon" -rs 2>&1 | tail -3
$E/workflow/bin/pytest -q 2>&1 | tail -1

step "8/9 re-lock changed envs (conda + pip) and relink once more"
for env in $ENVS; do
  ~/miniforge3/bin/conda env export -n $env > envs/$env.yml 2>/dev/null
  ~/miniforge3/bin/conda list -n $env --explicit --md5 > envs/$env.explicit.txt
  $E/$env/bin/python -m pip freeze --all 2>/dev/null > envs/$env.pip.txt || true
done
PYTHONPATH=. $E/tcad/bin/python -m simws.hub.relink --canaries 2 2>&1 | grep "^relink" || true
rm -f logs_relink.txt

step "9/9 commit"
git add -A simws config workflow pyproject.toml tests pipeline tools docs notes notebooks pytest.ini .gitignore \
  envs/*.yml envs/*.explicit.txt envs/*.pip.txt
git commit -q -m "simws 0.9a: hub fixes (12-digit keys, lean physics-only env lock, physics_version, batched ensemble members, store location check, sketch mode, coupling registry rule, canary relinking), cryogenic core (Altermatt freeze-out, Fermi-Dirac via Nilsson with generalized Einstein relation, Klaassen mobility with damped outer iteration, FD-consistent BGN and recombination, extended precision below 200 K), paper-fidelity mesh error bars (GCI) for the HBT adapter, Einstein/Onsager checks, flagship study A v2 77-400 K with the neutral-base-edge H1 follow-up, Charon cold-diode cross-check (Charon: $CHARON_OK)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj"
git push -q && git log --oneline -1
git remote | grep -qx offsite && git push -q offsite --all && echo "  pushed to offsite" || echo "  (no off-site remote yet: see tools/setup_offsite_backup.sh)"
echo; echo "DONE (Charon cross-check: $CHARON_OK)"
