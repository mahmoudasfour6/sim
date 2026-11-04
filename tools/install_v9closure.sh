#!/usr/bin/env bash
# 0.9 closure, batch 1 (run from ~/sim_workspace after unpacking simws_v9closure1.tgz). Stops at the first failure.
#   FROM=N bash tools/install_v9closure.sh   resumes at step N
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs; CONDA=~/miniforge3/bin/conda; L=logs_install_v9closure; mkdir -p $L
FROM=${FROM:-1}
step(){ echo; echo "== $*"; }
CORES=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['cores'])")

if [ "$FROM" -le 1 ]; then
step "1/7 packages: Kwant (tcad), Apptainer (workflow)"
$E/tcad/bin/python -c "import kwant" > /dev/null 2>&1 || $E/tcad/bin/python -c "from mpi4py import MPI; import kwant" > /dev/null 2>&1 \
  || { $CONDA install -y -q -n tcad -c conda-forge kwant > $L/conda_kwant.txt 2>&1 || { tail -n 20 $L/conda_kwant.txt; exit 1; }; }
$E/tcad/bin/python -c "from simws.validation.kwant_barrier import _import_kwant; print('  kwant', _import_kwant().__version__)"
$CONDA install -y -q -n workflow -c conda-forge apptainer > $L/conda_apptainer.txt 2>&1 \
  && echo "  apptainer $($E/workflow/bin/apptainer --version)" || echo "  apptainer not installable here (container step will try docker)"
fi

if [ "$FROM" -le 2 ]; then
step "2/7 unit tests"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= tests/unit > $L/unit.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" $L/unit.txt | head -30; exit 1; }
tail -n 1 $L/unit.txt
fi

if [ "$FROM" -le 3 ]; then
step "3/7 contact transport checks incl. Kwant vs the numpy Green's function, and WKB vs exact transmission"
PYTHONPATH=. $E/tcad/bin/python - <<'PY' 2>&1 | grep -v "RCWA\|DEVSIM\|UMFPACK\|Loading\|Skipping\|Searching" | tee $L/transport.txt
import json
from simws.validation import contact_transport_checks as C
r = C.run_all(kwant=True)
for k, v in r.items():
    if isinstance(v, dict):
        print(f"{k:18s}", json.dumps({kk: vv for kk, vv in v.items() if kk != "rows"}, default=float)[:200])
for row in r["tb_vs_wkb"]["rows"]:
    print(f"  V={row['V']}: J_exact/J_WKB = {row['ratio']:.3f}")
json.dump(r, open("logs_install_v9closure/transport.json", "w"), indent=1, default=float)
print("TRANSPORT CHECKS", "PASS" if r["pass"] else "FAIL")
PY
grep -q "TRANSPORT CHECKS PASS" $L/transport.txt || exit 1
fi

if [ "$FROM" -le 4 ]; then
step "4/7 flagship A part 2: f_T, f_max and noise from 77 to 400 K (Dask runtime; about 30-60 min)"
$E/workflow/bin/snakemake results/flagship_a_ac/flagship_a_ac.json --cores $CORES -q || { tail -n 30 logs/flagship_a_ac.log; exit 1; }
cat results/flagship_a_ac/flagship_a_ac_table.txt
fi

if [ "$FROM" -le 5 ]; then
step "5/7 container: build the image and compare reference values inside and outside (about an hour the first time)"
CONT=skipped
if bash tools/container_check.sh > $L/container.txt 2>&1; then CONT=$(grep "CONTAINER CHECK" $L/container.txt); else
  CONT="not completed ($L/container.txt)"; fi
tail -n 8 $L/container.txt; echo "  $CONT"
fi

if [ "$FROM" -le 6 ]; then
step "6/7 paper package v1 on this repository (local only, nothing uploaded)"
git add -A simws tools workflow tests notes container && git commit -q -m "0.9 closure batch 1 (pre-package commit)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj" || true
$E/workflow/bin/python tools/paper_package.py --title "simws flagship A (SiGe HBT 77-400 K)" --version 0.9.1 \
  --author "Asfour, Mahmoud|Sabanci University|" 2>&1 | tail -n 2
head -n 25 release/sim_workspace-v0.9.1/validation/VALIDATION.md
fi

step "7/7 plumbing log and commit"
$E/workflow/bin/python tools/plumbing_log.py report --release 0.9
git add -A simws tools workflow tests notes container && git commit -q -m "0.9 closure batch 1: Schottky transport with image force inside the transmission and Fermi-Dirac supply (closed-form, Kwant and exact-transmission checks), Klaassen relaxation in the AC path, flagship A f_T/f_max/noise 77-400 K, container check, paper package run, plumbing log" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj" || echo "  (nothing new to commit)"
git push -q && git log --oneline -1
echo "DONE"
