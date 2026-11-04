#!/usr/bin/env bash
# Installs simws 0.9b + 0.9c (on top of the committed 0.9a) into ~/sim_workspace. Run from ~/sim_workspace after
# unpacking simws_v9bc.tgz. Everything runs on this laptop. Stops at the first failure; nothing is committed unless
# every blocking check passes. A log of each step goes to logs_install_v9bc/.
#   0.9b: Schottky contacts (thermionic, image force, WKB thermionic-field) checked against closed forms and Charon;
#         small-signal AC (Y-parameters, f_T, f_max) and impedance-field noise; one gmsh geometry for DEVSIM,
#         scikit-fem and Geant4; conservative field transfer; two-way electro-thermal coupling through preCICE.
#   0.9c: second engines (GPAW beside QE, FEniCSx beside scikit-fem, Solcore beside DEVSIM) with tolerances fixed in
#         advance and the code-to-code spread in the budgets; Dask runtime; container recipes; paper package v1;
#         SiC Schottky held-out harness (its dataset is approved separately; the gate stays pending until then).
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
CONDA=~/miniforge3/bin/conda
L=logs_install_v9bc; mkdir -p $L
FROM=${FROM:-1}          # FROM=5 bash tools/install_simws.sh: skip steps 2 and 4 on a rerun (packages, unit tests)
step(){ echo; echo "== $*"; }
DFTENV=$($E/workflow/bin/python -c "import yaml; print((yaml.safe_load(open('config/config.yaml')).get('dft') or {}).get('env', 'materials'))")
ENVS="workflow tcad quantum_iii_v process_rad $DFTENV fenicsx"

step "1/10 preconditions"
grep -q '__version__ = "0.9.0c0"' simws/__init__.py || { echo "simws/__init__.py is not 0.9c: unpack simws_v9bc.tgz first"; exit 1; }
git log --oneline -1
[ -n "$(git log --oneline -n 1 --grep='simws 0.9a')" ] || { echo "0.9a is not committed yet: finish the 0.9a install first"; exit 1; }

if [ "$FROM" -le 2 ]; then
step "2/10 packages: tcad (gmsh, shapely, meshio, preCICE, Dask, Solcore), $DFTENV (GPAW), new env fenicsx"
$CONDA install -y -q -n tcad -c conda-forge gmsh python-gmsh shapely meshio netcdf4 pyprecice precice dask \
  distributed > $L/conda_tcad.txt 2>&1 || { tail -n 20 $L/conda_tcad.txt; exit 1; }
$E/tcad/bin/python -m pip install -q "solcore==5.10.*" > $L/pip_solcore.txt 2>&1 || { tail -n 20 $L/pip_solcore.txt; exit 1; }
$E/tcad/bin/python -c "import gmsh, shapely, meshio, precice, dask.distributed, solcore, solsesame; print('  tcad: gmsh', gmsh.__version__, '| preCICE', precice.get_version_information().decode().split(';')[0], '| solcore', solcore.__version__)" 2>&1 | grep -v "RCWA\|S4"
# the hub runs materials.intrinsic in the workflow env, which lacked the hub's own dependencies (pint, pydantic)
$CONDA install -y -q -n workflow -c conda-forge pint pydantic > $L/conda_workflow.txt 2>&1 || { tail -n 20 $L/conda_workflow.txt; exit 1; }
PYTHONPATH=. $E/tcad/bin/python - <<'PYEOF'
import numpy as np, tempfile
from simws.hub.jobs import Hub
r = Hub(tempfile.mkdtemp() + "/store").submit("materials.intrinsic", params={"material": "silicon", "T": (np.array([300.0]), "K")})
print("  hub canary (materials.intrinsic in the workflow env):", r.status)
if r.status != "ok":
    raise SystemExit(r.message[-1500:])
PYEOF
$CONDA install -y -q -n $DFTENV -c conda-forge gpaw > $L/conda_gpaw.txt 2>&1 || { tail -n 20 $L/conda_gpaw.txt; exit 1; }
if ! $E/$DFTENV/bin/python -c "from gpaw.setup import create_setup; create_setup('Si', 'PBE')" > /dev/null 2>&1; then
  $E/$DFTENV/bin/python -m pip install -q gpaw-data > $L/pip_gpaw_data.txt 2>&1 || true
  $E/$DFTENV/bin/python -c "from gpaw.setup import create_setup; create_setup('Si', 'PBE')" > /dev/null 2>&1 || {
    mkdir -p sim_data/gpaw-setups
    $E/$DFTENV/bin/gpaw install-data sim_data/gpaw-setups --register > $L/gpaw_setups.txt 2>&1; }
fi
$E/$DFTENV/bin/python -c "import gpaw; from gpaw.setup import create_setup; create_setup('Si', 'PBE'); print('  $DFTENV: gpaw', gpaw.__version__, '+ PAW setups')"
[ -x $E/$DFTENV/bin/pw.x ] || command -v pw.x > /dev/null || { echo "pw.x not found in $DFTENV"; exit 1; }
if [ ! -x $E/fenicsx/bin/python ]; then
  $CONDA create -y -q -n fenicsx -c conda-forge "fenics-dolfinx>=0.8,<0.12" python-gmsh mpich numpy > $L/conda_fenicsx.txt 2>&1 \
    || { tail -n 20 $L/conda_fenicsx.txt; exit 1; }
fi
$E/fenicsx/bin/python -c "import dolfinx, gmsh; print('  fenicsx: dolfinx', dolfinx.__version__)"

fi

step "3/10 slow-tier additions and ignores"
grep -q "test_simws_gate_0p9b" tests/test_slow_solvers.py || cat tests/slow_simws_v9bc_addition.py >> tests/test_slow_solvers.py
rm -f tests/slow_simws_v9bc_addition.py
for pat in runs/ release/ "*.sif" logs_install_v9bc/; do grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore; done

if [ "$FROM" -le 4 ]; then
step "4/10 unit tests (tcad env)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/unit > $L/unit.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" $L/unit.txt | head -30; echo "unit tests failed (full log: $L/unit.txt)"; exit 1; }
tail -n 1 $L/unit.txt
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/connections > $L/conn.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" $L/conn.txt | head -30; echo "connection tests failed (full log: $L/conn.txt)"; exit 1; }
tail -n 1 $L/conn.txt
fi

CORES=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['cores'])")
MEM=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['mem_mb'])")

step "5/10 0.9b gate and 0.9c engine pairs (workflow rules gate_0p9b, engine_pairs_dft, engine_pairs; ~1 h)"
# DOLFINx compiles its forms at run time: the fenicsx env needs its own C compiler (no system gcc required)
if [ ! -x $E/fenicsx/bin/x86_64-conda-linux-gnu-cc ]; then
  $CONDA install -y -q -n fenicsx -c conda-forge c-compiler > $L/conda_fenicsx_cc.txt 2>&1 || { tail -n 20 $L/conda_fenicsx_cc.txt; exit 1; }
fi
echo "  fenicsx compiler: $($E/fenicsx/bin/x86_64-conda-linux-gnu-cc --version | head -n 1)"
$E/workflow/bin/snakemake results/gate_0p9b/gate_0p9b.json results/engine_pairs/engine_pairs.json \
  --cores $CORES --resources mem_mb=$MEM -q || { tail -n 30 logs/gate_0p9b.log logs/engine_pairs_dft.log logs/engine_pairs.log; exit 1; }
grep -E "^(contacts|small_signal|geometry|0.9b)" logs/gate_0p9b.log
grep -E "^dft:|DFT PAIR" logs/engine_pairs_dft.log
grep -E "^(fem|dd):|model-form|ENGINE PAIRS|recorded" logs/engine_pairs.log
PAIRS=$($E/workflow/bin/python -c "import json; r = json.load(open('results/engine_pairs/engine_pairs.json')); print('pass' if r['pass'] else 'FAIL (' + ', '.join(k for k in ('dft', 'fem', 'dd') if isinstance(r.get(k), dict) and r[k].get('pass') is not True) + ')')")
echo "  engine pairs: $PAIRS (reported; a failed pair is recorded, not hidden)"

step "6/10 rest of the workflow (studies, paper figures)"
$E/workflow/bin/snakemake studies --cores $CORES --resources mem_mb=$MEM -q --rerun-incomplete
$E/workflow/bin/snakemake paper --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/python tools/summarize_results.py

step "7/10 Charon cross-check incl. the Schottky cases (reported; zero-bias noise probe)"
CHARON_OK=skipped
if bash tools/run_charon_xcheck.sh > $L/charon.txt 2>&1; then CHARON_OK=pass; else
  rc=$?; [ $rc -eq 2 ] && CHARON_OK="skipped (Charon not found)" || CHARON_OK="FAIL ($L/charon.txt)"; fi
tail -n 25 $L/charon.txt; echo "  Charon cross-check: $CHARON_OK"

step "8/10 SiC Schottky held-out gate"
HELD=pending
PYTHONPATH=. $E/tcad/bin/python workflow/scripts/sic_heldout.py --outdir results/sic_schottky_heldout > $L/heldout.txt 2>&1 \
  && HELD=pass || { rc=$?; [ $rc -eq 3 ] && HELD="pending (no approved dataset yet)" || HELD="FAIL ($L/heldout.txt)"; }
tail -n 5 $L/heldout.txt; echo "  SiC held-out: $HELD"

step "9/10 slow tier: earlier results against golden values; record and confirm the 0.9b values"
$E/workflow/bin/pytest -m slow -q -k "simws and not gate_0p9b and not engine_pairs and not heldout and not charon" 2>&1 | tail -n 1
$E/workflow/bin/pytest -m slow -q -k gate_0p9b --update-golden 2>&1 | tail -n 1
$E/workflow/bin/pytest -m slow -q -k "gate_0p9b or engine_pairs or heldout or charon" -rs 2>&1 | tail -n 4 || true
$E/workflow/bin/pytest -q 2>&1 | tail -n 1

step "10/10 re-lock envs (conda + pip), relink, commit"
for env in $ENVS; do
  $CONDA env export -n $env > envs/$env.yml 2>/dev/null
  $CONDA list -n $env --explicit --md5 > envs/$env.explicit.txt
  $E/$env/bin/python -m pip freeze --all 2>/dev/null > envs/$env.pip.txt || true
done
PYTHONPATH=. $E/tcad/bin/python -m simws.hub.relink --canaries 2 2>&1 | grep "^relink" || true
git add -A simws config workflow pyproject.toml tests pipeline tools docs notes notebooks container pytest.ini .gitignore \
  envs/*.yml envs/*.explicit.txt envs/*.pip.txt
git commit -q -m "simws 0.9b+0.9c: Schottky contacts (thermionic, image force, WKB thermionic-field) vs closed forms and Charon; small-signal AC (f_T, f_max) and impedance-field noise; one gmsh geometry for DEVSIM, scikit-fem and Geant4; conservative field transfer; preCICE two-way electro-thermal coupling; second engines (GPAW, FEniCSx, Solcore) with tolerances fixed in advance and model-form budgets; Dask runtime; container recipes; paper package v1; SiC Schottky held-out harness (engine pairs: $PAIRS; Charon: $CHARON_OK; held-out: $HELD)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj"
git push -q && git log --oneline -1
git remote | grep -qx offsite && git push -q offsite --all && echo "  pushed to offsite" || echo "  (no off-site remote yet: see tools/setup_offsite_backup.sh)"
echo; echo "DONE (engine pairs: $PAIRS; Charon: $CHARON_OK; SiC held-out: $HELD)"
