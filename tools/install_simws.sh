#!/usr/bin/env bash
# Installs the simws infrastructure (upgrades 1-9; simws 0.8) into ~/sim_workspace. Run from the repo root after unpacking.
# Everything runs on this laptop. Stops at the first failure; nothing is committed unless every check passes.
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
step(){ echo; echo "== $*"; }

step "1/9 retire legacy runners (git mv, history kept)"
for f in heavy_sim_runner.py run_all_6_workspaces_pipeline.py run_level6_autonomous_suite.py \
         run_multiscale_pipeline.py run_spatial_coupled_pipeline.py Snakefile sim_mpi_launcher.sh; do
  [ -e pipeline/$f ] && git mv pipeline/$f pipeline/legacy/$f && echo "  moved pipeline/$f"
done

step "2/9 Python dependencies"
$E/tcad/bin/pip install -q "scikit-fem==12.0.2"
$E/tcad/bin/python -c "import matplotlib, meshio" 2>/dev/null || $E/tcad/bin/pip install -q matplotlib meshio
$E/quantum_iii_v/bin/pip install -q pint
$E/process_rad/bin/python -c "import pint, yaml, numpy" 2>/dev/null || $E/process_rad/bin/pip install -q pint pyyaml numpy
DFTENV=$($E/workflow/bin/python -c "import yaml; print((yaml.safe_load(open('config/config.yaml')).get('dft') or {}).get('env', 'materials'))")
$E/$DFTENV/bin/python -c "import pint, yaml, numpy" 2>/dev/null || $E/$DFTENV/bin/pip install -q pint pyyaml numpy
ls $E/$DFTENV/bin/pw.x >/dev/null || { echo "pw.x not found in env '$DFTENV': set dft.env in config/config.yaml"; exit 1; }
$E/$DFTENV/bin/python -c "import chgnet" 2>/dev/null && echo "  $DFTENV: CHGNet available (benchmark runs)" \
  || echo "  $DFTENV: no CHGNet (the CHGNet benchmark is skipped; optional: $E/$DFTENV/bin/pip install chgnet)"
# simws 0.8 hub: typed envelopes (pydantic v2) and the knowledge graph (networkx) in every env that runs adapters
for env in workflow tcad process_rad $DFTENV; do
  $E/$env/bin/python -c "import pydantic, networkx; assert int(pydantic.VERSION.split('.')[0]) >= 2" 2>/dev/null \
    || $E/$env/bin/pip install -q "pydantic>=2.5,<3" "networkx>=3.1"
  $E/$env/bin/python -c "import pydantic, networkx; print('  $env: pydantic', pydantic.VERSION, 'networkx', networkx.__version__)"
done
# PseudoDojo v0.4 (ONCVPSP, norm-conserving) Si pseudopotentials: the default DFT set from 0.8 on when the fetch works
if bash tools/fetch_pseudodojo.sh sim_data/pseudopotentials/pseudodojo_v0.4; then
  grep -q "pseudo_set:" config/config.yaml || sed -i 's|^dft:$|dft:\n  pseudo_set: pseudodojo  # bundled = the 0.7 set in simws/dft/pseudo; pseudodojo = sim_data/pseudopotentials/pseudodojo_v0.4|' config/config.yaml
else
  echo "  keeping the bundled pseudopotentials (DFT results flagged 'pseudos:bundled')"
fi
grep -n "pseudo_set" config/config.yaml || true
for env in tcad quantum_iii_v process_rad $DFTENV; do
  mods="yaml,pint,numpy"; [ $env = tcad ] && mods="yaml,pint,numpy,scipy,meshio,matplotlib,skfem,devsim"
  $E/$env/bin/python -c "import ${mods}; print('  $env imports OK')"
done

step "3/9 build the Geant4 ion-track tool (process_rad)"
ls $E/process_rad/bin/x86_64-conda-linux-gnu-c++ >/dev/null 2>&1 || command -v g++ >/dev/null \
  || ~/miniforge3/bin/mamba install -y -q -n process_rad -c conda-forge cxx-compiler
bash tools/geant4_track/build.sh $E/process_rad

step "4/9 slow-tier additions, test collection, gitignore"
grep -q "test_simws_unit_suite_in_tcad_env" tests/test_slow_solvers.py || cat tests/slow_simws_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_axisymmetric_seu" tests/test_slow_solvers.py || cat tests/slow_simws_v3_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_geant4_event_seu" tests/test_slow_solvers.py || cat tests/slow_simws_v4_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_mos_and_tid" tests/test_slow_solvers.py || cat tests/slow_simws_v5_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_sige_checks" tests/test_slow_solvers.py || cat tests/slow_simws_v6_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_dft_params_quick" tests/test_slow_solvers.py || cat tests/slow_simws_v7_addition.py >> tests/test_slow_solvers.py
grep -q "test_simws_hub_gate_devices" tests/test_slow_solvers.py || cat tests/slow_simws_v8_addition.py >> tests/test_slow_solvers.py
# the 0.7 DFT golden values were recorded with the bundled pseudopotentials: pin that test to them
sed -i 's|"--ranks", "4"\], REPO, timeout=3600, threads=1, extra_env=_simws_env())|"--ranks", "4"], REPO, timeout=3600, threads=1, extra_env={**_simws_env(), "SIMWS_DFT_PSEUDOS": "bundled"})|' tests/test_slow_solvers.py
grep -q "SIMWS_DFT_PSEUDOS" tests/test_slow_solvers.py || { echo "could not pin the 0.7 DFT test to the bundled pseudopotentials"; exit 1; }
rm -f tests/slow_simws_addition.py tests/slow_simws_v3_addition.py tests/slow_simws_v4_addition.py tests/slow_simws_v5_addition.py \
      tests/slow_simws_v6_addition.py tests/slow_simws_v7_addition.py tests/slow_simws_v8_addition.py
# unit tests need the tcad env: keep the workflow-env runs from collecting (and skipping) them; the slow tier
# runs them in tcad with the ini options overridden
grep -q -- "--ignore=tests/unit" pytest.ini || sed -i 's|^addopts = \(.*\)$|addopts = \1 --ignore=tests/unit|' pytest.ini
grep -q -- "--ignore=tests/connections" pytest.ini || sed -i 's|^addopts = \(.*\)$|addopts = \1 --ignore=tests/connections|' pytest.ini
sed -i 's|"-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/unit"\]|"-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "tests/unit"]|' tests/test_slow_solvers.py
grep -n "^addopts" pytest.ini
for pat in results/ logs/ .snakemake/ release/ tools/geant4_track/g4track store/ sim_data/pseudopotentials/pseudodojo_v0.4/; do grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore; done

step "5/9 unit tests (tcad env)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -rf tests/unit > logs_unit.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" logs_unit.txt | head -30; echo "unit tests failed (full log: logs_unit.txt)"; exit 1; }
tail -1 logs_unit.txt; rm -f logs_unit.txt
echo "  connection tests (producer -> consumer hand-overs through the hub)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/connections > logs_conn.txt 2>&1 \
  || { grep -E "^(FAILED|ERROR|E  )" logs_conn.txt | head -30; echo "connection tests failed (full log: logs_conn.txt)"; exit 1; }
tail -1 logs_conn.txt; rm -f logs_conn.txt

step "6/9 benchmark this laptop -> config/local_tuning.yaml (about 10 min)"
if [ -f config/local_tuning.yaml ]; then
  echo "  already benchmarked ($(grep -m1 measured config/local_tuning.yaml)); delete config/local_tuning.yaml to redo"
else
  PYTHONPATH=. $E/tcad/bin/python tools/bench_local.py 2>&1 | grep -E "^(machine|QE|  QE|DEVSIM|  sweep|recommend|wrote)"
fi
CORES=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['cores'])")
MEM=$($E/workflow/bin/python -c "import yaml; print(yaml.safe_load(open('config/local_tuning.yaml'))['snakemake']['mem_mb'])")

step "7/9 workflow: core results, then studies (sweep + verification)"
$E/workflow/bin/snakemake -n --cores $CORES -q
$E/workflow/bin/snakemake --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/snakemake studies --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/snakemake paper --cores $CORES --resources mem_mb=$MEM -q
$E/workflow/bin/python tools/summarize_results.py

step "8/9 slow tier: record golden values for the new checks, then confirm"
$E/workflow/bin/pytest -m slow -q -k simws --update-golden 2>&1 | tail -1
$E/workflow/bin/pytest -m slow -q -k simws 2>&1 | tail -1
$E/workflow/bin/pytest -q 2>&1 | tail -1

step "9/9 re-lock changed envs and commit"
for env in workflow tcad quantum_iii_v process_rad $DFTENV; do
  ~/miniforge3/bin/conda env export -n $env > envs/$env.yml 2>/dev/null
  ~/miniforge3/bin/conda list -n $env --explicit --md5 > envs/$env.explicit.txt
done
mkdir -p notes && [ -f notes/README.md ] || $E/workflow/bin/python tools/study.py index
git add -A simws config workflow pyproject.toml tests pipeline tools docs notes notebooks pytest.ini .gitignore envs/*.yml envs/*.explicit.txt
git commit -q -m "simws 0.8: hub core contract - typed envelopes, canonical job keys, content-addressed store, SQLite knowledge graph with lineage and staleness, gatekeeper, isolated adapters (DEVSIM HBT/MOS/strike/MMS, TID, SER, Quantum ESPRESSO band edges, material laws), conventions registry, correlated ensembles and error budgets, physics guards, MMS code verification, material laws with validity boxes, Al2O3/HfO2/interface distributions, PseudoDojo pseudopotentials, citations, overnight queue, retention pruning, flagship study A screening" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gc1UGcUaEy5777LUU4pPwj"
git push -q && git log --oneline -1
git remote | grep -qx offsite && git push -q offsite --all && echo "  pushed to offsite" || echo "  (no off-site remote yet: see tools/setup_offsite_backup.sh)"
WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
cp tools/register_nightly.ps1 "/mnt/c/Users/$WINUSER/sim_backup/register_nightly.ps1" 2>/dev/null || true
echo; echo "DONE. To schedule the nightly run, open Windows PowerShell and run:"
echo "  powershell -ExecutionPolicy Bypass -File \$env:USERPROFILE\\sim_backup\\register_nightly.ps1"
