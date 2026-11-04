#!/usr/bin/env bash
# Build the simws image and check that the reference values inside equal those outside (0.9 closure).
#   bash tools/container_check.sh            (from ~/sim_workspace, after the commit that should be in the image)
# Uses Apptainer if present (conda-forge package `apptainer`, unprivileged build with --fakeroot), else Docker.
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
mkdir -p runs/container
PYTHONPATH=. $E/tcad/bin/python tools/reference_values.py runs/container/host.json > /dev/null 2>&1
if command -v apptainer > /dev/null || [ -x $E/workflow/bin/apptainer ]; then
  AP=$(command -v apptainer || echo $E/workflow/bin/apptainer)
  [ -f runs/container/simws.sif ] || $AP build --fakeroot runs/container/simws.sif container/simws.def
  $AP exec --bind $PWD/runs/container:/out runs/container/simws.sif bash -lc \
    'cd /opt/sim_workspace && PYTHONPATH=. /opt/conda/envs/tcad/bin/python tools/reference_values.py /out/image.json' > /dev/null 2>&1
elif command -v docker > /dev/null; then
  docker build -f container/Dockerfile -t simws:check .
  docker run --rm -v $PWD/runs/container:/out simws:check bash -lc \
    'cd /opt/sim_workspace && PYTHONPATH=. /opt/conda/envs/tcad/bin/python tools/reference_values.py /out/image.json' > /dev/null 2>&1
else
  echo "neither apptainer nor docker found: conda install -n workflow -c conda-forge apptainer"; exit 2
fi
$E/workflow/bin/python - <<'PY'
import json
h = json.load(open("runs/container/host.json")); i = json.load(open("runs/container/image.json"))
worst = 0.0
for k in h:
    r = abs(i[k] / h[k] - 1) if h[k] else abs(i[k])
    worst = max(worst, r); print(f"  {k:28s} host {h[k]:.12g}  image {i[k]:.12g}  rel {r:.1e}")
print("CONTAINER CHECK", "PASS" if worst < 1e-9 else "FAIL", f"(worst {worst:.1e}, gate 1e-9)")
PY
