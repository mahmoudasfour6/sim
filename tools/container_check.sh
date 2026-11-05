#!/usr/bin/env bash
# Build the simws image and check that the reference values inside equal those outside (0.9 closure).
#   bash tools/container_check.sh            (from ~/sim_workspace, after the commit that should be in the image)
# Uses Apptainer if present (conda-forge package `apptainer`, unprivileged build with --fakeroot), else Docker.
set -euo pipefail
cd ~/sim_workspace
E=~/miniforge3/envs
mkdir -p runs/container
rm -f runs/container/missing.txt runs/container/RESULT
PYTHONPATH=. $E/tcad/bin/python tools/reference_values.py runs/container/host.json > /dev/null 2>&1
if command -v apptainer > /dev/null || [ -x $E/workflow/bin/apptainer ]; then
  AP=$(command -v apptainer || echo $E/workflow/bin/apptainer)
  # without newuidmap (no uidmap package, i.e. no root access) Apptainer can still build in a root-mapped user
  # namespace: --ignore-subuid. The recipe needs no apt-get, so nothing inside the build requires real root.
  FR="--fakeroot"; command -v newuidmap > /dev/null || FR="--fakeroot --ignore-subuid"
  # Apptainer's working space (build tree, and the unpacked image when squashfuse is absent) and its layer cache stay
  # on the workspace disk, for the build and for every run below
  export APPTAINER_TMPDIR=$PWD/runs/container/tmp APPTAINER_CACHEDIR=$PWD/runs/container/cache
  mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
  if [ ! -f runs/container/simws.sif ] && [ ! -f runs/container/simws.new.sif ]; then
    rm -rf $APPTAINER_TMPDIR/build-temp-* 2>/dev/null || true          # leftovers of a failed build
    # preflight: the envs as installed on the host x 1.6 (unpacked tree + compressed .sif) + 5 GB of headroom.
    # Under WSL the Linux disk is a VHDX that grows into the Windows drive, so the Windows drive's free space counts.
    need=$({ du -scm $E/workflow $E/tcad $E/materials $E/process_rad $E/quantum_iii_v $E/fenicsx 2>/dev/null || true; } \
           | tail -n1 | awk '{printf "%d", $1 * 1.6 / 1024 + 5}')
    free=$(df -Pm $APPTAINER_TMPDIR | awk 'NR==2{printf "%d", $4 / 1024}')
    if [ -d /mnt/c ]; then
      wfree=$(df -Pm /mnt/c | awk 'NR==2{printf "%d", $4 / 1024}')
      if [ "$wfree" -lt "$free" ]; then free=$wfree; fi
    fi
    echo "container build: need about ${need} GB, free ${free} GB (Linux disk and Windows C: drive, the smaller)"
    if [ "$free" -lt "$need" ]; then
      echo "NOT ENOUGH DISK SPACE for the image build: free ${free} GB, need about ${need} GB."
      echo "  Linux side:  df -h ~ ;  du -sh ~/miniforge3/pkgs ~/.apptainer ~/.cache/pip"
      echo "  Windows side: free space on C: (the WSL disk grows into it)"
      exit 4
    fi
    # --notest: a failing %test must not throw away an hour of build; the image is tested below and only
    # renamed to simws.sif when it passes (a rerun after a test failure re-tests simws.new.sif without rebuilding)
    $AP build --notest $FR runs/container/simws.new.sif container/simws.def
    rm -rf $APPTAINER_TMPDIR/build-temp-* 2>/dev/null || true
  fi
  SIF=runs/container/simws.sif; [ -f $SIF ] || SIF=runs/container/simws.new.sif
  # without squashfuse, Apptainer unpacks the whole image for every single run: unpack it once into a sandbox
  # directory and run the checks there (removed again after a pass)
  RUN=$SIF; SB=runs/container/sandbox
  if ! command -v squashfuse > /dev/null; then
    if [ ! -f $SB/.from ] || [ "$(cat $SB/.from)" != "$(stat -c %Y $SIF)" ]; then
      rm -rf $SB; echo "unpacking the image once (no squashfuse) ..."
      $AP build --sandbox $SB $SIF > /dev/null
      stat -c %Y $SIF > $SB/.from
    fi
    RUN=$SB
  fi
  if [ $SIF = runs/container/simws.new.sif ]; then
    echo "container self-test (%test, host environment excluded):"
    $AP test --cleanenv $RUN
    mv runs/container/simws.new.sif runs/container/simws.sif
    [ ! -d $SB ] || stat -c %Y runs/container/simws.sif > $SB/.from
  fi
  $AP exec --cleanenv --bind $PWD/runs/container:/out $RUN bash -lc \
    'cd /opt/sim_workspace && PYTHONPATH=. /opt/conda/envs/tcad/bin/python tools/reference_values.py /out/image.json' \
    > runs/container/image_run.log 2>&1 || { echo "reference values inside the image failed:"; tail -n 20 runs/container/image_run.log; exit 1; }
  $AP exec --cleanenv $RUN cat /opt/sim_workspace/CONTAINER_MISSING.txt > runs/container/missing.txt 2>/dev/null || true
elif command -v docker > /dev/null; then
  docker build -f container/Dockerfile -t simws:check .
  docker run --rm -v $PWD/runs/container:/out simws:check bash -lc \
    'cd /opt/sim_workspace && PYTHONPATH=. /opt/conda/envs/tcad/bin/python tools/reference_values.py /out/image.json' \
    > runs/container/image_run.log 2>&1 || { echo "reference values inside the image failed:"; tail -n 20 runs/container/image_run.log; exit 1; }
  docker run --rm simws:check cat /opt/sim_workspace/CONTAINER_MISSING.txt > runs/container/missing.txt 2>/dev/null || true
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
import os
miss = open("runs/container/missing.txt").read().strip() if os.path.isfile("runs/container/missing.txt") else ""
print("pip packages not in the image:", miss or "none")
ok = worst < 1e-9
open("runs/container/RESULT", "w").write("PASS" if ok else "FAIL")
PY
# the unpacked sandbox is about the size of the envs: drop it after a pass (the .sif stays)
if [ "$(cat runs/container/RESULT 2>/dev/null)" = PASS ] && [ -d runs/container/sandbox ]; then
  chmod -R u+w runs/container/sandbox 2>/dev/null; rm -rf runs/container/sandbox runs/container/tmp
fi
