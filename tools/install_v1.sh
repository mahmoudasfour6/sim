#!/usr/bin/env bash
# simws architecture v1: apply the patch on a NEW branch and test it on this machine. Master is not touched.
#   cd ~/sim_workspace && bash /path/to/install_v1.sh /path/to/simws_v1.patch
# Logs go to logs_install_v1/. Nothing is merged: review the branch, then merge it yourself.
set -uo pipefail
PATCH=${1:?usage: install_v1.sh /path/to/simws_v1.patch}
PATCH=$(readlink -f "$PATCH")
E=~/miniforge3/envs
L=logs_install_v1
BR=arch/v1
mkdir -p "$L"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "STOP: the working tree has uncommitted changes. Commit or stash them first."; exit 1
fi
if git rev-parse --verify -q "$BR" >/dev/null; then
  echo "STOP: branch $BR already exists. Delete it (git branch -D $BR) or pick another name."; exit 1
fi
BASE=$(git rev-parse --short HEAD)
echo "== base commit $BASE on $(git rev-parse --abbrev-ref HEAD); new branch $BR"
if ! git apply --check "$PATCH" 2> "$L/apply_check.txt"; then
  echo "STOP: the patch does not apply to $BASE (it was made against master a79994f). Nothing was changed."
  echo "---- paste everything from here back to Claude ----"
  head -n 12 "$L/apply_check.txt"
  for f in pytest.ini tools/nightly.sh simws/hub/model_form.py simws/adapters/__init__.py; do
    echo "$(git hash-object "$f") $(wc -c < "$f") $f"
  done
  exit 1
fi
git checkout -q -b "$BR"
git apply "$PATCH"
git add -A simws tests tools docs pytest.ini

fail=0
echo "== v1 tests + connection tests (tcad env; a few minutes)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rxXfs tests/v1 tests/connections \
  > "$L/v1.txt" 2>&1 || fail=1
tail -n 3 "$L/v1.txt"
echo "== existing unit tests (tcad env; about 10 minutes)"
PYTHONPATH=. $E/tcad/bin/python -m pytest -q -p no:cacheprovider -o addopts= -rf tests/unit \
  > "$L/unit.txt" 2>&1 || fail=1
tail -n 3 "$L/unit.txt"
echo "== Charon translator (only if Charon is installed here)"
PYTHONPATH=. $E/tcad/bin/python - > "$L/charon.txt" 2>&1 <<'EOF' || fail=1
from simws import engines
m = engines.module("charon")
print("charon available on this machine:", m.available())
print("NOTE: the Charon run/collect path has never been executed. Run tools/v1_charon_check.py to test it.")
EOF
tail -n 2 "$L/charon.txt"

if [ "$fail" = 0 ]; then
  git commit -q -m "simws architecture v1: engine-neutral description, engine translators, trust labels (installed and tested on $(hostname))"
  echo "== PASS. Committed on branch $BR ($(git rev-parse --short HEAD)). Master is unchanged."
  echo "   Expected in $L/v1.txt: 1 xfailed = the recorded R4 failure (docs/ARCHITECTURE_V1.md, amendment A1.2)."
else
  echo "== FAIL. Nothing committed; the changes are staged on branch $BR. Send back $L/v1.txt and $L/unit.txt."
fi
