#!/usr/bin/env bash
# 0.9a gate: DEVSIM vs Charon on a cold pn diode (simws.validation.charon_xcheck). Runs on the laptop.
#   bash tools/run_charon_xcheck.sh [OUT_DIR]          (default ~/sim_workspace/runs/charon_xcheck)
# Environment overrides: CHARON_ROOT (default ~/tcad), CHARON_EXE, CHARON_INTERP
set -uo pipefail
WS=${WS:-$HOME/sim_workspace}
OUT=${1:-$WS/runs/charon_xcheck}
ROOT=${CHARON_ROOT:-$HOME/tcad}
# installed Charon executables and the interpreter are usually symlinks into the build tree: accept links too
findx(){ find -L "$@" 2>/dev/null | head -1; }
EXE=${CHARON_EXE:-$(command -v charon_mp.exe 2>/dev/null || findx "$ROOT" -name 'charon_mp.exe' -type f)}
INTERP=${CHARON_INTERP:-$(command -v charonInterpreter.py 2>/dev/null || findx "$ROOT" -name charonInterpreter.py -type f -not -path '*parseGenerator*')}
[ -x "$EXE" ] || { echo "charon_mp.exe not found on PATH or under $ROOT (set CHARON_EXE=/path/to/charon_mp.exe)"; exit 2; }
[ -f "$INTERP" ] || { echo "charonInterpreter.py not found on PATH or under $ROOT (set CHARON_INTERP)"; exit 2; }
echo "charon: $EXE"; echo "interpreter: $INTERP"; echo "out: $OUT"
mkdir -p "$OUT"                             # the meshes come from simws/validation/data (written by write)
source "$(conda info --base 2>/dev/null || echo ~/miniforge3)/etc/profile.d/conda.sh"
conda activate tcad
cd "$WS"
python -m simws.validation.charon_xcheck write "$OUT" > /dev/null || exit 3
cd "$OUT"
MPI=$(command -v mpirun || true)
: > status.txt
for d in $(cat runlist_all.txt); do
  src=$(head -1 "$d.inp" | awk '{print $4}')
  if [ ! -f "$src" ]; then echo "skip  $d (no $src)" | tee -a status.txt; continue; fi
  if [ -f "$d.exo" ] && [ "$d.exo" -nt "$d.inp" ]; then echo "done  $d" >> status.txt; continue; fi
  python "$INTERP" -i "$d.inp" > "$d.interp.log" 2>&1 || { echo "INTERP-FAIL $d" | tee -a status.txt; continue; }
  if ${MPI:+$MPI -np 1} "$EXE" --i="$d.inp.xml" --current > "$d.log" 2>&1; then
    echo "ok    $d" | tee -a status.txt
  else
    rm -f "$d.exo"                          # an aborted run leaves a partial file
    echo "FAIL  $d  ($(grep -E '^\|\|F\|\|' "$d.log" | tail -1))" | tee -a status.txt
  fi
done
cd "$WS"
python -m simws.validation.charon_xcheck devsim "$OUT" > "$OUT/devsim.log" 2>&1 || { echo "DEVSIM side failed: $OUT/devsim.log"; tail -5 "$OUT/devsim.log"; }
python -m simws.validation.charon_xcheck compare "$OUT" 2>/dev/null | grep -v -i "exodus\|copyright\|engineering\|contract\|rights in\|^$\|You are using"
rc=${PIPESTATUS[0]}
# hand the logs (and Charon's SRH model source, for the recombination comparison) over for review
WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
DBG=/mnt/c/Users/$WINUSER/sim_backup/charon_dbg/xcheck
if [ -n "$WINUSER" ]; then
  mkdir -p "$DBG"; cp -f "$OUT"/*.log "$OUT"/status.txt "$OUT"/*.json "$DBG/" 2>/dev/null
  for f in Charon_RecombRate_SRH_impl.hpp Charon_RecombRate_SRH_decl.hpp Charon_RecombRate_Total_impl.hpp; do
    p=$(find -L "$ROOT" -name "$f" 2>/dev/null | head -1); [ -n "$p" ] && cp -f "$p" "$DBG/"
  done
  echo "logs copied to $DBG"
fi
exit $rc
