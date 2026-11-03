#!/usr/bin/env bash
# 0.9a gate: DEVSIM vs Charon on a cold pn diode (simws.validation.charon_xcheck). Runs on the laptop.
#   bash tools/run_charon_xcheck.sh [OUT_DIR]          (default ~/sim_workspace/runs/charon_xcheck)
# Environment overrides: CHARON_ROOT (default ~/tcad), CHARON_EXE, CHARON_INTERP, CHARON_TEMPS ("77 300")
set -euo pipefail
WS=${WS:-$HOME/sim_workspace}
OUT=${1:-$WS/runs/charon_xcheck}
ROOT=${CHARON_ROOT:-$HOME/tcad}
TEMPS=${CHARON_TEMPS:-"77 300"}
# installed Charon executables and the interpreter are usually symlinks into the build tree: accept links too
findx(){ find -L "$@" 2>/dev/null | head -1; }
EXE=${CHARON_EXE:-$(command -v charon_mp.exe 2>/dev/null || findx "$ROOT" -name 'charon_mp.exe' -type f)}
INTERP=${CHARON_INTERP:-$(command -v charonInterpreter.py 2>/dev/null || findx "$ROOT" -name charonInterpreter.py -type f -not -path '*parseGenerator*')}
MESH=$(findx "$ROOT" -path '*capability*' -name pndiode.exo -type f)
[ -x "$EXE" ] || { echo "charon_mp.exe not found on PATH or under $ROOT (set CHARON_EXE=/path/to/charon_mp.exe)"; exit 2; }
[ -f "$INTERP" ] || { echo "charonInterpreter.py not found on PATH or under $ROOT (set CHARON_INTERP)"; exit 2; }
[ -f "$MESH" ] || { echo "pndiode.exo not found under $ROOT (Charon test/nightlyTests/capability)"; exit 2; }
echo "charon: $EXE"; echo "interpreter: $INTERP"; echo "mesh: $MESH"; echo "out: $OUT"
mkdir -p "$OUT"; cp -f "$MESH" "$OUT/pndiode.exo"
source "$(conda info --base 2>/dev/null || echo ~/miniforge3)/etc/profile.d/conda.sh"
conda activate tcad
cd "$WS"
python -m simws.validation.charon_xcheck write "$OUT" --T $TEMPS
cd "$OUT"
for T in $TEMPS; do
  for d in $(cat "runlist_T${T}.txt"); do
    if [ -f "$d.exo" ] && [ "$d.exo" -nt "$d.inp" ]; then echo "  $d: done"; continue; fi
    echo "  $d"
    python "$INTERP" -i "$d.inp" > "$d.interp.log" 2>&1 || { echo "interpreter failed: $OUT/$d.interp.log"; exit 3; }
    "$EXE" --i="$d.inp.xml" > "$d.log" 2>&1 || { echo "charon failed: $OUT/$d.log (last lines below)"; tail -20 "$d.log"; exit 4; }
  done
done
cd "$WS"
python -m simws.validation.charon_xcheck devsim "$OUT" --T $TEMPS
python -m simws.validation.charon_xcheck compare "$OUT" --T $TEMPS
