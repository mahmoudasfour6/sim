#!/usr/bin/env bash
# Install the G4SEE engine (contract A8.8): a conda env "g4see" with Geant4 11.2.2, and G4SEE v1.0 built into it.
#
#     bash tools/install_g4see.sh
#
# 1. creates the env (conda-forge: geant4 11.2.2, a C++ compiler); the first time this downloads a few GB of
#    Geant4 physics data. It does not touch your other envs (tcad, process_rad, ...).
# 2. downloads G4SEE at tag v1.0 from CERN GitLab into ~/.cache/simws/g4see-build
# 3. compiles it with geant4-config (Geant4's CMake files ask for development headers that the conda packages
#    do not ship; tools/geant4_track/build.sh does the same) and puts g4see and mergeHistograms into the env.
# Safe to run again: finished steps are skipped.
set -euo pipefail
ENVS_ROOT="${ENVS_ROOT:-$HOME/miniforge3/envs}"
ENVP="${G4SEE_ENV:-$ENVS_ROOT/g4see}"
BUILD="${G4SEE_BUILD:-$HOME/.cache/simws/g4see-build}"
TAG=v1.0
COMMIT=8c6fc45                                    # the commit of tag v1.0 that contract A8 was tested with
REPO=https://gitlab.cern.ch/g4see/g4see.git
G4VER=11.2.2

if [ -x "$ENVP/bin/g4see" ] && [ "$("$ENVP/bin/geant4-config" --version 2>/dev/null)" = "$G4VER" ]; then
    echo "G4SEE is already installed in $ENVP"; exit 0
fi
CONDA="${SIMWS_CONDA:-}"
if [ -z "$CONDA" ]; then
    for c in "$HOME/miniforge3/bin/mamba" "$HOME/miniforge3/bin/conda" "$(command -v mamba || true)" "$(command -v conda || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
    done
fi
[ -n "$CONDA" ] || { echo "STOP: no conda or mamba found (set SIMWS_CONDA=/path/to/conda)"; exit 2; }
command -v git >/dev/null || { echo "STOP: git not found (sudo apt install git)"; exit 2; }

if [ ! -x "$ENVP/bin/geant4-config" ]; then
    echo "step 1: creating the conda env $ENVP with Geant4 $G4VER (several GB, 10 to 30 minutes) ..."
    "$CONDA" create -y -p "$ENVP" -c conda-forge --override-channels "geant4=$G4VER" cxx-compiler make
fi
V=$("$ENVP/bin/geant4-config" --version)
[ "$V" = "$G4VER" ] || { echo "STOP: $ENVP has Geant4 $V; G4SEE $TAG needs $G4VER. Remove that env or set G4SEE_ENV."; exit 2; }
echo "step 1: Geant4 $V in $ENVP"

mkdir -p "$BUILD"
if [ ! -d "$BUILD/src/.git" ]; then
    echo "step 2: downloading G4SEE $TAG ..."
    git clone -q "$REPO" "$BUILD/src"
fi
git -C "$BUILD/src" fetch -q --tags origin 2>/dev/null || true
git -C "$BUILD/src" checkout -q "$TAG"
GOT=$(git -C "$BUILD/src" rev-parse --short=7 HEAD)
[ "$GOT" = "$COMMIT" ] || { echo "STOP: tag $TAG is commit $GOT here, contract A8 was tested with $COMMIT"; exit 2; }
echo "step 2: G4SEE source at $TAG ($GOT)"

echo "step 3: compiling G4SEE (5 to 15 minutes) ..."
SRC="$BUILD/src"; OBJ="$BUILD/obj"; mkdir -p "$OBJ/include"
CXX="$ENVP/bin/x86_64-conda-linux-gnu-c++"
[ -x "$CXX" ] || CXX="$ENVP/bin/c++"
G4C="$ENVP/bin/geant4-config"
cat > "$OBJ/include/version.hh" <<VER
#ifndef VERSION_HH
#define VERSION_HH
#include <string>
const std::string kGitVersion = "$TAG";
const std::string kGitTag = "$TAG";
const std::string kGitCommitID = "$GOT";
const std::string kGitBranch = "$TAG";
#endif
VER
CFLAGS="-O2 -std=c++17 -I$SRC/include -I$OBJ/include -I$OBJ $($G4C --cflags)"
export CXX CFLAGS OBJ
LOG="$BUILD/build.log"; : > "$LOG"
NP=$(nproc 2>/dev/null || echo 2)
if ! ls "$SRC"/src/*.cc "$SRC/g4see.cc" | xargs -P "$NP" -I{} sh -c '$CXX $CFLAGS -c "$1" -o "$OBJ/$(basename "$1" .cc).o"' _ {} >> "$LOG" 2>&1; then
    echo "STOP: compiling failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
fi
QT=""
for q in Qt5Core Qt6Core; do [ -e "$ENVP/lib/lib$q.so" ] && { QT="-l$q"; break; }; done
if ! { "$CXX" -o "$OBJ/g4see" "$OBJ"/*.o $($G4C --libs) $QT -Wl,-rpath,"$ENVP/lib" \
       && "$CXX" -O2 -std=c++17 -I"$SRC/include" -I"$OBJ/include" -I"$OBJ" "$SRC/mergeHistograms.cc" -o "$OBJ/mergeHistograms"; } >> "$LOG" 2>&1; then
    echo "STOP: linking failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
fi
install -m 755 "$OBJ/g4see" "$OBJ/mergeHistograms" "$ENVP/bin/"
"$ENVP/bin/g4see" -v 2>&1 | grep -i version | head -1
echo "G4SEE installed: $ENVP/bin/g4see"
