#!/usr/bin/env bash
# Build the Geant4 implant program (contract A13.2) into the g4see conda env.
#
#     bash tools/install_g4implant.sh
#
# Needs the env that tools/install_g4see.sh made (Geant4 11.2.2 and its C++ compiler). Compiles
# tools/g4implant/g4implant.cc with TestEm7's G4ScreenedNuclearRecoil.cc and G4LindhardPartition.cc from that
# env's Geant4 examples, and installs <env>/bin/g4implant. Takes about a minute. Safe to run again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_ROOT="${ENVS_ROOT:-$HOME/miniforge3/envs}"
ENVP="${G4SEE_ENV:-${SIMWS_G4SEE_PREFIX:-$ENVS_ROOT/g4see}}"
G4VER=11.2.2
G4C="$ENVP/bin/geant4-config"
[ -x "$G4C" ] || { echo "STOP: no Geant4 in $ENVP (run tools/install_g4see.sh first, or set G4SEE_ENV)"; exit 2; }
V=$("$G4C" --version)
[ "$V" = "$G4VER" ] || { echo "STOP: $ENVP has Geant4 $V; contract A13 was written for $G4VER"; exit 2; }
CXX="$ENVP/bin/x86_64-conda-linux-gnu-c++"
[ -x "$CXX" ] || CXX="$ENVP/bin/c++"
[ -x "$CXX" ] || { echo "STOP: no C++ compiler in $ENVP"; exit 2; }
T="$ENVP/share/Geant4/examples/extended/electromagnetic/TestEm7"
for f in src/G4ScreenedNuclearRecoil.cc src/G4LindhardPartition.cc include/G4ScreenedNuclearRecoil.hh; do
    [ -f "$T/$f" ] || { echo "STOP: $T/$f is missing (the Geant4 examples are not in this env)"; exit 2; }
done
BUILD="${G4IMPLANT_BUILD:-$HOME/.cache/simws/g4implant-build}"
mkdir -p "$BUILD"
LOG="$BUILD/build.log"; : > "$LOG"
CFLAGS="-O2 -std=c++17 -I$T/include $($G4C --cflags)"
echo "compiling g4implant (about a minute) ..."
for f in "$HERE/g4implant/g4implant.cc" "$T/src/G4ScreenedNuclearRecoil.cc" "$T/src/G4LindhardPartition.cc"; do
    if ! "$CXX" $CFLAGS -c "$f" -o "$BUILD/$(basename "$f" .cc).o" >> "$LOG" 2>&1; then
        echo "STOP: compiling $(basename "$f") failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
    fi
done
QT=""
for q in Qt5Core Qt6Core; do [ -e "$ENVP/lib/lib$q.so" ] && { QT="-l$q"; break; }; done
if ! "$CXX" -o "$BUILD/g4implant" "$BUILD"/g4implant.o "$BUILD"/G4ScreenedNuclearRecoil.o "$BUILD"/G4LindhardPartition.o \
        $($G4C --libs) $QT -Wl,-rpath,"$ENVP/lib" >> "$LOG" 2>&1; then
    echo "STOP: linking failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
fi
install -m 755 "$BUILD/g4implant" "$ENVP/bin/g4implant"
echo "g4implant installed: $ENVP/bin/g4implant"
