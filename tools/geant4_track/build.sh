#!/usr/bin/env bash
# Build g4track against the Geant4 in the active environment (default: the process_rad conda env).
# Uses geant4-config directly: Geant4's CMake package config also demands development headers for every
# visualization dependency (Qt, Freetype, ...), which conda runtime packages do not ship.
set -euo pipefail
ENV_PREFIX=${1:-$HOME/miniforge3/envs/process_rad}
G4C=$ENV_PREFIX/bin/geant4-config
CXX=${CXX:-$(command -v "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++" || command -v g++)}
HERE=$(cd "$(dirname "$0")" && pwd)
$CXX -O2 -std=c++17 "$HERE/g4track.cc" -o "$HERE/g4track" $($G4C --cflags) $($G4C --libs) \
     -Wl,-rpath,"$ENV_PREFIX/lib"
echo "built $HERE/g4track against Geant4 $($G4C --version)"
