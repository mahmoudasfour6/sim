#!/usr/bin/env bash
# Install the circuit engine (contract A9.9): a conda env "spice" with ngspice 47 built with OSDI support, and
# OpenVAF-Reloaded (OSDI 0.3 build) to compile Verilog-A compact models for it.
#
#     bash tools/install_spice.sh
#
# Why a new env: the ngspice on conda-forge (version 41) is built without OSDI, so it cannot load any model that
# OpenVAF compiles. This builds ngspice from its release tarball instead. Your eda and tcad envs are not touched.
# Safe to run again: finished steps are skipped.
set -euo pipefail
ENVS_ROOT="${ENVS_ROOT:-$HOME/miniforge3/envs}"
ENVP="${SPICE_ENV:-$ENVS_ROOT/spice}"
BUILD="${SPICE_BUILD:-$HOME/.cache/simws/spice-build}"
NGV=47
NG_URL="https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/$NGV/ngspice-$NGV.tar.gz"
NG_SHA=894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f
OV_FILE=openvaf-reloaded-osdi_0.3-31-gf47d557-linux_x64.tar.gz
OV_URL="https://fides.fe.uni-lj.si/openvaf/download/$OV_FILE"
OV_SHA=9bc343d52e00529ddba452b90358cdd14da14ec7437d75dec2dbd621eda54bc8

have_ng() { [ -x "$ENVP/bin/ngspice" ] && "$ENVP/bin/ngspice" --version 2>/dev/null | grep -q "ngspice-$NGV"; }
if have_ng && [ -x "$ENVP/bin/openvaf" ]; then echo "ngspice $NGV and OpenVAF are already installed in $ENVP"; exit 0; fi
CONDA="${SIMWS_CONDA:-}"
if [ -z "$CONDA" ]; then
    for c in "$HOME/miniforge3/bin/mamba" "$HOME/miniforge3/bin/conda" "$(command -v mamba || true)" "$(command -v conda || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
    done
fi
[ -n "$CONDA" ] || { echo "STOP: no conda or mamba found (set SIMWS_CONDA=/path/to/conda)"; exit 2; }
command -v curl >/dev/null || { echo "STOP: curl not found"; exit 2; }
mkdir -p "$BUILD"

fetch() {   # url file sha256
    if [ ! -f "$2" ] || ! echo "$3  $2" | sha256sum -c --quiet - 2>/dev/null; then
        curl -fsSL --retry 3 -o "$2" "$1" || { echo "STOP: could not download $1"; exit 2; }
    fi
    echo "$3  $2" | sha256sum -c --quiet - || { echo "STOP: $2 does not have the expected SHA-256"; exit 2; }
}

if [ ! -x "$ENVP/bin/x86_64-conda-linux-gnu-cc" ]; then
    echo "step 1: creating the conda env $ENVP (compilers only, a few minutes) ..."
    "$CONDA" create -y -p "$ENVP" -c conda-forge --override-channels c-compiler cxx-compiler make
fi
echo "step 1: build env ready in $ENVP"

if ! have_ng; then
    echo "step 2: downloading and building ngspice $NGV with OSDI (5 to 15 minutes) ..."
    fetch "$NG_URL" "$BUILD/ngspice-$NGV.tar.gz" "$NG_SHA"
    rm -rf "$BUILD/ngspice-$NGV"; tar xzf "$BUILD/ngspice-$NGV.tar.gz" -C "$BUILD"
    mkdir -p "$BUILD/ngspice-$NGV/release"
    LOG="$BUILD/build.log"
    if ! ( cd "$BUILD/ngspice-$NGV/release" && export PATH="$ENVP/bin:$PATH" \
           && ../configure --prefix="$ENVP" --with-x=no --with-readline=no --enable-osdi --disable-debug \
           && make -j"$(nproc 2>/dev/null || echo 2)" && make install ) > "$LOG" 2>&1; then
        echo "STOP: the ngspice build failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
    fi
fi
"$ENVP/bin/ngspice" --version | grep "ngspice-"
echo "step 2: ngspice $NGV with OSDI in $ENVP"

if [ ! -x "$ENVP/bin/openvaf" ]; then
    echo "step 3: downloading OpenVAF-Reloaded (OSDI 0.3 build) ..."
    fetch "$OV_URL" "$BUILD/$OV_FILE" "$OV_SHA"
    mkdir -p "$BUILD/openvaf"; tar xzf "$BUILD/$OV_FILE" -C "$BUILD/openvaf"
    install -m 755 "$BUILD/openvaf/openvaf" "$ENVP/bin/openvaf"
fi
"$ENVP/bin/openvaf" --version
echo "step 3: OpenVAF in $ENVP"
echo "circuit engine installed: $ENVP/bin/ngspice and $ENVP/bin/openvaf"
