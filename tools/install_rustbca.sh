#!/usr/bin/env bash
# Build RustBCA 3.0.0 from source and install its Python module (libRustBCA) into the tcad env (contract A7.8),
# and its command-line program <env>/bin/RustBCA, which the damage runs use (contract A14.3).
#
#     bash tools/install_rustbca.sh            # PY=<python of the target env> to choose another env
#
# What it does: clones https://github.com/lcpp-org/RustBCA at one fixed commit into ~/.cache/simws/rustbca-build,
# builds ONE wheel there, and installs that wheel with --no-deps; then builds the command-line program from the
# same source and copies it into the env's bin folder. Nothing else in the env is added or changed.
# Needs: git, a C linker (gcc) and a Rust compiler (cargo). It does not install any of these itself.
set -euo pipefail
PY="${PY:-$HOME/miniforge3/envs/tcad/bin/python}"
COMMIT=51bee889956147e3287491d0c68957967235a33c      # RustBCA 3.0.0 as built and tested for contract A7
REPO=https://github.com/lcpp-org/RustBCA.git
BUILD="${RUSTBCA_BUILD:-$HOME/.cache/simws/rustbca-build}"
LOG="$BUILD/build.log"

[ -x "$PY" ] || { echo "STOP: no Python at $PY (set PY=/path/to/python)"; exit 2; }
BIN="$(dirname "$PY")/RustBCA"
HAVE_MODULE=0; "$PY" -c "import libRustBCA" 2>/dev/null && HAVE_MODULE=1
if [ "$HAVE_MODULE" = 1 ] && [ -x "$BIN" ]; then
    echo "RustBCA is already installed for $PY (module and $BIN)"; exit 0
fi
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
    cat <<'MSG'
STOP: no Rust compiler (cargo) on this machine. RustBCA is written in Rust and has to be compiled once.
Install Rust with its official installer (it goes into ~/.cargo, nothing system-wide), then run this again:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
MSG
    exit 3
fi
for tool in git cc; do
    command -v "$tool" >/dev/null 2>&1 || { echo "STOP: '$tool' not found (sudo apt install git build-essential)"; exit 3; }
done

mkdir -p "$BUILD"
if [ ! -d "$BUILD/src/.git" ]; then
    echo "downloading RustBCA ..."
    git clone -q "$REPO" "$BUILD/src"
fi
git -C "$BUILD/src" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$BUILD/src" fetch -q origin
git -C "$BUILD/src" checkout -q "$COMMIT"
echo "RustBCA source at $(git -C "$BUILD/src" rev-parse --short HEAD); compiler: $(cargo --version)"

if [ "$HAVE_MODULE" = 1 ]; then
    echo "the Python module is already installed; building only the command-line program"
else
rm -rf "$BUILD/wheel"; mkdir -p "$BUILD/wheel"
echo "building (about 10 minutes the first time; log: $LOG) ..."
if ! "$PY" -m pip wheel --no-deps -w "$BUILD/wheel" "$BUILD/src" > "$LOG" 2>&1; then
    # second way: a private build environment with the two build tools, no network isolation step
    echo "  first build route failed, trying a private build environment ..." | tee -a "$LOG"
    rm -rf "$BUILD/venv" "$BUILD/wheel"; mkdir -p "$BUILD/wheel"
    if ! { "$PY" -m venv "$BUILD/venv" && "$BUILD/venv/bin/pip" install -q setuptools setuptools-rust wheel \
           && "$BUILD/venv/bin/pip" wheel --no-deps --no-build-isolation -w "$BUILD/wheel" "$BUILD/src"; } >> "$LOG" 2>&1; then
        echo "STOP: the build failed. Last lines of $LOG:"; tail -n 25 "$LOG"; exit 4
    fi
fi
WHEEL=$(ls "$BUILD"/wheel/*.whl | head -1)
echo "installing $(basename "$WHEEL") into $(dirname "$(dirname "$PY")")"
"$PY" -m pip install -q --no-deps --force-reinstall "$WHEEL"
"$PY" -c "import libRustBCA; print('RustBCA installed: libRustBCA imports')"
fi

echo "building the command-line program (a few minutes; log: $BUILD/cli.log) ..."
if ! (cd "$BUILD/src" && cargo build --release --bin RustBCA) > "$BUILD/cli.log" 2>&1; then
    echo "STOP: building the command-line program failed. Last lines of $BUILD/cli.log:"; tail -n 25 "$BUILD/cli.log"; exit 4
fi
install -m 755 "$BUILD/src/target/release/RustBCA" "$BIN"
echo "RustBCA command-line program installed: $BIN"
