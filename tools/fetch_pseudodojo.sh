#!/usr/bin/env bash
# Fetch the PseudoDojo v0.4 norm-conserving (ONCVPSP) Si pseudopotentials in UPF format for Quantum ESPRESSO.
#   bash tools/fetch_pseudodojo.sh [target dir]        default: sim_data/pseudopotentials/pseudodojo_v0.4
# Writes (as available) Si.pbe.upf (scalar-relativistic PBE), Si_r.pbe.upf (fully relativistic PBE, spin-orbit), Si.pbesol.upf,
# Si.pw.upf (LDA, Perdew-Wang) and SOURCES.txt with the SHA-256 of each file. Exit 0 with Si.pbe.upf present means
# the set is usable (simws then defaults to it: dft.pseudo_set = pseudodojo); otherwise the bundled set stays in use
# and every DFT result carries the flag 'pseudos:bundled'.
# Sources: the pseudo-dojo.org table archives (nc-sr-04_<xc>_standard_upf.tgz, nc-fr-04_pbe_standard_upf.tgz); the
# fully relativistic PBE file also from the PseudoDojo GitHub repository (pinned commit, checksum verified).
# Cite: van Setten et al., Comput. Phys. Commun. 226, 39 (2018); Hamann, Phys. Rev. B 88, 085117 (2013).
set -u
T=${1:-sim_data/pseudopotentials/pseudodojo_v0.4}
mkdir -p "$T"; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
BASE=http://www.pseudo-dojo.org/pseudos
FR_GH=https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/7aa01a3fcf5ad226caf25bd387a9be9612be9f27/Si/Si_r.upf
FR_GH_SHA=cc91a6be43c3c89daf6c34410096e8d870f25152dbfb513bcf01afa5ab53eabf

get_table() {   # $1 archive stem, $2 target file name, $3 glob of the Si file inside the archive
  [ -s "$T/$2" ] && { echo "  $2 present"; return 0; }
  curl -fsSL --max-time 300 -o "$TMP/$1.tgz" "$BASE/$1.tgz" 2>/dev/null || { echo "  $1: download failed"; return 1; }
  mkdir -p "$TMP/$1" && tar xzf "$TMP/$1.tgz" -C "$TMP/$1" 2>/dev/null || { echo "  $1: not a tar archive"; return 1; }
  f=$(find "$TMP/$1" -type f -name "$3" | head -1)
  [ -n "$f" ] || { echo "  $1: no $3 inside"; return 1; }
  grep -q "<UPF" "$f" || { echo "  $1: $3 is not UPF"; return 1; }
  cp "$f" "$T/$2" && echo "  $2 <- $1.tgz"
}

get_table nc-sr-04_pbe_standard_upf    Si.pbe.upf    "Si.upf"
get_table nc-sr-04_pbesol_standard_upf Si.pbesol.upf "Si.upf"
get_table nc-sr-04_pw_standard_upf     Si.pw.upf     "Si.upf"
if ! get_table nc-fr-04_pbe_standard_upf Si_r.pbe.upf "Si*.upf"; then
  curl -fsSL --max-time 300 -o "$TMP/Si_r.upf" "$FR_GH" 2>/dev/null \
    && echo "$FR_GH_SHA  $TMP/Si_r.upf" | sha256sum -c --status && cp "$TMP/Si_r.upf" "$T/Si_r.pbe.upf" \
    && echo "  Si_r.pbe.upf <- GitHub (checksum ok)" || echo "  Si_r.pbe.upf: GitHub fallback failed"
fi
( cd "$T" && { echo "PseudoDojo v0.4 standard accuracy, norm-conserving ONCVPSP; fetched $(date -u +%F)"; \
  sha256sum *.upf 2>/dev/null; } > SOURCES.txt )
if [ -s "$T/Si.pbe.upf" ]; then
  echo "  PseudoDojo Si set ready in $T ($(ls "$T"/*.upf | wc -l) files)"; exit 0
fi
if [ -s "$T/Si_r.pbe.upf" ]; then
  # pw.x averages the j-components of a fully relativistic pseudopotential in a scalar run, so the FR file also
  # serves the scalar PBE runs; PBEsol and LDA then go through input_dft (results flagged)
  echo "  PseudoDojo: only the fully relativistic PBE file; it also serves scalar runs (flag pseudodojo_fr_as_scalar)"
  exit 0
fi
echo "  PseudoDojo Si set NOT available: the bundled pseudopotentials stay in use (results flagged 'pseudos:bundled')"
exit 1
