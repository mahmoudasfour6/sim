#!/usr/bin/env bash
# Move workspace dirs into the repo, leave symlinks at old home paths.
# Dry run by default. Execute with: DRY_RUN=0 bash tools/relink.sh
set -euo pipefail
cd ~/sim_workspace
DRY_RUN=${DRY_RUN:-1}
PAIRS="01_materials_dft_md:materials_workspace 02_process_radiation:process_rad_workspace 04_fem_multiphysics:fem_workspace 05_quantum_iii_v:quantum_iii_v_workspace 06_photonics_spl:spl_study"
run(){ if [ "$DRY_RUN" = 1 ]; then echo "  [dry] $*"; else echo "  [run] $*"; "$@"; fi; }

if [ "$DRY_RUN" = 0 ]; then
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "Uncommitted changes; commit first."; exit 1; }
  bk=~/sim_backup_$(date +%Y%m%d_%H%M).tar.gz
  tar -czf "$bk" -C ~ $(for p in $PAIRS; do echo "${p#*:}"; done)
  echo "Backup written: $bk"
fi

for p in $PAIRS; do
  L=${p%%:*}; T=$HOME/${p#*:}
  if [ ! -L "$L" ] || [ "$(readlink -f "$L")" != "$T" ]; then echo "SKIP $L (not a symlink to $T)"; continue; fi
  echo "== $L  <- $T  ($(du -sh "$T" | cut -f1))"
  find "$T" -type f -size +5M -exec du -h {} + | sed 's/^/  LARGE: /' || true
  find "$T" -maxdepth 3 -name .git | sed 's/^/  NESTED GIT: /' || true
  run git rm --cached -q "$L"
  run unlink "$L"
  run mv "$T" "$L"
  run ln -s "$PWD/$L" "$T"
done

if [ "$DRY_RUN" = 0 ]; then
  git add -A
  echo; git status --short | head -40
  echo; git diff --cached --shortstat
  echo "Review above, then: git commit -m 'Bring workspace dirs into repo'"
fi
