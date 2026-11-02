#!/usr/bin/env bash
# Off-laptop backup: git history to a private remote (GitHub / university GitLab), large results via DVC to
# OneDrive. Run once:   bash tools/setup_offsite_backup.sh git@github.com:<you>/sim_workspace.git
# (create an EMPTY private repository first; HTTPS URLs work too and will ask for a token).
# Afterwards the nightly job pushes to it automatically.
set -euo pipefail
cd ~/sim_workspace
URL=${1:?usage: setup_offsite_backup.sh <git remote URL of an empty private repository>}

echo "== git remote 'offsite' -> $URL"
if git remote | grep -qx offsite; then git remote set-url offsite "$URL"; else git remote add offsite "$URL"; fi
git push offsite --all
git push offsite --tags
echo "  pushed: $(git rev-parse --short HEAD) on $(git branch --show-current)"

echo "== size check (GitHub rejects files > 100 MB)"
git ls-files -z | xargs -0 du -b 2>/dev/null | awk '$1 > 50e6 {printf "  large tracked file: %s (%.0f MB)\n", $2, $1/1e6}' || true

WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
OD=$(ls -d /mnt/c/Users/$WINUSER/OneDrive* 2>/dev/null | head -1 || true)
if command -v dvc >/dev/null || [ -x ~/miniforge3/envs/workflow/bin/dvc ]; then
  DVC=$(command -v dvc || echo ~/miniforge3/envs/workflow/bin/dvc)
  if [ -d .dvc ] && [ -n "$OD" ]; then
    echo "== DVC remote 'onedrive' -> $OD/sim_workspace_dvc"
    mkdir -p "$OD/sim_workspace_dvc"
    $DVC remote add -f -d onedrive "$OD/sim_workspace_dvc"
    $DVC push -q && echo "  dvc push done"
    git add .dvc/config && git commit -q -m "DVC remote on OneDrive" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
    git push -q offsite
  else
    echo "  (no .dvc directory or OneDrive folder found: skipped the DVC remote)"
  fi
fi
echo "DONE. The nightly job now also pushes to 'offsite'."
