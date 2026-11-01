"""Quantum ESPRESSO Si SCF in a results directory; writes summary.json with provenance."""
import argparse, re, shutil, subprocess
from pathlib import Path

from simws.provenance import stamp, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--pseudo-dir", required=True)
ap.add_argument("--outdir", required=True); ap.add_argument("--ranks", type=int, default=1)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
deck = Path(a.input).read_text()
deck = re.sub(r"pseudo_dir\s*=\s*'[^']*'", f"pseudo_dir = '{Path(a.pseudo_dir).resolve()}'", deck)
(out / "si_scf.pwi").write_text(deck)
pw = [shutil.which("pw.x"), "-in", "si_scf.pwi"]
# WSL reports hyperthreads as cores in a way Open MPI counts as fewer "slots" than logical CPUs, so a rank
# count that runs fine is refused without --oversubscribe. MPICH has no such flag: fall back to plain -np.
attempts = ([["mpirun", "--oversubscribe", "-np", str(a.ranks)] + pw, ["mpirun", "-np", str(a.ranks)] + pw]
            if a.ranks > 1 else [pw])
for cmd in attempts:
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
    if r.returncode == 0:
        break
    print(f"failed ({r.returncode}): {' '.join(cmd)}\n--- stderr ---\n{r.stderr[-3000:]}\n--- stdout tail ---\n{r.stdout[-2000:]}")
else:
    raise SystemExit("pw.x failed with every launcher; see the output above")
log = r.stdout
(out / "si_scf.pwo").write_text(log)
energy = float(re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)\s+Ry", log, re.M)[-1])
n_iter = int(re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", log)[-1])
write_json(out / "summary.json", {"total_energy_Ry": energy, "scf_iterations": n_iter},
           stamp(inputs=[a.input], parameters={"ranks": a.ranks, "launcher": " ".join(cmd[:-3])}))
print(f"E_tot = {energy} Ry in {n_iter} iterations")
