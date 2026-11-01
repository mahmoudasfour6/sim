"""(100) Si film subbands, Schrödinger-Poisson at imposed sheet density."""
import argparse
from pathlib import Path

from simws.provenance import stamp, write_json
from simws.quantum.si_film import solve

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True)
ap.add_argument("--thickness-nm", type=float, default=5.0); ap.add_argument("--ns-cm2", type=float, default=1e13)
a = ap.parse_args()
r = solve(t_si_nm=a.thickness_nm, n_s_cm2=a.ns_cm2)
data = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
write_json(Path(a.outdir) / "summary.json", data, stamp(parameters=vars(a)))
print(f"E1(D2) = {r['E_D2_meV'][0]:.1f} meV, E1'(D4) = {r['E_D4_meV'][0]:.1f} meV, E_F = {r['EF_meV']:.1f} meV")
