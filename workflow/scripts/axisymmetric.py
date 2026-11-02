"""Axisymmetric heavy-ion strike on a cylindrical n+/p diode at realistic doping (1e20 / 1e17), fixed bias."""
import argparse
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--lets", default="1,10,40")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
runs = []
for let in map(float, a.lets.split(",")):
    r = seu_axisymmetric(AxiSEUCase(let=let))
    np.savetxt(out / f"axi_let{let:g}.csv", np.column_stack([r.pop("time_s"), r.pop("i_drain_A"), r.pop("i_sub_A")]),
               delimiter=",", header="time_s,i_drain_A,i_sub_A", comments="")
    runs.append(r)
    print(f"LET {let:6.3g}: deposited {r['deposited_charge_fC']:.1f} fC, collected drain {r['collected_charge_drain_fC']:.1f} fC, "
          f"peak {1e3 * r['peak_drain_current_A']:.3f} mA at {r['t_peak_current_s']:.2e} s, mesh factor {r['track_mesh_correction']:.4f}")
ref = next((r for r in runs if r["case"]["let"] == 10.0), runs[-1])
write_json(out / "summary.json", dict(ref, all_lets=runs), stamp(parameters=vars(a)))
