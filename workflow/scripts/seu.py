"""Mixed-mode checks: RC transient against the analytic solution, then a single-event upset of a reverse-biased
n+/p drain held by a resistive load (ion track from the LET given, planar 2D)."""
import argparse, json
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.tcad import mixed_mode as mm

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--lets", default="0.02,0.1,0.5,1,5")  # MeV cm2/mg
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
rc = mm.rc_check()
print(f"RC check: worst deviation {rc['worst_abs_dev_frac_of_vinf']:.2e} of V_inf")
runs = []
for let in map(float, a.lets.split(",")):
    r = mm.seu_diode(mm.SEUCase(let=let))
    np.savetxt(out / f"seu_let{let:g}.csv", np.column_stack([r.pop("time_s"), r.pop("v_node_V"), r.pop("i_sub_A")]),
               delimiter=",", header="time_s,v_node_V,i_sub_A_per_um_depth", comments="")
    runs.append(r)
    print(f"LET {let:6.3g}: deposited {r['deposited_charge_fC']:.3f} fC, collected {r['collected_charge_substrate_fC']:.3f} fC, "
          f"V_node {r['v_node_initial_V']:.3f} -> min {r['v_node_min_V']:.3f} V at {r['t_vmin_s']:.2e} s")
# critical LET: node dips below V_DD/2 (log-interpolated between the bracketing LETs)
lets, vmin = np.array([r["case"]["let"] for r in runs]), np.array([r["v_node_min_V"] for r in runs])
vdd = runs[0]["case"]["vdd"]
crit = None
for i in range(len(lets) - 1):
    if vmin[i] >= vdd / 2 > vmin[i + 1]:
        f = (vmin[i] - vdd / 2) / (vmin[i] - vmin[i + 1])
        crit = float(np.exp(np.log(lets[i]) + f * (np.log(lets[i + 1]) - np.log(lets[i]))))
print(f"critical LET (node < V_DD/2): {crit if crit is None else f'{crit:.3g} MeV cm2/mg'}")
write_json(out / "summary.json", {"rc_check": rc, "seu": runs, "critical_let_MeV_cm2_mg": crit},
           stamp(parameters=vars(a)))
