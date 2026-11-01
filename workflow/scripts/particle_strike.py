"""DEVSIM replica of Charon's particleStrike test + comparison against Charon's gold currents."""
import argparse
from pathlib import Path

import numpy as np

from simws.provenance import stamp, write_json
from simws.tcad import particle_strike as ps

ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--outdir", required=True)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
sim = ps.run()
np.savetxt(out / "devsim_currents.csv", np.column_stack([sim["time_s"], sim["cathode_A_per_cm"], sim["G_peak"]]),
           delimiter=",", header="time_s,cathode_A_per_cm,G_peak_cm3_s", comments="")
g = np.genfromtxt(a.gold, delimiter=",", names=True)
cmp = ps.compare(sim, g["time_s"], g["cathode_A_per_cm"])
write_json(out / "comparison.json", cmp, stamp(inputs=[a.gold], parameters={"n_i": sim["n_i"], "eps_r": sim["eps_r"],
           "material_profile": "silicon/charon"}))
print(f"max deviation over pulse: {100 * cmp['max_abs_dev_in_pulse_frac_of_peak']:.2f}% of peak")
