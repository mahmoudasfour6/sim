#!/usr/bin/env python3
"""Compare the DEVSIM replica (devsim_particle_strike.csv) against Charon's gold currents (gold_currents.csv)."""
import json
import numpy as np

g = np.genfromtxt("gold_currents.csv", delimiter=",", names=True)
d = np.genfromtxt("devsim_particle_strike.csv", delimiter=",", names=True)
tg, Ig = g["time_s"], g["cathode_A_per_cm"]
td, Id = d["time_s"], -d["cathode_A_per_cm"]          # DEVSIM's sign convention is opposite to Charon's
pulse = (tg > 1e-4) & (tg < 1e-3)
dev = np.interp(tg, td, Id) - Ig
res = {
    "dc_A_per_cm": {"charon": float(Ig[1]), "devsim": float(Id[1]), "rel_diff": float(Id[1] / Ig[1] - 1)},
    "peak_A_per_cm": {"charon": float(Ig.max()), "devsim": float(Id.max()), "rel_diff": float(Id.max() / Ig.max() - 1)},
    "peak_time_s": {"charon": float(tg[Ig.argmax()]), "devsim": float(td[Id.argmax()])},
    "extra_charge_C_per_cm": {"charon": float(np.trapezoid(Ig - Ig[1], tg)), "devsim": float(np.trapezoid(Id - Id[1], td))},
    "max_abs_dev_in_pulse_frac_of_peak": float(np.max(np.abs(dev[pulse])) / Ig.max()),
}
res["extra_charge_C_per_cm"]["rel_diff"] = res["extra_charge_C_per_cm"]["devsim"] / res["extra_charge_C_per_cm"]["charon"] - 1
print(json.dumps(res, indent=2))
json.dump(res, open("comparison.json", "w"), indent=2)
print("\n t [s]        Charon [A/cm]   DEVSIM [A/cm]   diff")
for i in np.where(pulse)[0][::2]:
    print(f"{tg[i]:.4e}   {Ig[i]:12.5f}   {np.interp(tg[i], td, Id):12.5f}   {100 * dev[i] / Ig.max():+6.2f}% of peak")
