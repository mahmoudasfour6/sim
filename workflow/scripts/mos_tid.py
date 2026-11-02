"""MOS capacitor checks (closed forms) and total-ionizing-dose response for three oxide thicknesses."""
import argparse
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.radiation.tid import dose_response
from simws.tcad.mos import MOSCase, cv
from simws.validation import mos_checks

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--doses-krad", default="0,10,30,100,300,1000")
ap.add_argument("--field-MV-cm", type=float, default=1.0)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
checks = mos_checks.run_all()
doses = [float(x) for x in a.doses_krad.split(",")]
tid = {}
for tox in (10.0, 20.0, 40.0):
    tid[f"tox_{tox:g}nm"] = dose_response(MOSCase(t_ox_nm=tox), V_irr=a.field_MV_cm * tox * 0.1, doses_krad=doses)
    r = tid[f"tox_{tox:g}nm"]
    print(f"t_ox {tox:g} nm: E_ox {r['E_ox_MV_cm']:.2f} MV/cm, dV_mg at {doses[-1]:g} krad = {r['dV_mg'][-1]:+.3f} V, "
          f"dV_inv = {r['dV_inv'][-1]:+.3f} V")
curve = cv(MOSCase(), np.linspace(-3, 3, 121))
np.savetxt(out / "cv_ideal.csv", np.column_stack([curve["V"], curve["C_F_cm2"], curve["psi_s_V"]]), delimiter=",",
           header="V_gate,C_F_per_cm2,psi_s_V", comments="")
write_json(out / "summary.json", {"checks": checks, "tid": tid, "C_ox_F_cm2": curve["C_ox_F_cm2"]}, stamp(parameters=vars(a)))
print("MOS checks pass:", checks["pass"])
if not checks["pass"]:
    raise SystemExit("MOS checks FAILED")
