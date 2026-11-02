"""Calibration self-test: synthetic I-V + C-V from known parameters with 5 % / 2 % noise; the fit must recover
them inside its 95 % intervals. Run before trusting a fit to real measurements."""
import argparse
from pathlib import Path
import numpy as np
from simws.calibrate import Calibration, DiodeSpec, Parameter, diode_model
from simws.provenance import stamp, write_json

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
spec = DiodeSpec(area_cm2=1e-4, base_thickness_um=50)
truth = {"N_base": 2.0e16, "srh_tau_max_p": 4e-7}
ivV, cvV = np.linspace(0.2, 0.65, 10), np.array([-3.0, -2.0, -1.0, 0.0])
clean = diode_model(spec, truth, ivV, cvV)
rng = np.random.default_rng(1)
data = {"iv": {"V": ivV, "y": clean["iv"] * np.exp(0.05 * rng.standard_normal(ivV.size))},
        "cv": {"V": cvV, "y": clean["cv"] * (1 + 0.02 * rng.standard_normal(cvV.size))}}
cal = Calibration([Parameter("N_base", 1e15, 1e17, log=True), Parameter("srh_tau_max_p", 1e-8, 1e-4, log=True)],
                  lambda p: diode_model(spec, p, ivV, cvV), data, {"iv": 0.05, "cv": 0.02})
fit = cal.fit()
print(fit.table())
params = {k: {"true": v, "fit": fit.best[k], "ci95": fit.interval_95[k],
              "inside_95": bool(fit.interval_95[k][0] <= v <= fit.interval_95[k][1])} for k, v in truth.items()}
res = {"parameters": params, "chi2_per_dof": fit.chi2_per_dof, "n_evals": fit.n_evals,
       "correlation": fit.correlation.tolist(), "pass": all(p["inside_95"] for p in params.values())}
write_json(out / "selftest.json", res, stamp())
if not res["pass"]:
    raise SystemExit("calibration self-test FAILED")
