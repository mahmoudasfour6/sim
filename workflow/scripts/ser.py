"""Critical LET, cross section and (with an environment spectrum) upset rate from the axisymmetric runs."""
import argparse, json, math
from pathlib import Path
from simws.provenance import stamp, write_json
from simws.radiation import ser

ap = argparse.ArgumentParser()
ap.add_argument("--axi", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--qcrit-fC", type=float, default=2.0)
ap.add_argument("--spectrum", default="")
ap.add_argument("--let-unit", default="MeV cm2/mg")
ap.add_argument("--flux-unit", default="1/cm2/s/sr")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
axi = json.loads(Path(a.axi).read_text())["results"]
runs = sorted(axi["all_lets"], key=lambda r: r["case"]["let"])
lets = [r["case"]["let"] for r in runs]
qc = [r["collected_charge_drain_fC"] for r in runs]
lc = ser.critical_let(lets, qc, a.qcrit_fC)
area = math.pi * (runs[0]["case"]["drain_radius_um"] * 1e-4) ** 2
res = {"q_crit_fC": a.qcrit_fC, "q_crit_note": "default = C_node x V_DD of the mixed-mode node (2 fF x 1 V); "
       "set seu_qcrit_fC in config/config.yaml for your circuit",
       "lets": lets, "collected_fC": qc, "critical_let": lc, "sensitive_area_cm2": area,
       "cross_section_model": "step at L_crit, saturated area = drain area (axial-strike approximation)",
       "engine_verification": ser.verify()}
if lc and a.spectrum and Path(a.spectrum).exists():
    spec = ser.LETSpectrum.from_file(a.spectrum, let_unit=a.let_unit, flux_unit=a.flux_unit)
    r = ser.rate_isotropic(spec, ser.StepCrossSection(lc, area))
    res.update(spectrum=spec.source, rate_per_bit_per_s=r, rate_per_bit_per_day=r * 86400)
else:
    res["rate_per_bit_per_day"] = None
    res["rate_note"] = "no environment spectrum configured (config: ser_spectrum, CREME96/OMERE output)"
write_json(out / "summary.json", res, stamp(parameters=vars(a)))
print(f"critical LET {lc} MeV cm2/mg for Q_crit {a.qcrit_fC} fC; area {area:.3e} cm2; "
      f"engine vs analytic {res['engine_verification']['rel_diff']:+.1e}; rate/day {res['rate_per_bit_per_day']}")
