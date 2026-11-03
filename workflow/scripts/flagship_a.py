"""Flagship study A (SiGe HBT temperature window), screening fidelity, through the hub."""
import argparse

from simws.hub.jobs import Hub
from simws.studies.flagship_a import run

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--store", default=None, help="hub store (default: the workspace store)")
ap.add_argument("--workers", type=int, default=1, help="jobs submitted in parallel")
a = ap.parse_args()
res = run(a.outdir, hub=Hub(a.store), workers=a.workers)
print(open(f"{a.outdir}/flagship_a_table.txt").read())
f = res["fits_trusted_window"]
if f:
    print(f"trusted-window fits: graded dE = {1e3 * f['graded_form']['dE_grade_eV']:.1f} meV "
          f"(rms log resid {f['graded_form']['rms_log_resid']:.3f}); exp dE = {1e3 * f['exp_form']['dE_eV']:.1f} meV "
          f"(rms {f['exp_form']['rms_log_resid']:.3f})")
w = res.get("fits_fit_window") or {}
if w.get("edge_plus_graded"):
    e = w["edge_plus_graded"]
    print(f"fit window {res['fit_window_T_K']} K: graded-only rms {w['graded_form']['rms_log_resid']:.3f}, exp-only rms "
          f"{w['exp_form']['rms_log_resid']:.3f}; H1b edge + graded: remaining grade "
          f"{1e3 * e['dE_grade_remaining_eV']:.1f} meV, A {e['A']:.3f}, rms {e['rms_log_resid']:.3f}")
