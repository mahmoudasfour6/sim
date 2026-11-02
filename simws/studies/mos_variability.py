"""Process variation of a MOS capacitor: oxide thickness, substrate doping, fixed charge and interface traps
-> flat-band voltage, inversion voltage (band bending 2 phi_F) and minimum quasi-static capacitance."""
from __future__ import annotations

from simws.surrogate import Variable

VARIABLES = [
    Variable("t_ox_nm", "normal", 10.0, 0.3),      # 3 % 1-sigma oxide thickness
    Variable("NA", "normal", 1e17, 1e16),          # 10 % 1-sigma substrate doping
    Variable("Q_f", "uniform", 0.0, 2e11),         # fixed interface charge, cm^-2
    Variable("D_it", "uniform", 0.0, 5e11),        # interface traps, cm^-2 eV^-1
]
OUTPUTS = ("V_fb", "V_inv", "C_min_over_C_ox")


def stage(p: dict) -> dict:
    import numpy as np
    from simws.materials import load
    from simws.tcad.mos import MOSCase, cv, flatband, gate_voltage_at_surface_potential
    from simws.validation.mos_checks import _bulk_psi
    case = MOSCase(t_ox_nm=p["t_ox_nm"], NA=p["NA"], Q_f=p["Q_f"], D_it=p["D_it"])
    psi_b = _bulk_psi(case, load("silicon"))[0]
    vfb = flatband(case)
    vinv = float(gate_voltage_at_surface_potential(case, [-2 * psi_b], V_lo=-3, V_hi=4, n=141)[0])
    r = cv(case, np.linspace(vfb, vinv, 25))
    return {"V_fb": vfb, "V_inv": vinv, "C_min_over_C_ox": float(r["C_F_cm2"].min() / r["C_ox_F_cm2"])}
