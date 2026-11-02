"""Total ionizing dose (TID) in MOS oxides: oxide-trapped and interface-trap buildup -> device response.

    from simws.radiation.tid import dose_response
    r = dose_response(MOSCase(t_ox_nm=20, NA=1e17), V_irr=+2.0, doses_krad=[0, 10, 30, 100, 300, 1000])
    r["dV_mg"], r["dV_fb"], r["N_ot"], r["N_it"]

Model (Co-60, steady irradiation, no annealing or dose-rate effects):
  holes escaping recombination per rad and cm^2:  g0 * Y(E_ox) * t_ox
      g0 = 8.1e12 pairs / (cm^3 rad(SiO2)),  Y(E) = (E1/|E| + 1)^-m   (Dozier et al. 1987)
  oxide-trapped charge (sheet near the interface the holes drift to):
      N_ot(D) = N_sat (1 - exp(-f_ot g0 Y t_ox D / N_sat))       linear f_ot g0 Y t_ox D at low dose
  interface traps (reduced Esqueda / hydrogen model, positive bias):
      N_it(D) = N_SiH (1 - exp(-0.5 sigma_it N_DH sigma_DH g0 Y t_ox^2 D))
      spread uniformly over the gap with the neutral level at midgap: D_it = N_it / E_g
The holes drift with the field: for a positive gate (field toward the silicon) the charge sits `centroid_nm`
from the Si interface; for a negative gate it sits that far from the gate and barely shifts V_fb.

The device response is computed, not estimated: each dose re-solves the MOS capacitor (simws.tcad.mos) with
these charges, giving V_fb, the midgap voltage V_mg and the voltage at strong-inversion band bending V_inv.
The midgap shift contains no interface-trap charge (neutral at midgap), so dV_mg separates N_ot from N_it,
the standard midgap-subthreshold separation used with measured data.
Parameters other than g0 and Y are technology dependent: calibrate f_ot, N_sat and the hydrogen-model set to
your oxide (simws.calibrate) before quoting absolute shifts.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simws.materials import load


def charge_yield(E_MV_cm, ox=None):
    ox = ox or load("sio2")
    E1, m = ox.get("tid_yield_E1", "MV/cm"), ox.get("tid_yield_m", "dimensionless")
    E = np.maximum(np.abs(np.asarray(E_MV_cm, float)), 1e-6)
    return (E1 / E + 1) ** (-m)


def buildup(dose_rad, t_ox_nm, E_ox_MV_cm, ox=None):
    """(N_ot, N_it) in cm^-2 after `dose_rad` rad(SiO2)."""
    ox = ox or load("sio2")
    D = np.asarray(dose_rad, float)
    t = t_ox_nm * 1e-7
    g0, Y = ox.get("tid_g0", "cm**-3/rad"), float(charge_yield(E_ox_MV_cm, ox))
    f, Ns = ox.get("tid_f_ot", "dimensionless"), ox.get("tid_N_ot_sat", "cm**-2")
    N_ot = Ns * (1 - np.exp(-f * g0 * Y * t * D / Ns))
    k = 0.5 * ox.get("tid_sigma_it", "cm**2") * ox.get("tid_N_DH", "cm**-3") * ox.get("tid_sigma_DH", "cm**2") * g0 * Y * t * t
    N_it = ox.get("tid_N_SiH", "cm**-2") * (1 - np.exp(-k * D)) if E_ox_MV_cm > 0 else np.zeros_like(D)
    return N_ot, N_it


def dose_response(case, V_irr, doses_krad, centroid_nm=1.0, material=None):
    from simws.tcad.mos import flatband, gate_voltage_at_surface_potential, oxide_field_MV_cm
    from simws.validation.mos_checks import _bulk_psi
    mat = material or load("silicon")
    E = oxide_field_MV_cm(case, V_irr, mat)
    Eg = mat.get("Eg", "eV")
    N_ot, N_it = buildup(np.asarray(doses_krad, float) * 1e3, case.t_ox_nm, E)
    d = centroid_nm if E > 0 else case.t_ox_nm - centroid_nm
    psi_b = _bulk_psi(case, mat)[0]
    out = {"V_irr": V_irr, "E_ox_MV_cm": E, "yield": float(charge_yield(E)), "doses_krad": list(doses_krad),
           "N_ot": N_ot.tolist(), "N_it": N_it.tolist(), "centroid_from_si_nm": d,
           "V_fb": [], "V_mg": [], "V_inv": []}
    for not_, nit in zip(N_ot, N_it):
        c = dataclasses.replace(case, oxide=list(case.oxide) + [(d, float(not_))], D_it=case.D_it + float(nit) / Eg)
        out["V_fb"].append(flatband(c, mat))
        vm, vi = gate_voltage_at_surface_potential(c, [-psi_b, -2 * psi_b], V_lo=-6, V_hi=6, n=241, material=mat)
        out["V_mg"].append(float(vm)); out["V_inv"].append(float(vi))
    for k in ("V_fb", "V_mg", "V_inv"):
        out["d" + k] = (np.array(out[k]) - out[k][0]).tolist()
    return out
