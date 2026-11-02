"""Semiconductor relations that need only the material set (no device simulator), usable from any environment."""
from __future__ import annotations

import math

from simws.units import K_B_EV


def intrinsic_density(mat, T):
    """n_i(T) from n_i(300 K), Varshni Eg(T) and Nc Nv ~ T^3. With densities of states in the material set
    (Nc_300, Nv_300 and their T exponents, e.g. the computed 'dft' profile) it is sqrt(Nc(T) Nv(T)) exp(-Eg/2kT)
    with the measured Varshni gap instead."""
    Eg0, a, b = mat.get("varshni_Eg0", "eV"), mat.get("varshni_alpha", "eV/K"), mat.get("varshni_beta", "K")
    Eg = lambda t: Eg0 - a * t * t / (t + b)
    if "Nc_300" in mat.parameters and "Nv_300" in mat.parameters:
        pc = mat.get("Nc_T_exponent", "dimensionless") if "Nc_T_exponent" in mat.parameters else 1.5
        pv = mat.get("Nv_T_exponent", "dimensionless") if "Nv_T_exponent" in mat.parameters else 1.5
        Nc = mat.get("Nc_300", "cm**-3") * (T / 300.0) ** pc
        Nv = mat.get("Nv_300", "cm**-3") * (T / 300.0) ** pv
        return math.sqrt(Nc * Nv) * math.exp(-Eg(T) / (2 * K_B_EV * T))
    ni = mat.get("n_i", "cm**-3")
    return ni * (T / 300.0) ** 1.5 * math.exp(Eg(300.0) / (2 * K_B_EV * 300.0) - Eg(T) / (2 * K_B_EV * T))
