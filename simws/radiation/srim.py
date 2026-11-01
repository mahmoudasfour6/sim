"""Parse SRIM-2013 'Ion Stopping and Range Table' output (SR Module, run natively on Windows) and derive the
LET along an ion's path by integrating the continuous-slowing-down equation dE/dx = -S(E).

Supported stopping units (read from the table header): eV/Angstrom, keV/micron, MeV/mm, keV/(ug/cm2),
MeV/(mg/cm2), keV/(mg/cm2), eV/(1E15 atoms/cm2) needs the atom density in the header (always present).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_E = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}
_L = {"A": 1e-4, "um": 1.0, "mm": 1e3, "m": 1e6, "km": 1e9}


@dataclass
class StoppingTable:
    ion: str
    density_g_cm3: float
    atom_density_cm3: float
    energy_MeV: np.ndarray
    s_elec_MeV_cm2_mg: np.ndarray
    s_nucl_MeV_cm2_mg: np.ndarray
    proj_range_um: np.ndarray

    @property
    def s_total(self):
        return self.s_elec_MeV_cm2_mg + self.s_nucl_MeV_cm2_mg

    def let(self, energy_MeV, electronic_only=True):
        s = self.s_elec_MeV_cm2_mg if electronic_only else self.s_total
        return np.interp(np.log(energy_MeV), np.log(self.energy_MeV), s)

    def let_vs_depth(self, E0_MeV, depth_um, electronic_only=True):
        """CSDA: march E(z) down from E0; returns LET (MeV cm2/mg) at each depth (0 once the ion stops).
        Electronic stopping is what creates e-h pairs; total stopping drives the slowing down."""
        z = np.asarray(depth_um, float)
        rho_mg_um = self.density_g_cm3 * 1e3 * 1e-4           # mg/cm2 per um
        fine = np.linspace(0, z.max(), max(2000, 20 * len(z)))
        E = np.empty_like(fine); E[0] = E0_MeV
        for i in range(1, len(fine)):
            if E[i - 1] <= self.energy_MeV[0]:
                E[i:] = 0.0
                break
            dE = self.let(E[i - 1], electronic_only=False) * rho_mg_um * (fine[i] - fine[i - 1])
            E[i] = max(E[i - 1] - dE, 0.0)
        out = np.where(E > self.energy_MeV[0], self.let(np.maximum(E, self.energy_MeV[0]), electronic_only), 0.0)
        return np.interp(z, fine, out)


def parse(text: str) -> StoppingTable:
    ion = re.search(r"Ion\s*=\s*([^\n,]+)", text).group(1).strip()
    rho = float(re.search(r"Target Density\s*=\s*([\d.Ee+-]+)\s*g/cm3", text).group(1))
    natm = float(re.search(r"=\s*([\d.Ee+-]+)\s*atoms/cm3", text).group(1))
    unit = re.search(r"Stopping Units\s*=\s*([^\n]+)", text).group(1).strip().replace(" ", "")
    to_mev_cm2_mg = {
        "eV/Angstrom": 1e-6 * 1e8 / (rho * 1e3), "keV/micron": 1e-3 * 1e4 / (rho * 1e3), "MeV/mm": 10.0 / (rho * 1e3),
        "keV/(ug/cm2)": 1.0, "MeV/(mg/cm2)": 1.0, "keV/(mg/cm2)": 1e-3,
    }
    if unit == "eV/(1E15atoms/cm2)":
        factor = 1e-6 * 1e-15 * natm / (rho * 1e3)
    elif unit in to_mev_cm2_mg:
        factor = to_mev_cm2_mg[unit]
    else:
        raise ValueError(f"unsupported SRIM stopping unit '{unit}'")
    row = re.compile(r"^\s*([\d.]+)\s+(eV|keV|MeV|GeV)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.]+)\s+(A|um|mm|m|km)\s", re.M)
    rows = row.findall(text)
    if not rows:
        raise ValueError("no stopping-table rows found")
    E = np.array([float(r[0]) * _E[r[1]] for r in rows])
    se = np.array([float(r[2]) for r in rows]) * factor
    sn = np.array([float(r[3]) for r in rows]) * factor
    rng = np.array([float(r[4]) * _L[r[5]] for r in rows])
    return StoppingTable(ion, rho, natm, E, se, sn, rng)


def load(path) -> StoppingTable:
    return parse(open(path, encoding="latin-1").read())
