"""Self-consistent 1D Schrödinger-Poisson for a (100) Si film between hard walls (validated: sim_workspace
commit 2b9a497, golden values 114.6 / 145.6 / 180.1 / 135.4 meV at 5 nm, n_s = 1e13 cm^-2).

Energies are measured from the bulk conduction-band edge at the walls. The sheet density is imposed and E_F
solved for (no gate stack). (100) confinement splits the six valleys into
  D2: m_z = m_l, m_dos = m_t,            g_v = 2   (lowest ladder)
  D4: m_z = m_t, m_dos = sqrt(m_l m_t),  g_v = 4
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la

from simws.units import EPS0_F_PER_CM, HBAR, K_B_EV, M0_KG, Q_E


def solve(t_si_nm=5.0, n_s_cm2=1.0e13, T=300.0, n_sub=4, N=201, material=None, tol_eV=1e-7):
    from simws.materials import load
    mat = material or load("silicon")
    m_l, m_t = mat.get("m_l", "dimensionless"), mat.get("m_t", "dimensionless")
    eps = mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM * 100.0      # F/m
    kT = K_B_EV * T
    L = t_si_nm * 1e-9
    dz = L / (N - 1)
    n_target = n_s_cm2 * 1e4
    valleys = {"D2": (m_l, m_t, 2), "D4": (m_t, float(np.sqrt(m_l * m_t)), 4)}
    lap = (np.diag(np.full(N - 2, 2.0)) - np.diag(np.ones(N - 3), 1) - np.diag(np.ones(N - 3), -1)) / dz**2
    H_kin = {k: (HBAR**2 / (2 * mz * M0_KG * Q_E)) * lap for k, (mz, _, _) in valleys.items()}
    U = np.zeros(N - 2)

    def occ(E, EF, mdos, gv):
        return gv * mdos * M0_KG * K_B_EV * Q_E * T / (np.pi * HBAR**2) * np.logaddexp(0.0, (EF - E) / kT)

    for it in range(1, 501):
        levels = {k: la.eigh(H_kin[k] + np.diag(U), subset_by_index=[0, n_sub - 1]) for k in valleys}
        lo, hi = -1.0, 2.0
        for _ in range(80):
            EF = 0.5 * (lo + hi)
            n_tot = sum(occ(levels[k][0], EF, valleys[k][1], valleys[k][2]).sum() for k in valleys)
            lo, hi = (EF, hi) if n_tot < n_target else (lo, EF)
        n_z = np.zeros(N - 2)
        for k, (E, psi) in levels.items():
            n_z += (occ(E, EF, valleys[k][1], valleys[k][2])[None, :] * psi**2 / dz).sum(axis=1)
        U_new = la.solve(lap, Q_E * n_z / eps)        # electron potential energy [eV]: U = -V, d2V/dz2 = q n/eps
        dU = np.max(np.abs(U_new - U))
        U = 0.7 * U + 0.3 * U_new
        if dU < tol_eV:
            break
    return {"E_D2_meV": levels["D2"][0] * 1e3, "E_D4_meV": levels["D4"][0] * 1e3, "EF_meV": EF * 1e3,
            "n_s_cm2": n_tot * 1e-4, "iterations": it, "converged": bool(dU < tol_eV)}
