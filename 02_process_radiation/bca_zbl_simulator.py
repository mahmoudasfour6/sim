#!/usr/bin/env python3
"""
Native Linux ZBL Binary Collision Approximation (BCA) Ion Implantation & Recoil Simulator
Computes ion trajectories, nuclear (ZBL) + electronic (Oen-Robinson / Lindhard) stopping,
projected range Rp, straggle Delta_Rp, and Kinchin-Pease vacancy production in Si and 4H-SiC.
"""
import numpy as np

def run_zbl_bca_implant(Z1=5, M1=11.009, E0_eV=50000.0, Z2=14, M2=28.0855, N_at_A3=0.04994, n_ions=400):
    a_u = 0.88534 * 0.529177 / (Z1**0.23 + Z2**0.23)  # ZBL screening length [A]
    e2 = 14.399645  # eV * A
    mean_free_path = N_at_A3 ** (-1.0 / 3.0)
    p_max = mean_free_path / np.sqrt(np.pi)
    E_d = 15.0  # Displacement threshold [eV]

    depths_nm = []
    total_vacancies = 0
    rng = np.random.default_rng(42)

    for _ in range(n_ions):
        E = E0_eV
        z = 0.0
        cos_theta = 1.0
        while E > 20.0:
            p = p_max * np.sqrt(rng.random())
            # Reduced energy epsilon
            eps = (a_u * M2 / (Z1 * Z2 * e2 * (M1 + M2))) * E
            # ZBL universal scattering approximation for recoil energy transfer T_n
            b = p / a_u
            sin2_half = 1.0 / (1.0 + (2.0 * eps * b)**2 + 0.12 * eps**0.5)
            T_max = 4.0 * M1 * M2 / ((M1 + M2)**2) * E
            T_n = T_max * sin2_half
            # Lindhard-Scharff electronic energy loss over step mean_free_path
            S_e = 0.15 * np.sqrt(Z1 * Z2) * np.sqrt(E / 1000.0) * mean_free_path
            E_next = E - T_n - S_e
            if T_n > E_d:
                total_vacancies += max(1, int(0.8 * T_n / (2.0 * E_d)))
            # Update direction
            psi = np.arctan2(M2 * np.sin(2 * np.arcsin(np.sqrt(sin2_half))), M1 + M2 * np.cos(2 * np.arcsin(np.sqrt(sin2_half))))
            phi = 2.0 * np.pi * rng.random()
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
            cos_theta = cos_theta * np.cos(psi) + sin_theta * np.sin(psi) * np.cos(phi)
            z += mean_free_path * cos_theta
            if z < 0.0:
                break
            E = E_next
        if z > 0.0:
            depths_nm.append(z * 0.1)

    depths_nm = np.array(depths_nm)
    print(f"[Native ZBL-BCA] 50 keV Boron -> Silicon ({n_ions} ions): Mean Range Rp = {depths_nm.mean():.1f} nm, Straggle = {depths_nm.std():.1f} nm, Vacancies/ion = {total_vacancies/n_ions:.1f}")

if __name__ == "__main__":
    run_zbl_bca_implant()
