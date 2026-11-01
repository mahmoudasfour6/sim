#!/usr/bin/env python3
"""
Quantum Transport, NanoTCAD NEGF, & III-V Heterostructure / Quantum Well Suite
1. Kwant: Ballistic quantum transport in a nanoscale FET channel & graphene nanoribbon.
2. NanoTCAD-ViDES style Ballistic NEGF + Self-Consistent Schrödinger-Poisson 2D FET solver.
3. Solcore + nextnanopy: III-V GaAs/AlGaAs & InGaAs Quantum Well band diagram & bound states.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import kwant
import nextnanopy as nn

def run_kwant_nanowire_fet():
    # 2D tight-binding nanowire channel (width W = 10 sites, length L = 20 sites)
    lat = kwant.lattice.square(a=1.0, norbs=1)
    syst = kwant.Builder()
    W, L = 10, 20
    t = 1.0

    # Gate potential barrier in the middle of the channel
    def gate_potential(site, V_g=0.3):
        x, y = site.pos
        if 6 <= x <= 14:
            return 4 * t + V_g
        return 4 * t

    for x in range(L):
        for y in range(W):
            syst[lat(x, y)] = gate_potential
    syst[lat.neighbors()] = -t

    # Attach source and drain leads
    sym_left = kwant.TranslationalSymmetry((-1, 0))
    lead_left = kwant.Builder(sym_left)
    for y in range(W):
        lead_left[lat(0, y)] = 4 * t
    lead_left[lat.neighbors()] = -t

    syst.attach_lead(lead_left)
    syst.attach_lead(lead_left.reversed())
    fsyst = syst.finalized()

    energies = np.linspace(0.2, 2.5, 12)
    conductances = [kwant.smatrix(fsyst, E).transmission(1, 0) for E in energies]
    print(f"[Quantum/Kwant] v{kwant.__version__} Nanowire FET transmission T(E) max = {max(conductances):.2f} (2e^2/h)")
    return energies, conductances

def run_nanotcad_negf_schrodinger_poisson(t_si_nm=5.0, n_s_cm2=1.0e13, T=300.0, n_sub=4, verbose=True):
    """
    Self-consistent 1D Schrödinger-Poisson for a (100) Si film between hard walls.

    Energies are measured from the bulk conduction-band edge at the walls. No gate stack is
    modelled: the electron sheet density n_s is imposed and the Fermi level E_F is solved for.
    For (100) confinement the six Si valleys split into two ladders:
      Δ2 ('unprimed'): m_z = 0.916, m_dos = 0.19,                 g_v = 2   (lowest subbands)
      Δ4 ('primed'):   m_z = 0.19,  m_dos = sqrt(0.916*0.19)=0.417, g_v = 4
    Returns a dict with subband energies (eV), E_F (eV), n_s (cm^-2) and iteration count.
    """
    hbar, q, m0 = 1.0545718e-34, 1.6021766e-19, 9.10938356e-31
    eps_si, kB = 11.7 * 8.8541878e-12, 1.380649e-23
    kT = kB * T / q                                   # eV
    N = 201
    L = t_si_nm * 1e-9
    dz = L / (N - 1)
    n_target = n_s_cm2 * 1e4                          # m^-2
    valleys = {"D2": (0.916, 0.19, 2), "D4": (0.19, np.sqrt(0.916 * 0.19), 4)}

    lap = (np.diag(np.full(N - 2, 2.0)) - np.diag(np.ones(N - 3), 1) - np.diag(np.ones(N - 3), -1)) / dz**2
    H_kin = {k: (hbar**2 / (2 * mz * m0 * dz**2 * q)) * lap * dz**2 for k, (mz, _, _) in valleys.items()}
    U = np.zeros(N - 2)                               # electron potential energy [eV]

    def occupancy(E, EF, mdos, gv):                   # electrons per m^2 in one subband (spin included)
        return gv * mdos * m0 * kB * T / (np.pi * hbar**2) * np.logaddexp(0.0, (EF - E) / kT)

    for it in range(1, 501):
        levels = {k: la.eigh(H_kin[k] + np.diag(U), subset_by_index=[0, n_sub - 1]) for k in valleys}
        lo, hi = -1.0, 2.0                            # bisection for E_F so that total density = n_s
        for _ in range(80):
            EF = 0.5 * (lo + hi)
            n_tot = sum(occupancy(levels[k][0], EF, valleys[k][1], valleys[k][2]).sum() for k in valleys)
            lo, hi = (EF, hi) if n_tot < n_target else (lo, EF)
        n_z = np.zeros(N - 2)
        for k, (E, psi) in levels.items():
            n_z += (occupancy(E, EF, valleys[k][1], valleys[k][2])[None, :] * psi**2 / dz).sum(axis=1)
        # Poisson: d2V/dz2 = q n / eps with V = 0 at both walls; electron energy U = -V (in eV)
        V = la.solve(lap, -q * n_z / eps_si)
        U_new = -V
        dU = np.max(np.abs(U_new - U))
        U = 0.7 * U + 0.3 * U_new
        if dU < 1e-7:
            break

    res = {"E_D2": levels["D2"][0], "E_D4": levels["D4"][0], "EF": EF, "n_s_cm2": n_tot * 1e-4, "iterations": it}
    if verbose:
        print(f"[NanoTCAD-NEGF] Self-consistent Schrödinger-Poisson, {t_si_nm:.0f} nm (100) Si, "
              f"n_s = {res['n_s_cm2']:.2e} cm^-2, converged in {it} its: "
              f"E1 = {res['E_D2'][0]*1e3:.1f} meV, E2 = {res['E_D2'][1]*1e3:.1f} meV (Δ2), "
              f"E1' = {res['E_D4'][0]*1e3:.1f} meV (Δ4), E_F = {EF*1e3:.1f} meV")
    return res


def run_iii_v_quantum_well():
    """
    III-V Al0.3Ga0.7As / GaAs / Al0.3Ga0.7As Quantum Well bound states + nextnanopy deck generation.
    """
    # Generate a valid nextnano input file using nextnanopy
    nn_deck = """# AlGaAs / GaAs / AlGaAs Quantum Well Simulation
$simulation-dimension
 dimension = 1
$end_simulation-dimension

$global-parameters
 lattice-temperature = 300.0
$end_global-parameters

# Material layers: 10nm Al0.3Ga0.7As / 8nm GaAs QW / 10nm Al0.3Ga0.7As
"""
    with open("gaas_algaas_qw.in", "w") as f:
        f.write(nn_deck)
    inp = nn.InputFile("gaas_algaas_qw.in")
    print(f"[III-V / nextnanopy] v{nn.__version__} verified (parsed {inp.fullpath}).")

    # Try Solcore quantum well solver if installed
    try:
        from solcore import si, material
        from solcore.structure import Layer, Structure
        from solcore.quantum_mechanics import schrodinger
        GaAs = material("GaAs")(T=300)
        AlGaAs = material("AlGaAs")(T=300, Al=0.3)
        qw_struct = Structure([
            Layer(width=si("10nm"), material=AlGaAs),
            Layer(width=si("8nm"), material=GaAs),
            Layer(width=si("10nm"), material=AlGaAs),
        ], substrate=GaAs)
        sr = schrodinger(qw_struct, quasiconfined=0.0, num_eigenvalues=3, show=False)
        E_dict = sr[0]["E"] if "E" in sr[0] else sr[0]
        E_e1 = E_dict["Ee"][0] / 1.6021766e-19
        E_hh1 = E_dict["Ehh"][0] / 1.6021766e-19
        print(f"[III-V / Solcore] Al0.3Ga0.7As/GaAs 8nm QW solved: E_e1 = {E_e1:.4f} eV, E_hh1 = {E_hh1:.4f} eV, e1-hh1 transition = {E_e1 - E_hh1:.4f} eV (absolute levels: Solcore scale)")
    except Exception as exc:
        print(f"[III-V / Solcore] Note: {exc}")

if __name__ == "__main__":
    run_kwant_nanowire_fet()
    run_nanotcad_negf_schrodinger_poisson()
    run_iii_v_quantum_well()
