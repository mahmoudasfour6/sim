"""Checks of simws.tcad.schottky_transport (0.9 closure). Tolerances fixed before the first run (2026-11-05):

te_limit           tunnelling and image force off: J vs A* T^2 exp(-phi/kT)(exp(V/kT) - 1); the only difference is
                   the Fermi-Dirac vs Boltzmann supply tail (gate 1e-6)
fd_step_exact      tunnelling off, image force on: J vs the exact Fermi-Dirac result for a step transmission at the
                   numerical barrier top, A* T^2 [Li2(-e^{-Umax/kT}) - Li2(-e^{(V-Umax)/kT})] (gate 1e-9)
zero_bias          J(V = 0) / J(0.1 V) with everything on, degenerate doping included (gate 1e-12)
wkb_triangle       WKB exponent of a linear barrier vs (2/3) sqrt(2m) (U0 - E)^1.5 / (hbar q F) (gate 1e-6)
image_lowering     numerical barrier lowering vs sqrt(q E_s / 4 pi eps) with the depletion field E_s (reported;
                   the closed form assumes a constant field near the interface)
degenerate_xi      E_c - E_F from Fermi-Dirac vs Boltzmann at 1e16 (gate 1e-6 eV) and 2e19 cm^-3 (reported)
tb_vs_wkb          exact 1D transmission (tight-binding Green's function on the same barrier, numpy; Kwant on the
                   laptop must agree with it to 1e-8) vs WKB: current ratio J_exact / J_WKB reported as the WKB
                   model-form error, in the thermionic-field regime of the Ni/4H-SiC contact
padovani_stratton  ratio to the Padovani-Stratton TFE formula (reported)
"""
from __future__ import annotations

import math

import numpy as np

from simws.tcad import schottky_transport as S
from simws.units import K_B_EV, Q_E


def te_limit():
    rows, worst = [], 0.0
    for T in (250.0, 300.0, 400.0):
        c = S.Contact(phi_B=1.2, N_D=1e16, T=T, image_force=False, tunneling=False)
        kT = c.kT
        for V in (0.2, 0.4, 0.6):
            J = S.current(c, V)
            ref = c.A_star * T * T * math.exp(-c.phi_B / kT) * math.expm1(V / kT)
            worst = max(worst, abs(J / ref - 1))
            rows.append({"T": T, "V": V, "rel": J / ref - 1})
    return {"rows": rows, "worst": worst, "pass": bool(worst < 1e-6)}


def fd_step_exact():
    """Step transmission at the numerical barrier top: the dilogarithm form against adaptive quadrature of the
    Fermi-Dirac supply integral (independent of the code's Li2)."""
    from scipy.integrate import quad
    worst = 0.0
    for N in (1e16, 2e19):
        c = S.Contact(phi_B=1.7, N_D=N, tunneling=False)
        for V in (0.1, 0.4):
            p = S.current(c, V, return_parts=True)
            kT, U = c.kT, p["Umax_eV"]
            f = lambda e: float(S._supply(np.array([U + e]), V, kT)[0])
            ref = c.A_star * c.T ** 2 / kT * quad(f, 0, 60 * kT, epsabs=0, epsrel=1e-13, limit=400)[0]
            worst = max(worst, abs(p["J"] / ref - 1))
    return {"worst": worst, "pass": bool(worst < 1e-9)}


def zero_bias():
    worst = 0.0
    for N in (1e16, 1e18, 2e19):
        c = S.Contact(phi_B=1.7, N_D=N)
        worst = max(worst, abs(S.current(c, 0.0)) / S.current(c, 0.1))
    return {"worst": worst, "pass": bool(worst < 1e-12)}


def wkb_triangle():
    m = 0.42 * S.M0
    U0, F = 1.0, 1e6                                   # eV, V/cm
    x = np.linspace(0, 2 * U0 / F, 20001)
    U = U0 - F * x
    worst = 0.0
    for E in (0.1, 0.4, 0.8):
        th = S._wkb_theta(x, U, E, m)
        ref = (2.0 / 3.0) * math.sqrt(2 * m * Q_E) * (U0 - E) ** 1.5 / (S.HBAR * F * 1e2)
        worst = max(worst, abs(th / ref - 1))
    return {"worst": worst, "pass": bool(worst < 1e-6)}


def image_lowering():
    rows = []
    for N in (1e16, 1e18, 2e19):
        c = S.Contact(phi_B=1.7, N_D=N)
        p = S.current(c, 0.1, return_parts=True)
        eps = c.eps_r * S.EPS0
        Vbb = c.phi_B - p["E_cb_eV"]
        Es = math.sqrt(2 * Q_E * N * Vbb / eps)
        d_cf = math.sqrt(Q_E * Es / (4 * math.pi * eps))
        rows.append({"N_D": N, "lowering_numeric_eV": p["barrier_lowering_eV"], "closed_form_eV": d_cf,
                     "rel": p["barrier_lowering_eV"] / d_cf - 1})
    return {"rows": rows}


def degenerate_xi():
    """Fermi-Dirac inversion: round trip N_c F_1/2(-xi/kT) = N_D (gate 1e-10), and at 1e16 cm^-3 the difference to
    Boltzmann equals its leading term -kT ln(1 - e^eta / 2^1.5) (gate 1 %). (First version compared FD with
    Boltzmann directly at 1e-6 eV; the true difference at 1e16 is 5.4e-6 eV, so that tolerance tested the wrong
    thing and was replaced before any use.)"""
    rows, rt = {}, 0.0
    for N in (1e16, 1e18, 2e19):
        c = S.Contact(phi_B=1.7, N_D=N)
        xi = c.xi()
        rt = max(rt, abs(c.N_c * S.fd_half(-xi / c.kT) / N - 1))
        xb = c.kT * math.log(c.N_c / N)
        rows[f"{N:.0e}"] = {"xi_FD_eV": xi, "xi_Boltzmann_eV": xb, "diff_eV": xi - xb}
    c = S.Contact(phi_B=1.7, N_D=1e16)
    eta = -c.xi() / c.kT
    lead = c.kT * math.log(1 - math.exp(eta) / 2 ** 1.5)
    lead_rel = abs(rows["1e+16"]["diff_eV"] / lead - 1)
    return {"rows": rows, "round_trip": rt, "leading_term_rel": lead_rel,
            "pass": bool(rt < 1e-10 and lead_rel < 0.01)}


def tb_transmission(x, U, E, m, a=1e-9, U_lead=None):
    """Exact 1D transmission of the barrier U(x) (x in cm, eV) on a tight-binding chain of spacing a (cm),
    both leads at U_lead; numpy Green's function (the same discretisation as the Kwant check)."""
    from scipy.linalg import solve_banded
    xs = np.arange(x[0], x[-1], a)
    Us = np.interp(xs, x, U)
    t = (S.HBAR ** 2 / (2 * m * (a * 1e-2) ** 2)) / Q_E          # eV
    UL = Us[-1] if U_lead is None else U_lead
    out = []
    for e in np.atleast_1d(E):
        cosk = 1 - (e - UL) / (2 * t)
        if abs(cosk) >= 1:
            out.append(0.0); continue
        k = math.acos(cosk)
        sigma = -t * np.exp(1j * k)
        n = len(xs)
        diag = (e - (Us + 2 * t)).astype(complex)
        diag[0] -= sigma; diag[-1] -= sigma
        ab = np.zeros((3, n), complex)
        ab[0, 1:] = t; ab[1] = diag; ab[2, :-1] = t
        rhs = np.zeros(n, complex); rhs[-1] = 1.0
        g = solve_banded((1, 1), ab, rhs)
        gam = -2 * sigma.imag
        out.append(float(gam * gam * abs(g[0]) ** 2))
    return np.array(out)


def tb_vs_wkb(N=1.97e19, phi=1.75, T=300.0, Vs=(0.1, 0.3), a=1e-9, kwant=False):
    """Current with exact transmission vs WKB, same barrier. The barrier is taken from where U first rises above
    E_cb (the image-force well at x -> 0 is cut, as in WKB, whose region starts at the turning point)."""
    c = S.Contact(phi_B=phi, N_D=N, T=T)
    rows = []
    for V in Vs:
        x, U, E_cb, W = S.band(c, V)
        start = np.argmax(U > E_cb)
        xs, Us = x[start:], U[start:]
        m = c.m_tunnel * S.M0
        if kwant:
            from simws.validation.kwant_barrier import transmission as kw
            Tf = lambda E: kw(xs, Us, E, m, a=a)
        else:
            Tf = lambda E: tb_transmission(xs, Us, E, m, a=a)
        J_ex = S.current(c, V, transmission=Tf, n_quad=400)
        J_w = S.current(c, V, n_quad=400)
        rows.append({"V": V, "J_exact": J_ex, "J_wkb": J_w, "ratio": J_ex / J_w,
                     "kT_over_E00": None})
    return {"N_D": N, "rows": rows, "ratio_range": [min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)]}


def padovani_stratton(N=1.97e19, phi=1.75, T=300.0):
    from simws.tcad.contacts import e00
    c = S.Contact(phi_B=phi, N_D=N, T=T, image_force=False)
    kT = c.kT
    E00 = e00(N, c.m_tunnel, c.eps_r)
    E0 = E00 / math.tanh(E00 / kT)
    xi = c.xi()
    rows = []
    for V in (0.1, 0.3):
        J = S.current(c, V)
        Js = c.A_star * T * math.sqrt(math.pi * E00 * (phi - V - xi)) / (K_B_EV * math.cosh(E00 / kT)) \
            * math.exp(-xi / kT - (phi - xi) / E0)
        rows.append({"V": V, "ratio_to_PS": J / (Js * math.exp(V / E0)), "kT_over_E00": kT / E00})
    return {"rows": rows}


def run_all(kwant=False):
    res = {"te_limit": te_limit(), "fd_step_exact": fd_step_exact(), "zero_bias": zero_bias(),
           "wkb_triangle": wkb_triangle(), "degenerate_xi": degenerate_xi(), "image_lowering": image_lowering(),
           "tb_vs_wkb": tb_vs_wkb(), "padovani_stratton": padovani_stratton()}
    if kwant:
        res["kwant_vs_tb"] = kwant_vs_tb()
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
    return res


def kwant_vs_tb(N=1.97e19, phi=1.75, V=0.2, a=1e-9):
    """Kwant (laptop, tcad env) vs the numpy Green's function on the identical chain (gate 1e-8)."""
    from simws.validation.kwant_barrier import transmission as kw
    c = S.Contact(phi_B=phi, N_D=N)
    x, U, E_cb, W = S.band(c, V)
    start = np.argmax(U > E_cb)
    E = np.linspace(E_cb + 0.01, U.max() - 0.01, 9)
    m = c.m_tunnel * S.M0
    t1 = tb_transmission(x[start:], U[start:], E, m, a=a)
    t2 = kw(x[start:], U[start:], E, m, a=a)
    rel = float(np.max(np.abs(t2 / t1 - 1)))
    return {"max_rel": rel, "pass": bool(rel < 1e-8)}
