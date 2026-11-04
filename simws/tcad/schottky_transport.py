"""Schottky-contact transport with tunnelling through the image-force-rounded barrier and Fermi-Dirac supply
(0.9 closure; replaces the TE x (1 + Gamma) x image-force-factor form for heavily doped contacts).

Energy zero: the metal Fermi level. Forward bias V puts the semiconductor Fermi level at qV. Uniform donors N_D,
depletion approximation, conduction band (eV):

    U(x) = E_cb + (q N_D / 2 eps) (W - x)^2  -  q / (16 pi eps_i x),      0 < x < W
    E_cb = qV + xi,  xi = E_c - E_F in the bulk from N_D = N_c F_1/2(-xi/kT)   (Fermi-Dirac, degenerate allowed)
    (q N_D / 2 eps) W^2 = phi_B - E_cb   (the band bends from phi_B at the interface to E_cb at x = W)

Current density (Crowell & Rideout 1969; Tsu-Esaki form), m* entering only through A* and the WKB mass:

    J = A* T / k  INT_{E_cb}^{inf} T(E) ln[(1 + exp((qV - E)/kT)) / (1 + exp(-E/kT))] dE

    T(E) = exp(-2 INT sqrt(2 m_t (U(x) - E)) / hbar dx)  over U > E   (WKB; 1 above the rounded barrier top)

The image force therefore lowers and rounds the barrier INSIDE the transmission, so it is counted once. The supply
function is Fermi-Dirac on both sides, so degenerate doping (N_D > N_c) is handled and the current vanishes
identically at V = 0. Options: image_force (default True), tunneling (default True; False = T = step at the top),
transmission ("wkb" or a callable T(E) for the Kwant / tight-binding comparison).

Not included (stated in every result): band-gap and barrier temperature dependence beyond the given phi_B(T); the
depletion approximation's kT tail at the edge; quantum reflection above the barrier top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from simws.units import K_B_EV, Q_E

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
EPS0 = 8.8541878128e-14            # F/cm


def fd_half(eta):
    """Fermi-Dirac integral F_1/2(eta), normalised (2/sqrt(pi)) INT sqrt(e)/(1+exp(e-eta)) de, by quadrature."""
    from scipy.integrate import quad
    f = lambda e: math.sqrt(e) / (1.0 + math.exp(min(e - eta, 700.0)))
    hi = max(eta, 0.0) + 60.0
    return 2.0 / math.sqrt(math.pi) * quad(f, 0.0, hi, limit=400, epsabs=0, epsrel=1e-13)[0]


from functools import lru_cache


@lru_cache(maxsize=4096)
def xi_from_doping(N_D, N_c, kT):
    """E_c - E_F (eV) in the bulk for fully ionised donors: N_D = N_c F_1/2(-xi/kT)."""
    from scipy.optimize import brentq
    g = lambda eta: math.log(fd_half(eta)) - math.log(N_D / N_c)
    eta = brentq(g, -60.0, 60.0, xtol=1e-14)
    return -eta * kT


@dataclass
class Contact:
    phi_B: float                 # eV, metal Fermi level to semiconductor conduction band at the interface (no image)
    N_D: float                   # cm^-3
    T: float = 300.0
    eps_r: float = 9.7           # static permittivity (depletion)
    eps_r_image: Optional[float] = None    # image-force permittivity (default: eps_r)
    m_tunnel: float = 0.42       # m*/m0 for tunnelling (4H-SiC: along c, about 0.33-0.42)
    A_star: float = 146.0        # A cm^-2 K^-2
    N_c300: float = 1.7e19       # cm^-3 at 300 K, scaled as T^1.5
    image_force: bool = True
    tunneling: bool = True

    @property
    def kT(self):
        return K_B_EV * self.T

    @property
    def N_c(self):
        return self.N_c300 * (self.T / 300.0) ** 1.5

    def xi(self):
        return xi_from_doping(self.N_D, self.N_c, self.kT)


def band(c: Contact, V):
    """(x grid cm, U(x) eV, E_cb, W) for bias V (V)."""
    eps = c.eps_r * EPS0
    eps_i = (c.eps_r_image or c.eps_r) * EPS0
    E_cb = V + c.xi()
    Vbb = c.phi_B - E_cb
    if Vbb <= 0:
        raise ValueError(f"flat band reached: phi_B - (qV + xi) = {Vbb:.4f} eV")
    W = math.sqrt(2 * eps * Vbb / (Q_E * c.N_D))
    # graded grid: fine near the interface (image force), coarse in the bulk of the depletion region
    x = W * np.concatenate([np.geomspace(1e-6, 1e-2, 400, endpoint=False), np.linspace(1e-2, 1.0, 1600)])
    if not c.image_force:
        x = np.concatenate([[0.0], x])                  # the barrier top is the interface itself
    U = E_cb + (Q_E * c.N_D / (2 * eps)) * (W - x) ** 2
    if c.image_force:
        U = U - Q_E / (16 * math.pi * eps_i * x)
    return x, U, E_cb, W


def _wkb_theta(x, U, E, m):
    """INT sqrt(2 m (U - E)) / hbar dx over U > E (x in cm, energies in eV), trapezoid on the region with exact
    turning points by linear interpolation."""
    k = np.sqrt(np.clip(U - E, 0.0, None) * Q_E * 2 * m) / HBAR * 1e-2      # 1/cm
    inside = U > E
    if not inside.any():
        return 0.0
    i0, i1 = np.argmax(inside), len(inside) - np.argmax(inside[::-1]) - 1
    xs = x[i0:i1 + 1].copy(); ks = k[i0:i1 + 1].copy()
    # extend to the turning points (k = 0 there); sqrt-type endpoints are integrated exactly on the last cell
    th = np.trapezoid(ks, xs)
    for j, side in ((i0, -1), (i1, +1)):
        jn = j + side
        if 0 <= jn < len(x):
            dU0, dU1 = U[j] - E, U[jn] - E          # dU0 > 0 >= dU1
            xt = x[j] + (x[jn] - x[j]) * dU0 / (dU0 - dU1)
            th += (2.0 / 3.0) * ks[0 if side < 0 else -1] * abs(xt - x[j])     # INT sqrt(linear) = 2/3 k L
    return float(th)


def transmission_wkb(c: Contact, V, E, band_xU=None):
    x, U = band_xU if band_xU is not None else band(c, V)[:2]
    Umax = U.max()
    E = np.atleast_1d(E)
    m = c.m_tunnel * M0
    return np.array([1.0 if e >= Umax else math.exp(-2 * _wkb_theta(x, U, e, m)) for e in E])


def _wkb_theta_many(x, U, E, m):
    return np.array([_wkb_theta(x, U, e, m) for e in E])


def li2_neg(y):
    """Li2(-y) for y >= 0, accurate for tiny y (series) and large y (inversion formula)."""
    if y < 0.5:
        k = np.arange(1, 80)
        return float(np.sum((-y) ** k / k ** 2))
    if y <= 2.0:
        from scipy.special import spence
        return float(spence(1.0 + y))
    # Li2(-y) = -pi^2/6 - ln(y)^2/2 - Li2(-1/y)
    return -math.pi ** 2 / 6 - 0.5 * math.log(y) ** 2 - li2_neg(1.0 / y)


def _supply(E, V, kT):
    """ln[(1 + exp((qV - E)/kT)) / (1 + exp(-E/kT))], evaluated without overflow."""
    a, b = (V - E) / kT, -E / kT
    return np.logaddexp(0.0, a) - np.logaddexp(0.0, b)


def current(c: Contact, V, transmission: Optional[Callable] = None, n_quad=600, return_parts=False):
    """J (A/cm^2) at bias V. transmission: None = WKB (or step if tunneling is off), else T(E) callable."""
    x, U, E_cb, W = band(c, V)
    kT = c.kT
    Umax = float(U.max())
    lo = E_cb
    # above the top: T = 1, integrate the supply to +inf (closed form with the dilogarithm)
    a = (V - Umax) / kT
    above = kT * (-li2_neg(math.exp(a)) + li2_neg(math.exp(-Umax / kT))) if a < 700 else float("nan")
    # Li2 form: INT_{Umax}^inf ln(1+exp((qV-E)/kT)) dE = -kT Li2(-exp((qV-Umax)/kT)); same for the metal side
    tunnel = 0.0
    if c.tunneling and Umax > lo:
        # Gauss-Legendre on [E_cb, Umax] in pieces (the integrand varies over several decades)
        edges = np.linspace(lo, Umax, 41)
        g, w = np.polynomial.legendre.leggauss(max(4, n_quad // 40))
        for e0, e1 in zip(edges[:-1], edges[1:]):
            E = 0.5 * (e1 - e0) * g + 0.5 * (e1 + e0)
            Tt = transmission(E) if transmission is not None else transmission_wkb(c, V, E, band_xU=(x, U))
            tunnel += 0.5 * (e1 - e0) * float(np.sum(w * Tt * _supply(E, V, kT)))
    J = c.A_star * c.T * c.T / kT * (above + tunnel)
    if return_parts:
        return {"J": J, "J_over_top": c.A_star * c.T ** 2 / kT * above, "J_tunnel": c.A_star * c.T ** 2 / kT * tunnel,
                "Umax_eV": Umax, "E_cb_eV": E_cb, "W_cm": W, "barrier_lowering_eV": c.phi_B - Umax}
    return J


def ideality(c: Contact, V, dV=1e-3):
    """n = (q/kT) / d ln J / dV, by central difference."""
    j1, j2 = current(c, V - dV), current(c, V + dV)
    return 1.0 / (math.log(j2 / j1) / (2 * dV) * c.kT)
