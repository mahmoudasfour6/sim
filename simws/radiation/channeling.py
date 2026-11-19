"""Channeling in (100) silicon from published analytic fits (contract A15).

    from simws.radiation import channeling as ch
    m = ch.model("B")                          # or "As"
    x = np.linspace(0, 1.0, 2001)              # depth, um
    prof = m.profile(x, energy_keV=35, dose_cm2=1e14)   # dict: density (1/um per ion), channeled_fraction, parts

Boron: A. F. Tasch et al., J. Electrochem. Soc. 136, 810 (1989): two Pearson densities, D1/DT from their Fig. 5.
Arsenic: K. Suzuki et al., IEEE Trans. Electron Devices 47, 44 (2000): joined half Gaussian plus a channeling tail,
parameters from their Fig. 4. The numbers and how they were read are in simws/data/channeling.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "channeling"
ANGSTROM_UM = 1e-4
NM_UM = 1e-3

# ------------------------------------------------------------------------------------------------ Pearson densities


def pearson_coefficients(dRp: float, gamma: float, beta: float) -> Tuple[float, float, float, float]:
    """(a, b0, b1, b2) of d ln f / dx = (x - a) / (b0 + b1 x + b2 x^2), x measured from the mean (A15.3)."""
    A = 10.0 * beta - 12.0 * gamma ** 2 - 18.0
    if A == 0:
        raise ValueError("pearson: 10 beta - 12 gamma^2 - 18 = 0 (no Pearson density)")
    a = -dRp * gamma * (beta + 3.0) / A
    b0 = -dRp ** 2 * (4.0 * beta - 3.0 * gamma ** 2) / A
    b2 = -(2.0 * beta - 3.0 * gamma ** 2 - 6.0) / A
    return a, b0, a, b2


def _support(dRp: float, gamma: float, beta: float, span: float) -> Tuple[float, float]:
    """The interval around the mode where the denominator keeps its sign, cut to +-span standard deviations."""
    a, b0, b1, b2 = pearson_coefficients(dRp, gamma, beta)
    lo, hi = -span * dRp, span * dRp
    if abs(b2) < 1e-14:                      # type III: linear denominator, one root
        if abs(b1) > 1e-14:
            r = -b0 / b1
            if r > a:
                hi = min(hi, r)
            else:
                lo = max(lo, r)
        return lo, hi
    disc = b1 * b1 - 4.0 * b2 * b0
    if disc > 0:
        r = np.sort(np.roots([b2, b1, b0]).real)
        below = [v for v in r if v < a]
        above = [v for v in r if v > a]
        if below:
            lo = max(lo, max(below))
        if above:
            hi = min(hi, min(above))
    return lo, hi


def pearson_shape(dRp: float, gamma: float, beta: float, n: int = 60001, span: float = 3000.0):
    """(u, f): the untruncated Pearson density on a grid of u = x - Rp, normalised to 1 on its support. The grid is
    sinh-spaced (fine near the mode, coarse far out) over +-span standard deviations, so heavy power-law tails
    (Pearson VII with beta near 7) keep their fourth moment."""
    if not dRp > 0:
        raise ValueError("pearson: dRp must be > 0")
    a, b0, b1, b2 = pearson_coefficients(dRp, gamma, beta)
    lo, hi = _support(dRp, gamma, beta, span)
    eps = 1e-9 * dRp
    c = 0.05 * dRp
    t = np.linspace(np.arcsinh((lo + eps - a) / c), np.arcsinh((hi - eps - a) / c), n)
    u = a + c * np.sinh(t)
    g = (u - a) / (b0 + b1 * u + b2 * u * u)          # d ln f / du
    # integrate from the mode outward (trapezoid on the grid), so ln f = 0 at the mode
    lnf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(u))])
    lnf -= lnf[np.argmin(np.abs(u - a))]
    f = np.exp(np.clip(lnf, -745, 0))
    f /= np.trapezoid(f, u)
    return u, f


def pearson_moments(dRp: float, gamma: float, beta: float) -> Dict[str, float]:
    """Numerical mean (relative to Rp), standard deviation, skewness and kurtosis of the untruncated density (R80)."""
    u, f = pearson_shape(dRp, gamma, beta)
    m1 = np.trapezoid(u * f, u)
    m2 = np.trapezoid((u - m1) ** 2 * f, u)
    m3 = np.trapezoid((u - m1) ** 3 * f, u)
    m4 = np.trapezoid((u - m1) ** 4 * f, u)
    return {"mean_offset": float(m1), "std": float(math.sqrt(m2)), "gamma": float(m3 / m2 ** 1.5),
            "beta": float(m4 / m2 ** 2)}


def pearson_on(x_um: np.ndarray, Rp: float, dRp: float, gamma: float, beta: float) -> np.ndarray:
    """The Pearson density (1/um) on depths x_um >= 0, truncated to x >= 0 and renormalised there."""
    u, f = pearson_shape(dRp, gamma, beta)
    xs = Rp + u
    keep = xs >= 0
    if keep.sum() < 2:
        raise ValueError("pearson: the whole density lies in front of the surface")
    norm = np.trapezoid(f[keep], xs[keep])
    return np.interp(x_um, xs, f, left=0.0, right=0.0) / norm


# ------------------------------------------------------------------------------------------------ boron (Tasch)

@lru_cache(maxsize=None)
def _tasch():
    return json.loads((DATA / "tasch1989_boron.json").read_text())


TASCH_ENERGIES = (15.0, 25.0, 35.0, 50.0)
TASCH_DOSES = (2e13, 2e15)


def tasch_parameters(E: float) -> Dict[str, float]:
    """Table I linearly in energy (A15.2); lengths in um."""
    t = _tasch()["table_I"]
    e = np.array(t["energy_keV"], float)
    get = lambda k: float(np.interp(E, e, np.array(t[k], float)))  # noqa: E731
    return {"R1": get("R1_A") * ANGSTROM_UM, "dRp1": get("dRp1_A") * ANGSTROM_UM, "g1": get("S1"), "b1": get("K1"),
            "R2": get("R2_A") * ANGSTROM_UM, "dRp2": get("dRp2_A") * ANGSTROM_UM, "g2": get("S2"), "b2": get("K2")}


def tasch_main_fraction(E: float, dose: float, shift: float = 0.0) -> float:
    """D1/DT: linear in log10(dose) along each Fig. 5 curve, linear in energy between curves (A15.2)."""
    f = _tasch()["fig5_D1_over_DT"]
    ld = np.log10(np.array(f["dose_cm2"], float))
    vals = []
    for ek in TASCH_ENERGIES:
        c = f["curves"][str(int(ek))]
        pts = [(x, y) for x, y in zip(ld, c) if y is not None]
        xs, ys = zip(*pts)
        vals.append(float(np.interp(math.log10(dose), xs, ys)) + shift)
    return float(np.clip(np.interp(E, TASCH_ENERGIES, vals), 0.0, 1.0))


# ------------------------------------------------------------------------------------------------ arsenic (Suzuki)

@lru_cache(maxsize=None)
def _suzuki():
    return json.loads((DATA / "suzuki2000_arsenic.json").read_text())


SUZUKI_ENERGIES = (40.0, 200.0)
SUZUKI_CHAN_ENERGIES = (40.0, 120.0, 200.0)
SUZUKI_DOSES = (1e11, 1e16)


def suzuki_parameters(E: float, dlen_nm: float = 0.0, dalpha: float = 0.0) -> Dict[str, float]:
    """Fig. 4(a) linearly in energy; lengths in um. dlen_nm / dalpha move the read values (uncertainty, A15.5):
    positive dlen widens (front and back straggle and L longer)."""
    a = _suzuki()["fig4a"]
    e = np.array(a["energy_keV"], float)
    get = lambda k: float(np.interp(E, e, np.array(a[k], float)))  # noqa: E731
    return {"Rpm": get("Rpm_nm") * NM_UM, "dRp1": (get("dRp1_nm") + dlen_nm) * NM_UM,
            "dRp2": (get("dRp2_nm") + dlen_nm) * NM_UM, "L": (get("L_nm") + dlen_nm) * NM_UM,
            "alpha": get("alpha") + dalpha}


def suzuki_channel_fraction(E: float, dose: float, scale: float = 1.0) -> float:
    """Phi_chan / Phi: linear in log10(dose) along the 40, 120, 200 keV curves (A15.2: the ratio itself, not the
    channel dose on log-log axes), linear in energy between them."""
    b = _suzuki()["fig4b_channel_dose"]
    doses = np.array(b["dose_cm2"], float)
    ld = np.log10(doses)
    vals = []
    for ek in SUZUKI_CHAN_ENERGIES:
        ratio = np.array(b["curves"][str(int(ek))], float) / doses
        vals.append(float(np.interp(math.log10(dose), ld, ratio)))
    return float(np.clip(np.interp(E, SUZUKI_CHAN_ENERGIES, vals) * scale, 0.0, 1.0))


def jhg(x_um: np.ndarray, Rpm: float, dRp1: float, dRp2: float) -> np.ndarray:
    """Joined half Gaussian with peak height 1 (Gibbons and Mylroie 1973; Suzuki eq. 5 with its squares)."""
    s = np.where(x_um < Rpm, dRp1, dRp2)
    return np.exp(-((x_um - Rpm) ** 2) / (2.0 * s * s))


def suzuki_channel_shape(x_um: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """n_c before normalisation (peak 1 at Rpm): JHG up to Rpm + dRp2, kappa (JHG + tail) beyond (eqs. 4, 6, 7)."""
    Rpm, d2, L, al = p["Rpm"], p["dRp2"], p["L"], p["alpha"]
    xj = Rpm + d2
    tail = lambda x: np.exp(-np.power(np.clip((x - Rpm) / L, 0.0, None), al))  # noqa: E731
    g_j = float(jhg(np.array([xj]), Rpm, p["dRp1"], d2)[0])
    kappa = g_j / (g_j + float(tail(np.array([xj]))[0]))
    base = jhg(x_um, Rpm, p["dRp1"], d2)
    return np.where(x_um < xj, base, kappa * (base + tail(x_um)))


def _normalise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y / np.trapezoid(y, x)


# ------------------------------------------------------------------------------------------------ the two models

class _Model:
    ion = ""
    cite = ""
    conditions = ""
    tilt_deg = 7.0
    rotation_deg = 0.0
    energies = (0.0, 0.0)
    doses = (0.0, 0.0)


class Tasch(_Model):
    ion = "B"
    rotation_deg = 30.0
    energies = (TASCH_ENERGIES[0], TASCH_ENERGIES[-1])
    doses = TASCH_DOSES

    @property
    def cite(self):
        return _tasch()["source"]["cite"]

    @property
    def conditions(self):
        return _tasch()["source"]["conditions"]

    def components(self, x_um, energy_keV):
        p = tasch_parameters(energy_keV)
        return (pearson_on(x_um, p["R1"], p["dRp1"], p["g1"], p["b1"]),
                pearson_on(x_um, p["R2"], p["dRp2"], p["g2"], p["b2"]))

    def profile(self, x_um, energy_keV, dose_cm2, perturb: int = 0) -> Dict[str, object]:
        """perturb -1 / +1: Fig. 5 values moved by -/+ 0.002 (less / more in the main peak)."""
        f1, f2 = self.components(x_um, energy_keV)
        main = tasch_main_fraction(energy_keV, dose_cm2, 0.002 * perturb)
        dens = main * f1 + (1.0 - main) * f2
        return {"density": dens, "channeled_fraction": 1.0 - main, "main": f1}


class Suzuki(_Model):
    ion = "As"
    rotation_deg = 22.0
    energies = SUZUKI_ENERGIES
    doses = SUZUKI_DOSES

    @property
    def cite(self):
        return _suzuki()["source"]["cite"]

    @property
    def conditions(self):
        return _suzuki()["source"]["conditions"]

    def components(self, x_um, energy_keV, dlen_nm=0.0, dalpha=0.0):
        """n_a and n_c on x_um, each normalised on [0, inf) (eq. 8), not on the points asked for."""
        p = suzuki_parameters(energy_keV, dlen_nm, dalpha)
        xx = np.asarray(x_um, float)
        far = p["Rpm"] + 60.0 * max(p["L"], p["dRp2"])              # exp(-60^alpha) is far below 1e-20
        xn = np.concatenate([np.linspace(0.0, p["Rpm"] + 8 * p["dRp2"], 40001)[:-1],
                             np.linspace(p["Rpm"] + 8 * p["dRp2"], far, 40001)])
        za = np.trapezoid(jhg(xn, p["Rpm"], p["dRp1"], p["dRp2"]), xn)
        zc = np.trapezoid(suzuki_channel_shape(xn, p), xn)
        return jhg(xx, p["Rpm"], p["dRp1"], p["dRp2"]) / za, suzuki_channel_shape(xx, p) / zc

    def profile(self, x_um, energy_keV, dose_cm2, perturb: int = 0) -> Dict[str, object]:
        """perturb -1 / +1: lengths -/+ 1 nm, alpha +/- 0.01 (a lower alpha is a longer tail), channel dose -/+ 5 %."""
        na, nc = self.components(x_um, energy_keV, 1.0 * perturb, -0.01 * perturb)
        fc = suzuki_channel_fraction(energy_keV, dose_cm2, 1.0 + 0.05 * perturb)
        dens = (1.0 - fc) * na + fc * nc
        return {"density": dens, "channeled_fraction": fc, "main": na}


def model(ion_symbol: str) -> Optional[_Model]:
    return {"B": Tasch(), "As": Suzuki()}.get(ion_symbol)
