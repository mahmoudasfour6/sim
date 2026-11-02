"""Single-event upset rate from simulated critical charge and an environment LET spectrum.

    from simws.radiation import ser
    lc = ser.critical_let(lets=[0.1, 0.3, 1, 10, 40], q_collected_fC=[...], q_crit_fC=2.0)   # from axisymmetric runs
    xs = ser.StepCrossSection(l_crit=lc, area_cm2=math.pi * (1e-4) ** 2)                     # sensitive area
    spec = ser.LETSpectrum.from_file("creme96_geo.txt", let_unit="MeV cm2/g", flux_unit="1/m2/s/sr")
    print(ser.rate_isotropic(spec, xs) * 86400, "upsets per bit per day")

Method (the standard integral-RPP / effective-LET approach for a thin, flat sensitive volume):
    rate = integral over LET of f(L) * integral over all directions of sigma(L / |cos t|) |cos t| dOmega
         = 4 pi * integral f(L) * integral_mu_min^1 mu sigma(L / mu) dmu dL
f(L) is the differential omnidirectional flux per steradian, sigma(L) the normal-incidence cross section.
A particle at angle t carries the effective LET L/cos t through the thin volume and sees the projected area
sigma cos t. mu_min > 0 caps the path length (the cosine law over-counts grazing ions in a volume of finite
lateral size): mu_min = depth / longest chord of the sensitive volume.

Cross sections: StepCrossSection (from simulation: no upset below L_crit, area A above it) or
WeibullCrossSection (fitted to heavy-ion test data). Spectra are never built in: an upset rate is only as good
as its environment, so supply CREME96 (or OMERE, SPENVIS) output for your orbit and shielding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LET_TO_MG = {"MeV cm2/mg": 1.0, "MeV cm2/g": 1e-3, "MeV/(mg/cm2)": 1.0, "MeV/(g/cm2)": 1e-3}
_AREA_TO_CM2 = {"cm2": 1.0, "m2": 1e4}


@dataclass
class LETSpectrum:
    """Integral omnidirectional flux F(>L) per steradian on a LET grid. Units after construction:
    L in MeV cm2/mg, F in 1/(cm2 s sr)."""
    let: np.ndarray
    integral_flux: np.ndarray
    source: str = ""

    @classmethod
    def from_file(cls, path, let_col=0, flux_col=1, let_unit="MeV cm2/mg", flux_unit="1/cm2/s/sr",
                  integral=True, source=None):
        """Read a whitespace or comma separated table (non-numeric lines skipped). Defaults suit a plain CSV in
        MeV cm2/mg and 1/(cm2 s sr). For CREME96 LET output use let_unit="MeV cm2/g" and the flux unit printed in
        its header (commonly per m2 s sr); check the header before trusting the result. flux_unit may be
        '1/cm2/s/sr', '1/m2/s/sr', '1/cm2/day/sr', '1/cm2/s' (omnidirectional total: divided by 4 pi), etc."""
        rows = []
        for line in open(path, encoding="latin-1"):
            parts = line.replace(",", " ").split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) > max(let_col, flux_col):
                rows.append((vals[let_col], vals[flux_col]))
        a = np.array(rows, float)
        L = a[:, 0] * _LET_TO_MG[let_unit]
        F = a[:, 1] * cls._flux_factor(flux_unit, let_unit if not integral else None)
        if not integral:                                      # differential given: integrate to F(>L)
            o = np.argsort(L); L, f = L[o], F[o]
            seg = 0.5 * (f[1:] + f[:-1]) * np.diff(L)
            F = np.append(np.cumsum(seg[::-1])[::-1], 0.0)
        o = np.argsort(L)
        return cls(L[o], F[o], source or str(path))

    @staticmethod
    def _flux_factor(unit, diff_let_unit=None):
        u = unit.replace(" ", "")
        f = 1.0
        if "/m2" in u:
            f /= 1e4
        if "/day" in u:
            f /= 86400.0
        if "/sr" not in u:
            f /= 4 * math.pi
        if diff_let_unit:                                     # per (MeV cm2/g) -> per (MeV cm2/mg)
            f /= _LET_TO_MG[diff_let_unit]
        return f

    @classmethod
    def power_law(cls, k, gamma, l_min=0.01, l_max=110.0, n=400):
        """SYNTHETIC test spectrum F(>L) = k (L^(1-gamma) - l_max^(1-gamma)) / (gamma - 1): for verifying the
        integration only. It is not a radiation environment."""
        L = np.geomspace(l_min, l_max, n)
        F = k * (L ** (1 - gamma) - l_max ** (1 - gamma)) / (gamma - 1)
        return cls(L, F, f"synthetic power law k={k}, gamma={gamma} (test only, not an environment)")

    def differential(self, L):
        """f(L) = -dF/dL (1/(cm2 s sr (MeV cm2/mg))): derivative of F on its own grid, interpolated in log L.
        Done in linear F because F falls to 0 at the top of a spectrum, where log F has no derivative."""
        f_grid = -np.gradient(self.integral_flux, self.let)
        L = np.asarray(L, float)
        f = np.interp(np.log(L), np.log(self.let), f_grid)
        return np.where((L >= self.let[0]) & (L <= self.let[-1]), np.clip(f, 0, None), 0.0)


@dataclass
class StepCrossSection:
    l_crit: float          # MeV cm2/mg
    area_cm2: float

    def __call__(self, L):
        return np.where(np.asarray(L) >= self.l_crit, self.area_cm2, 0.0)


@dataclass
class WeibullCrossSection:
    """sigma(L) = sat (1 - exp(-((L - L0)/W)^s)) for L > L0 (heavy-ion test-data form)."""
    l0: float
    w: float
    s: float
    sat_cm2: float

    def __call__(self, L):
        x = np.clip((np.asarray(L, float) - self.l0) / self.w, 0, None)
        return self.sat_cm2 * (1 - np.exp(-x ** self.s))


def rate_isotropic(spec: LETSpectrum, xs, mu_min=0.0, n_let=1500, n_mu=400):
    """Upsets per second per sensitive volume in an isotropic environment (both hemispheres)."""
    L = np.geomspace(spec.let[0], spec.let[-1], n_let)
    f = spec.differential(L)
    # mu grid on a log scale near 0 as well: sigma(L/mu) switches on at mu = L/L_crit, which is small for low L
    mu = np.unique(np.concatenate([np.geomspace(max(mu_min, 1e-6), 1.0, n_mu), np.linspace(max(mu_min, 1e-6), 1.0, n_mu)]))
    inner = np.trapezoid(mu[None, :] * xs(L[:, None] / mu[None, :]), mu, axis=1)
    return float(4 * math.pi * np.trapezoid(f * inner, L))


def rate_normal_beam(spec_integral_at, xs_value):
    """Beam test bookkeeping: events/s = sigma * flux."""
    return xs_value * spec_integral_at


def critical_let(lets, q_collected_fC, q_crit_fC):
    """LET at which collected charge reaches Q_crit (log-log interpolation; None if not bracketed)."""
    L, Q = np.asarray(lets, float), np.asarray(q_collected_fC, float)
    o = np.argsort(L); L, Q = L[o], Q[o]
    for i in range(len(L) - 1):
        if Q[i] < q_crit_fC <= Q[i + 1]:
            f = (math.log(q_crit_fC) - math.log(Q[i])) / (math.log(Q[i + 1]) - math.log(Q[i]))
            return float(math.exp(math.log(L[i]) + f * (math.log(L[i + 1]) - math.log(L[i]))))
    return None


def verify(k=1e-4, gamma=2.2, l_crit=3.0, area=1e-8):
    """Numerical engine vs the closed form for a step cross section and a power-law spectrum:
    rate = 4 pi A k [ integral_{lmin}^{Lc} L^-g (L/Lc)^2 / 2 dL + integral_{Lc}^{lmax} L^-g / 2 dL ]."""
    spec = LETSpectrum.power_law(k, gamma)
    lmin, lmax = spec.let[0], spec.let[-1]
    g = gamma
    a1 = (l_crit ** (3 - g) - lmin ** (3 - g)) / ((3 - g) * 2 * l_crit ** 2)
    a2 = (lmax ** (1 - g) - l_crit ** (1 - g)) / ((1 - g) * 2)
    exact = 4 * math.pi * area * k * (a1 + a2)
    num = rate_isotropic(spec, StepCrossSection(l_crit, area))
    return {"numerical": num, "analytic": exact, "rel_diff": num / exact - 1}
