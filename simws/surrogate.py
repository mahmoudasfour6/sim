"""Polynomial-chaos surrogates for process variation and uncertainty: a few hundred simulations stand in for
millions of Monte Carlo samples.

    from simws.surrogate import Variable, PCE
    vars_ = [Variable("t_ox_nm", "normal", 10.0, 0.3), Variable("NA", "normal", 1e17, 1e16),
             Variable("Q_f", "uniform", 0.0, 2e11)]                 # normal: mean, std; uniform: low, high
    design = PCE.design(vars_, n=120, seed=0)                      # Latin hypercube in the inputs' own distributions
    ... run the simulations (simws.sweep.run_design), collect y ...
    pce = PCE.fit(vars_, design, y, max_degree=4)                  # degree chosen by leave-one-out error
    pce.loo_q2, pce.sobol(), pce.sample(1_000_000)                 # accuracy, sensitivities, output distribution

Orthonormal bases: Legendre for uniform inputs, probabilists' Hermite for normal inputs, total-degree truncation.
Coefficients by least squares; leave-one-out residuals in closed form (e_i / (1 - h_ii)), reported as
Q^2 = 1 - PRESS / SS_tot: a surrogate with Q^2 < 0.99 should not be trusted for tails. Sobol indices follow
analytically from the coefficients (Sudret 2008). Inputs are independent; correlated process parameters need a
transform first.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Variable:
    name: str
    dist: str          # "normal" (a = mean, b = std) or "uniform" (a = low, b = high)
    a: float
    b: float

    def to_std(self, x):
        x = np.asarray(x, float)
        return (x - self.a) / self.b if self.dist == "normal" else 2 * (x - self.a) / (self.b - self.a) - 1

    def from_unit(self, u):
        """Inverse CDF from u in (0, 1)."""
        u = np.asarray(u, float)
        if self.dist == "normal":
            from scipy.stats import norm
            return self.a + self.b * norm.ppf(u)
        return self.a + (self.b - self.a) * u


def _legendre(x, n):
    P = [np.ones_like(x), x]
    for k in range(1, n):
        P.append(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1))
    return [P[k] * math.sqrt(2 * k + 1) for k in range(n + 1)]            # orthonormal for U(-1, 1)


def _hermite(x, n):
    H = [np.ones_like(x), x]
    for k in range(1, n):
        H.append(x * H[k] - k * H[k - 1])
    return [H[k] / math.sqrt(math.factorial(k)) for k in range(n + 1)]     # orthonormal for N(0, 1)


def _multi_indices(d, p):
    return [a for k in range(p + 1) for a in itertools.product(range(k + 1), repeat=d) if sum(a) == k]


class PCE:
    def __init__(self, variables: List[Variable], degree, alphas, coef, loo_q2, n_train):
        self.variables, self.degree, self.alphas, self.coef = variables, degree, alphas, coef
        self.loo_q2, self.n_train = loo_q2, n_train

    @staticmethod
    def design(variables: List[Variable], n, seed=0):
        from scipy.stats import qmc
        U = qmc.LatinHypercube(d=len(variables), seed=seed).random(n)
        return [{v.name: float(v.from_unit(U[i, j])) for j, v in enumerate(variables)} for i in range(n)]

    @staticmethod
    def _basis(variables, X, alphas, p):
        Z = np.column_stack([v.to_std(X[:, j]) for j, v in enumerate(variables)])
        polys = [(_hermite if v.dist == "normal" else _legendre)(Z[:, j], p) for j, v in enumerate(variables)]
        return np.column_stack([np.prod([polys[j][a[j]] for j in range(len(variables))], axis=0) for a in alphas])

    @classmethod
    def fit(cls, variables, design, y, max_degree=5, degree=None):
        X = np.array([[row[v.name] for v in variables] for row in design], float)
        y = np.asarray(y, float)
        best = None
        for p in ([degree] if degree else range(1, max_degree + 1)):
            alphas = _multi_indices(len(variables), p)
            if len(alphas) >= len(y):            # need more samples than terms
                break
            Psi = cls._basis(variables, X, alphas, p)
            c, *_ = np.linalg.lstsq(Psi, y, rcond=None)
            Q, _ = np.linalg.qr(Psi)
            h = np.sum(Q * Q, axis=1)                                  # leverage
            res = (y - Psi @ c) / np.clip(1 - h, 1e-12, None)
            q2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
            if best is None or q2 > best[3]:
                best = (p, alphas, c, q2)
        p, alphas, c, q2 = best
        return cls(variables, p, alphas, c, q2, len(y))

    def predict(self, rows):
        X = np.array([[r[v.name] for v in self.variables] for r in rows], float) if isinstance(rows[0], dict) else np.asarray(rows, float)
        return self._basis(self.variables, X, self.alphas, self.degree) @ self.coef

    @property
    def mean(self):
        return float(self.coef[0])

    @property
    def variance(self):
        return float((self.coef[1:] ** 2).sum())

    def sobol(self):
        V = self.variance
        out = {}
        for j, v in enumerate(self.variables):
            first = sum(c * c for a, c in zip(self.alphas, self.coef) if a[j] > 0 and sum(a) == a[j])
            total = sum(c * c for a, c in zip(self.alphas, self.coef) if a[j] > 0)
            out[v.name] = {"S1": first / V if V else 0.0, "ST": total / V if V else 0.0}
        return out

    def sample(self, n=200_000, seed=1):
        rng = np.random.default_rng(seed)
        U = rng.random((n, len(self.variables)))
        X = np.column_stack([v.from_unit(U[:, j]) for j, v in enumerate(self.variables)])
        return self.predict(X)

    def summary(self, n=200_000, seed=1, quantiles=(0.00135, 0.0228, 0.5, 0.9772, 0.99865)):
        s = self.sample(n, seed)
        return {"degree": self.degree, "terms": len(self.alphas), "n_train": self.n_train, "loo_q2": self.loo_q2,
                "mean": self.mean, "std": math.sqrt(self.variance),
                "quantiles": {f"{q:.5g}": float(np.quantile(s, q)) for q in quantiles}, "sobol": self.sobol()}
