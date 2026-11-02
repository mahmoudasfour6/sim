"""Calibrate material-database parameters against measured device data, with uncertainty.

    from simws.calibrate import Parameter, Calibration, DiodeSpec, diode_model, load_measurement
    data = load_measurement("measured/diode_iv.csv")            # columns: V, I  (and/or V, C) - see below
    spec = DiodeSpec(area_cm2=1e-4, p_plus=1e19, base_thickness_um=300)
    cal = Calibration(
        params=[Parameter("N_base", 1e15, 1e17, log=True),        # device parameter (spec field)
                Parameter("srh_tau_max_n", 1e-7, 1e-3, log=True)],# material-database parameter (silicon.yaml)
        model=lambda p: diode_model(spec, p, iv_V=data["iv"]["V"], cv_V=data["cv"]["V"]),
        data=data, rel_sigma={"iv": 0.05, "cv": 0.02})
    fit = cal.fit()                      # least squares in log-parameter space + Laplace (Gaussian) uncertainty
    post = cal.sample(n_steps=2000)      # optional Metropolis MCMC around the fit (full posterior)

Residuals are relative (log I for currents, so a 1e-12 A point weighs as much as a 1e-3 A point; C / C_meas - 1
for capacitance) and divided by the stated measurement uncertainty, so chi^2 / dof ~ 1 means the model explains
the data to within its error bars. The fit reports: best values, 1-sigma from the Laplace approximation, the
parameter correlation matrix (strong correlation = the data cannot separate those parameters), chi^2/dof.

Measurement CSV format: a header line, then rows. Columns named V and I (A) are an I-V curve, V and C (F) a C-V
curve; both may be in one file (blank cells allowed). Lines starting with # are metadata.

Names in `params` that are fields of DiodeSpec set the device; any other name must be a material parameter in
silicon.yaml (in its stored unit) and is applied with Material.with_overrides, so a calibrated value can be
written back to the database with its reference ("calibrated to <dataset>").
"""
from __future__ import annotations

import csv
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class Parameter:
    name: str
    lo: float
    hi: float
    log: bool = False
    x0: Optional[float] = None

    def to_u(self, v):
        return math.log10(v) if self.log else v

    def from_u(self, u):
        return 10 ** u if self.log else u

    @property
    def start(self):
        return self.x0 if self.x0 is not None else (math.sqrt(self.lo * self.hi) if self.log else 0.5 * (self.lo + self.hi))


def load_measurement(path) -> Dict[str, Dict[str, np.ndarray]]:
    rows = [r for r in csv.reader(l for l in open(path) if l.strip() and not l.startswith("#"))]
    head = [h.strip() for h in rows[0]]
    cols = {h: [] for h in head}
    for r in rows[1:]:
        for h, v in zip(head, r):
            cols[h].append(float(v) if v.strip() else np.nan)
    out = {}
    V = np.array(cols["V"])
    for key, col in (("iv", "I"), ("cv", "C")):
        if col in cols:
            y = np.array(cols[col]); ok = ~np.isnan(y)
            out[key] = {"V": V[ok], "y": y[ok]}
    return out


@dataclass
class FitResult:
    names: List[str]
    best: Dict[str, float]
    sigma_log10: Dict[str, Optional[float]]   # 1-sigma in log10 units for log parameters, else absolute
    interval_95: Dict[str, tuple]
    correlation: np.ndarray
    chi2_per_dof: float
    n_evals: int
    residuals: np.ndarray

    def table(self):
        lines = [f"{'parameter':22s} {'best':>12s} {'95% interval':>28s}"]
        for n in self.names:
            lo, hi = self.interval_95[n]
            lines.append(f"{n:22s} {self.best[n]:12.4g}   [{lo:11.4g}, {hi:11.4g}]")
        lines.append(f"chi2/dof = {self.chi2_per_dof:.3g}")
        if len(self.names) > 1:
            off = np.abs(self.correlation - np.eye(len(self.names))).max()
            lines.append(f"largest parameter correlation |r| = {off:.2f}" + ("  (data barely separate them)" if off > 0.95 else ""))
        return "\n".join(lines)


@dataclass
class Calibration:
    params: List[Parameter]
    model: Callable[[Dict[str, float]], Dict[str, np.ndarray]]   # returns {"iv": I(V), "cv": C(V)} as arrays
    data: Dict[str, Dict[str, np.ndarray]]
    rel_sigma: Dict[str, float] = field(default_factory=lambda: {"iv": 0.05, "cv": 0.02})
    n_evals: int = 0

    def _values(self, u):
        return {p.name: p.from_u(ui) for p, ui in zip(self.params, u)}

    def residuals(self, u):
        self.n_evals += 1
        pred = self.model(self._values(u))
        res = []
        for key, d in self.data.items():
            y, m = np.asarray(d["y"], float), np.asarray(pred[key], float)
            if key == "iv":
                r = (np.log(np.abs(m) + 1e-300) - np.log(np.abs(y) + 1e-300)) / self.rel_sigma[key]
            else:
                r = (m / y - 1) / self.rel_sigma[key]
            res.append(r)
        return np.concatenate(res)

    def fit(self, **ls_kwargs) -> FitResult:
        from scipy.optimize import least_squares
        u0 = np.array([p.to_u(p.start) for p in self.params])
        lb = np.array([p.to_u(p.lo) for p in self.params]); ub = np.array([p.to_u(p.hi) for p in self.params])
        opts = dict(x_scale="jac", diff_step=1e-3, xtol=1e-10, ftol=1e-10)
        opts.update(ls_kwargs)
        sol = least_squares(self.residuals, u0, bounds=(lb, ub), **opts)
        J, r = sol.jac, sol.fun
        dof = max(1, r.size - len(self.params))
        chi2 = float(r @ r) / dof
        # Laplace approximation in u-space; residuals are already normalised by the stated errors, so the
        # covariance is (J^T J)^-1, inflated by chi2/dof when the model does not reach the error bars
        cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(sd, sd) if np.all(sd > 0) else np.full_like(cov, np.nan)
        names = [p.name for p in self.params]
        best = self._values(sol.x)
        iv95 = {p.name: (p.from_u(x - 1.96 * s), p.from_u(x + 1.96 * s)) for p, x, s in zip(self.params, sol.x, sd)}
        self._u_best, self._cov = sol.x, cov
        return FitResult(names, best, dict(zip(names, sd)), iv95, corr, chi2, self.n_evals, r)

    def sample(self, n_steps=2000, seed=0, scale=1.0):
        """Random-walk Metropolis in u-space started at the fit, proposal = fit covariance (scaled 2.38^2/d).
        Flat prior inside the bounds. Returns dict name -> samples (after 20 % burn-in) and the acceptance rate.
        Each step is one model evaluation, so this is for cheap models or overnight runs."""
        if not hasattr(self, "_u_best"):
            self.fit()
        rng = np.random.default_rng(seed)
        d = len(self.params)
        L = np.linalg.cholesky(self._cov * (2.38 ** 2 / d) * scale + 1e-14 * np.eye(d))
        lb = np.array([p.to_u(p.lo) for p in self.params]); ub = np.array([p.to_u(p.hi) for p in self.params])
        u = self._u_best.copy()
        logp = -0.5 * float(np.sum(self.residuals(u) ** 2))
        chain, acc = [], 0
        for _ in range(n_steps):
            v = u + L @ rng.standard_normal(d)
            if np.all(v >= lb) and np.all(v <= ub):
                lp = -0.5 * float(np.sum(self.residuals(v) ** 2))
                if math.log(rng.random()) < lp - logp:
                    u, logp = v, lp
                    acc += 1
            chain.append(u.copy())
        chain = np.array(chain[int(0.2 * n_steps):])
        return {"samples": {p.name: np.array([p.from_u(x) for x in chain[:, i]]) for i, p in enumerate(self.params)},
                "acceptance": acc / n_steps}


# ----------------------------------------------------------------------------------------------- diode model
@dataclass
class DiodeSpec:
    """1D abrupt p+/n (or n+/p with polarity='n+p') diode for I-V and C-V calibration."""
    area_cm2: float = 1e-4
    p_plus: float = 1e19              # heavily doped side (cm^-3)
    N_base: float = 1e16              # lightly doped side (cm^-3)
    junction_um: float = 1.0          # depth of the junction below the heavy-side contact
    base_thickness_um: float = 50.0   # junction to back contact
    polarity: str = "p+n"
    T: float = 300.0
    series_ohm: float = 0.0           # lumped series resistance, applied after the device solve


def diode_model(spec: DiodeSpec, p: Dict[str, float], iv_V=(), cv_V=(), material=None):
    """Run DEVSIM on the diode with parameters `p` (DiodeSpec fields and/or material parameters). Returns
    {"iv": currents (A) at iv_V, "cv": capacitance (F) at cv_V}. Positive V = forward bias."""
    from devsim import set_parameter
    from simws.units import Q_E
    from simws.materials import load
    from simws.tcad.physics import Physics, contact_current, setup_region
    from simws.tcad.solve import ramp_bias
    from simws.validation.physics_checks import _mesh

    mat = material or load("silicon")
    spec = dataclasses.replace(spec, **{k: v for k, v in p.items() if k in {f.name for f in dataclasses.fields(DiodeSpec)}})
    over = {k: (v, mat.parameters[k].unit) for k, v in p.items() if k in mat.parameters}
    if over:
        mat = mat.with_overrides(**over)
    xj, L = spec.junction_um * 1e-4, (spec.junction_um + spec.base_thickness_um) * 1e-4
    _mesh(L, [(0, xj / 50, "L"), (xj, xj / 200, None), (L, L / 100, "R")])
    heavy, base = (-spec.p_plus, spec.N_base) if spec.polarity == "p+n" else (spec.p_plus, -spec.N_base)
    net = f"ifelse(x < {xj}, {heavy}, {base})"
    tot = f"ifelse(x < {xj}, {abs(heavy)}, {abs(base)})"
    setup_region("d", "si", mat, net, tot, Physics(T=spec.T), {"left": False, "right": False})
    sign = 1.0 if spec.polarity == "p+n" else -1.0          # forward = left contact positive for p+n
    opts = dict(absolute_error=1e10, relative_error=1e-9, maximum_iterations=40)
    out = {}
    if len(iv_V):
        I = []
        for V in iv_V:
            ramp_bias("d", "left", sign * V, step=0.05, **opts)
            I.append(sign * contact_current("d", "left") * spec.area_cm2)
        I = np.array(I)
        if spec.series_ohm > 0:               # V_applied = V_junction + I R: re-interpolate on the junction curve
            Va = np.asarray(iv_V) + I * spec.series_ohm
            I = np.interp(iv_V, Va, I)
        out["iv"] = I
    if len(cv_V):
        # quasi-static C = |dQ/dV| with Q = depletion charge of the base, q sum (|N| - majority) * NodeVolume over
        # x > xj. Minority carriers spilling over from the heavy side (a few-nm Debye tail) belong to the other
        # plate and are excluded; an ohmic contact sits in neutral material, so the contact charge is not usable.
        from devsim import get_node_model_values
        g = lambda n: np.array(get_node_model_values(device="d", region="si", name=n))
        x = g("x"); base = x > xj

        def q_base():
            maj = g("Electrons") if spec.polarity == "p+n" else g("Holes")
            return Q_E * float(np.sum((np.abs(g("NetDoping")) - maj)[base] * g("NodeVolume")[base]))

        C, dV = [], 0.01
        for V in cv_V:
            q = []
            for v in (V - dV / 2, V + dV / 2):
                ramp_bias("d", "left", sign * v, step=0.05, **opts)
                q.append(q_base())
            C.append(abs(q[1] - q[0]) / dV * spec.area_cm2)
        out["cv"] = np.array(C)
    set_parameter(device="d", name="left_bias", value=0.0)
    return out


def write_back_note(fit: FitResult, dataset: str) -> str:
    """YAML lines to paste into silicon.yaml (a calibration profile) for the fitted material parameters."""
    lines = [f"  calibrated_{dataset}:"]
    for n in fit.names:
        lo, hi = fit.interval_95[n]
        lines.append(f"    {n}: {{value: {fit.best[n]:.4g}, unit: <as in parameters>, "
                     f"ref: \"calibrated to {dataset}, 95% [{lo:.3g}, {hi:.3g}], chi2/dof {fit.chi2_per_dof:.2g}\", status: verified}}")
    return "\n".join(lines)
