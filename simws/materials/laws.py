"""Material laws: parameters as functions of the environment (T now; strain and dose hooks), each with a validity
box. Outside the box a value is still returned but flagged 'extrapolated' (spec decision: flag, don't reject); a law
whose fitted range is not documented returns 'validity_unknown'.

    from simws.materials import laws
    v = laws.evaluate("silicon", "Eg", T=77.0)        # LawValue(value=1.165..., unit='eV', flags=[])
    v = laws.evaluate("silicon", "mu_n", T=77.0)      # flags=['extrapolated:T=77 outside [200,500]']
    laws.spec("silicon", "Eg")                        # MaterialLawSpec for the knowledge graph
    laws.sample_interface("4H-SiC/SiO2", "Dit_near_Ec", n=100, seed=1)   # literature spread as samples

Forms
    varshni          Eg0 - alpha T^2 / (T + beta)
    power_law_neg    x300 (T/300)^-a          (Si mobility, thermal conductivity: a > 0)
    power_law        x300 (T/300)^g           (Caughey-Thomas exponents stored with their sign, 4H-SiC)
    intrinsic        simws.materials.semiconductor.intrinsic_density (Varshni + DOS scaling)
    constant         the 300 K value (no environment dependence modelled)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

FORMS: Dict[str, Callable] = {
    "varshni": lambda T, Eg0, alpha, beta: Eg0 - alpha * T ** 2 / (T + beta),
    "power_law_neg": lambda T, x300, a: x300 * (T / 300.0) ** (-a),
    "power_law": lambda T, x300, g: x300 * (T / 300.0) ** g,
    "constant": lambda x300, **_: x300,
}


@dataclass
class LawValue:
    value: float
    unit: str
    flags: List[str] = field(default_factory=list)


@dataclass
class Law:
    material: str
    parameter: str
    form: str
    coeffs: Dict[str, Tuple[str, str]]               # formula argument -> (material key, unit)
    unit: str
    validity: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    validity_basis: str = ""
    source: str = ""

    def coefficients(self, mat) -> Dict[str, float]:
        return {arg: mat.get(key, u) for arg, (key, u) in self.coeffs.items()}

    def __call__(self, mat=None, **state) -> LawValue:
        from simws.materials import load
        mat = mat or load(self.material)
        flags = []
        if not self.validity:
            flags.append("validity_unknown")
        for var, (lo, hi) in self.validity.items():
            if var in state:
                a = np.atleast_1d(np.asarray(state[var], float))
                if a.min() < lo or a.max() > hi:
                    flags.append(f"extrapolated:{var}={a.min():g}..{a.max():g} outside [{lo:g},{hi:g}]")
        if any(mat.parameters[k].status not in ("verified", "computed") for k, _ in self.coeffs.values()):
            flags.append("provisional_parameters")
        if self.form == "intrinsic":
            from simws.materials.semiconductor import intrinsic_density
            T = np.asarray(state["T"], float)
            val = np.vectorize(lambda t: intrinsic_density(mat, float(t)))(T)
        else:
            args = self.coefficients(mat)
            if self.form != "constant":
                args["T"] = np.asarray(state["T"], float)
            val = FORMS[self.form](**args)
        val = float(val) if np.ndim(val) == 0 else np.asarray(val)
        return LawValue(val, self.unit, flags)

    def spec(self, mat=None):
        from simws.hub.schema import MaterialLawSpec, Quantity
        from simws.materials import load
        mat = mat or load(self.material)
        coeffs = {arg: Quantity(name=key, value=mat.get(key, u), unit=u, source=mat.parameters[key].ref)
                  for arg, (key, u) in self.coeffs.items()}
        stat = {mat.parameters[k].status for k, _ in self.coeffs.values()}
        status = "verified" if stat <= {"verified"} else ("computed" if stat <= {"verified", "computed"} else "provisional")
        return MaterialLawSpec(material=self.material, parameter=self.parameter,
                               form="constant" if self.form == "constant" else "expression",
                               expression=self.form, coefficients=coeffs, unit=self.unit,
                               validity={k: tuple(v) for k, v in self.validity.items()},
                               source=self.source + (f" | validity: {self.validity_basis}" if self.validity_basis else ""),
                               status=status)


_V = ("varshni_Eg0", "eV"), ("varshni_alpha", "eV/K"), ("varshni_beta", "K")

LAWS: Dict[Tuple[str, str], Law] = {}


def _add(law: Law):
    LAWS[(law.material, law.parameter)] = law


# --- silicon
_add(Law("silicon", "Eg", "varshni", dict(zip(("Eg0", "alpha", "beta"), _V)), "eV", {"T": (0.0, 1000.0)},
         "range planned in the build plan; confirm against the temperature range of Thurmond's data", "Thurmond (1975)"))
_add(Law("silicon", "n_i", "intrinsic", {**dict(zip(("Eg0", "alpha", "beta"), _V)), "ni300": ("n_i", "cm**-3")}, "cm**-3", {"T": (250.0, 400.0)},
         "n_i(300 K) with T^1.5 DOS scaling: checked against Altermatt (2003) near room temperature only",
         "Altermatt et al. (2003); Thurmond (1975)"))
for c in ("n", "p"):
    _add(Law("silicon", f"mu_{c}", "power_law_neg", {"x300": (f"mu_{c}", "cm**2/V/s"), "a": (f"mu_{c}_T_exponent", "dimensionless")},
             "cm**2/V/s", {"T": (200.0, 500.0)},
             "lattice-limited power law; below ~200 K the exponent changes (Jacoboni 1977), above ~500 K unverified here",
             "Masetti (1983); Palankovski (2000)"))
_add(Law("silicon", "thermal_k", "power_law_neg", {"x300": ("thermal_k", "W/cm/K"), "a": ("thermal_k_T_exponent", "dimensionless")},
         "W/cm/K", {"T": (300.0, 600.0)}, "power-law fit to Glassbrenner & Slack over 300-600 K",
         "Glassbrenner & Slack (1964)"))
# --- 4H-SiC (fit ranges not documented in the cited tables -> validity_unknown until checked)
_add(Law("sic_4h", "Eg", "varshni", dict(zip(("Eg0", "alpha", "beta"), _V)), "eV", {}, "", "Ayalew (2004) Table 3.2"))
for c in ("n", "p"):
    _add(Law("sic_4h", f"mu_{c}", "power_law", {"x300": (f"mu_{c}", "cm**2/V/s"), "g": (f"ct_Tmax_{c}", "dimensionless")},
             "cm**2/V/s", {}, "", "Ayalew (2004) Table 3.5"))
# --- oxides (no environment dependence modelled)
for m in ("sio2", "al2o3", "hfo2"):
    _add(Law(m, "eps_r", "constant", {"x300": ("eps_r", "dimensionless")}, "dimensionless", {}, "", m))
    _add(Law(m, "Eg", "constant", {"x300": ("Eg", "eV")}, "eV", {}, "", m))


def get(material: str, parameter: str) -> Law:
    try:
        return LAWS[(material, parameter)]
    except KeyError:
        raise KeyError(f"no law for {material}.{parameter}; known: {sorted(LAWS)}") from None


def evaluate(material: str, parameter: str, mat=None, **state) -> LawValue:
    return get(material, parameter)(mat, **state)


def spec(material: str, parameter: str, mat=None):
    return get(material, parameter).spec(mat)


# ------------------------------------------------------------------------------------------------ interfaces
def interfaces() -> dict:
    p = resources.files("simws.materials").joinpath("data", "interfaces.yaml")
    return yaml.safe_load(p.read_text())["interfaces"]


def interface(name: str, key: str) -> dict:
    d = interfaces()
    if name not in d or key not in d[name]:
        raise KeyError(f"no interface entry {name}.{key}; known: { {k: sorted(v) for k, v in d.items()} }")
    return d[name][key]


def sample_interface(name: str, key: str, n: int, seed: Optional[int] = None) -> np.ndarray:
    e = {k: (float(v) if k in ("low", "high", "mean", "sigma") else v) for k, v in interface(name, key).items()}
    rng = np.random.default_rng(seed)
    kind = e["distribution"]
    if kind == "loguniform":
        return 10 ** rng.uniform(math.log10(e["low"]), math.log10(e["high"]), n)
    if kind == "uniform":
        return rng.uniform(e["low"], e["high"], n)
    if kind == "normal":
        return rng.normal(e["mean"], e["sigma"], n)
    raise ValueError(f"unknown distribution {kind}")
