"""Versioned material-parameter database.

    >>> from simws.materials import load
    >>> si = load("silicon")                    # literature values
    >>> si.get("mu_n", "cm**2/V/s")
    1414.0
    >>> load("silicon", profile="charon").get("n_i", "cm**-3")   # values Charon uses internally
    14471200000.0

Each parameter carries a unit, a reference and a status (verified / provisional). Values are returned as
plain floats in the unit the caller asks for, so unit mismatches fail loudly instead of silently.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, Optional

import yaml

from simws.units import Q_, ureg

_DATA = "data"


@dataclass(frozen=True)
class Parameter:
    value: float
    unit: str
    ref: str
    status: str

    @property
    def quantity(self):
        return Q_(self.value, self.unit)


@dataclass
class Material:
    name: str
    profile: Optional[str]
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    overrides: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, unit: str) -> float:
        """Value of `key` converted to `unit` (pint syntax, e.g. 'cm**2/V/s', 'W/cm/K', 'cm**-3')."""
        if key not in self.parameters:
            raise KeyError(f"{self.name}: no parameter '{key}'. Available: {sorted(self.parameters)}")
        return float(self.parameters[key].quantity.to(unit).magnitude)

    def quantity(self, key: str):
        return self.parameters[key].quantity

    def provisional(self):
        """Names of parameters not yet checked against their source (flag these in any paper)."""
        return sorted(k for k, p in self.parameters.items() if p.status != "verified")

    def with_overrides(self, **values_with_units) -> "Material":
        """Case-specific overrides, e.g. m.with_overrides(n_i=(1e10, 'cm**-3')). Recorded in .overrides."""
        new = copy.deepcopy(self)
        for k, (val, unit) in values_with_units.items():
            Q_(val, unit)  # validates the unit string
            new.parameters[k] = Parameter(float(val), unit, "case-specific override", "override")
            new.overrides[k] = float(val)
        return new

    def table(self) -> str:
        rows = [f"{k:22s} {p.value:>12.6g} {p.unit:14s} {p.status:12s} {p.ref}" for k, p in sorted(self.parameters.items())]
        return f"{self.name} (profile={self.profile})\n" + "\n".join(rows)


def available():
    return sorted(p.name[:-5] for p in resources.files(__package__).joinpath(_DATA).iterdir() if p.name.endswith(".yaml"))


def load(name: str, profile: Optional[str] = None) -> Material:
    path = resources.files(__package__).joinpath(_DATA, f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"no material '{name}'; available: {available()}")
    raw = yaml.safe_load(path.read_text())
    entries = dict(raw["parameters"])
    if profile is not None:
        profiles = raw.get("profiles") or {}
        if profile not in profiles:
            raise KeyError(f"{name}: no profile '{profile}'; available: {sorted(profiles)}")
        entries.update(profiles[profile])
    params = {}
    for key, e in entries.items():
        unit = e["unit"]
        ureg.parse_units(unit)  # fail early on a malformed unit string
        params[key] = Parameter(float(e["value"]), unit, e.get("ref", ""), e.get("status", "provisional"))
    return Material(raw["name"], profile, params)
