"""Versioned material-parameter database.

    >>> from simws.materials import load
    >>> si = load("silicon")                    # literature values
    >>> si.get("mu_n", "cm**2/V/s")
    1414.0
    >>> load("silicon", profile="charon").get("n_i", "cm**-3")   # values Charon uses internally
    14471200000.0
    >>> load("silicon", profile="dft").sigma("m_l")              # first-principles values carry an uncertainty

Each parameter carries a unit, a reference and a status (verified / provisional / computed). Values are returned
as plain floats in the unit the caller asks for, so unit mismatches fail loudly instead of silently.

Profiles overlay the base set. Hand-written profiles live in the material's YAML; computed ones (simws.studies.
dft_params) are YAML files {material: {profile: {key: {value, unit, sigma, ref, status}}}} found in, in order:
the `computed_dir` argument, $SIMWS_COMPUTED_DIR, and simws/materials/data/computed/ (promoted, git-tracked:
tools/promote_computed.py).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from importlib import resources
from typing import Dict, Optional

import yaml

from simws.units import Q_, ureg

_DATA = "data"
_NOT_MATERIALS = {"interfaces.yaml"}     # interface distributions: simws.materials.laws


@dataclass(frozen=True)
class Parameter:
    value: float
    unit: str
    ref: str
    status: str
    sigma: Optional[float] = None       # one standard uncertainty, same unit (computed parameters)

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

    def sigma(self, key: str, unit: Optional[str] = None) -> Optional[float]:
        p = self.parameters[key]
        if p.sigma is None:
            return None
        return float(Q_(p.sigma, p.unit).to(unit).magnitude) if unit else p.sigma

    def computed(self):
        return sorted(k for k, p in self.parameters.items() if p.status == "computed")

    def provisional(self):
        """Names of parameters not yet checked against their source (flag these in any paper)."""
        return sorted(k for k, p in self.parameters.items() if p.status not in ("verified", "computed"))

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
    return sorted(p.name[:-5] for p in resources.files(__package__).joinpath(_DATA).iterdir()
                  if p.name.endswith(".yaml") and p.name not in _NOT_MATERIALS)


def _computed_profile(name, profile, computed_dir):
    dirs = [d for d in (computed_dir, os.environ.get("SIMWS_COMPUTED_DIR")) if d]
    dirs.append(str(resources.files(__package__).joinpath(_DATA, "computed")))
    for d in dirs:
        f = Path(d) / f"{name}.{profile}.yaml"
        if f.is_file():
            doc = yaml.safe_load(f.read_text()) or {}
            entries = (doc.get(name) or {}).get(profile)
            if entries:
                return entries, str(f)
    return None, None


def computed_profiles(name: str, computed_dir: Optional[str] = None):
    """Names of computed profiles available for `name` (any search location)."""
    dirs = [d for d in (computed_dir, os.environ.get("SIMWS_COMPUTED_DIR")) if d]
    dirs.append(str(resources.files(__package__).joinpath(_DATA, "computed")))
    out = set()
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            out |= {f.name[len(name) + 1:-5] for f in p.glob(f"{name}.*.yaml")}
    return sorted(out)


def load(name: str, profile: Optional[str] = None, computed_dir: Optional[str] = None) -> Material:
    path = resources.files(__package__).joinpath(_DATA, f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"no material '{name}'; available: {available()}")
    raw = yaml.safe_load(path.read_text())
    entries = dict(raw["parameters"])
    if profile is not None:
        profiles = raw.get("profiles") or {}
        if profile in profiles:
            entries.update(profiles[profile])
        else:
            comp, _ = _computed_profile(name, profile, computed_dir)
            if comp is None:
                raise KeyError(f"{name}: no profile '{profile}'; available: {sorted(profiles)} "
                               f"+ computed {computed_profiles(name, computed_dir)}")
            entries.update(comp)
    params = {}
    for key, e in entries.items():
        unit = e["unit"]
        ureg.parse_units(unit)  # fail early on a malformed unit string
        sig = e.get("sigma")
        params[key] = Parameter(float(e["value"]), unit, e.get("ref", ""), e.get("status", "provisional"),
                                float(sig) if sig is not None else None)
    return Material(raw["name"], profile, params)
