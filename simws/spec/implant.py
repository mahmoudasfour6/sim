"""Engine-neutral description of an ion implantation (architecture v1, contract amendment A7).

An ImplantProblem says WHAT is implanted into WHAT: a beam (ion, energy, tilt, optional dose), a layered target
(atoms and their densities per layer), the collision models, and how many ions to follow. It is the second
problem family next to the device Problem of simws.spec.schema; no class here imports an engine.

Units are written in the field names: micrometres, keV for the beam, eV for the binding and cutoff energies,
atomic mass units, atoms per cm^3, ions per cm^2.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BINS = 5000


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Atom(_M):
    """An atomic species with the three energies every binary-collision code asks for."""
    symbol: str = Field(min_length=1)
    Z: int = Field(ge=1, le=118)
    mass_amu: float = Field(gt=0)                  # for an implanted ion: the isotope mass
    cutoff_eV: float = Field(gt=0)                 # a moving atom slower than this counts as stopped
    surface_binding_eV: float = Field(default=0.0, ge=0)
    bulk_binding_eV: float = Field(default=0.0, ge=0)


class Layer(_M):
    name: str
    thickness_um: float = Field(gt=0)
    density_cm3: Dict[str, float]                  # symbol -> atoms per cm^3

    @model_validator(mode="after")
    def _densities(self):
        if not self.density_cm3:
            raise ValueError(f"layer '{self.name}': density_cm3 needs at least one species")
        for sym, n in self.density_cm3.items():
            if not n > 0:
                raise ValueError(f"layer '{self.name}': density of {sym} must be > 0")
        return self


class Target(_M):
    atoms: List[Atom] = Field(min_length=1)
    layers: List[Layer] = Field(min_length=1)      # the beam side first
    structure: Literal["amorphous"] = "amorphous"

    @model_validator(mode="after")
    def _consistent(self):
        syms = [a.symbol for a in self.atoms]
        if len(set(syms)) != len(syms):
            raise ValueError("target: atom symbols must be unique")
        names = [la.name for la in self.layers]
        if len(set(names)) != len(names):
            raise ValueError("target: layer names must be unique")
        for la in self.layers:
            for sym in la.density_cm3:
                if sym not in syms:
                    raise ValueError(f"layer '{la.name}': species '{sym}' is not in target.atoms")
        return self

    @property
    def thickness_um(self) -> float:
        return float(sum(la.thickness_um for la in self.layers))


class Beam(_M):
    ion: Atom
    energy_keV: float = Field(gt=0)
    tilt_deg: float = Field(default=0.0, ge=0, lt=90)        # from the surface normal
    dose_cm2: Optional[float] = Field(default=None, gt=0)


class ImplantModels(_M):
    potential: Literal["zbl", "kr_c", "moliere"] = "zbl"
    electronic_stopping: Literal["lindhard_scharff"] = "lindhard_scharff"
    recoils: bool = False                          # follow the knocked-on target atoms (needed for damage)


class Statistics(_M):
    n_ions: int = Field(default=20000, ge=100, le=2_000_000)
    seed: int = Field(default=1, ge=0)


class ImplantProblem(_M):
    spec_version: str = "1.0"
    kind: Literal["implant"] = "implant"
    beam: Beam
    target: Target
    models: ImplantModels = Field(default_factory=ImplantModels)
    statistics: Statistics = Field(default_factory=Statistics)
    depth_bin_um: float = Field(default=0.005, gt=0)

    @model_validator(mode="after")
    def _bins(self):
        if self.target.thickness_um / self.depth_bin_um > MAX_BINS:
            raise ValueError(f"depth_bin_um: {self.target.thickness_um:g} um of target in {self.depth_bin_um:g} um "
                             f"bins is more than {MAX_BINS} bins; use a larger bin or a thinner target")
        return self

    def problem_id(self) -> str:
        """Content hash of the whole description (32 hex characters), as Problem.problem_id."""
        from simws.hub.canonical import job_id
        return job_id(self.model_dump(mode="json"))

    def with_changes(self, changes: Dict[str, object]) -> "ImplantProblem":
        """A copy with dotted-path overrides, validated again:
        p.with_changes({"beam.energy_keV": 50.0, "statistics.seed": 2})"""
        d = self.model_dump(mode="json")
        for path, value in changes.items():
            node = d
            parts = path.split(".")
            for k in parts[:-1]:
                node = node[int(k)] if isinstance(node, list) else node[k]
            last = parts[-1]
            if isinstance(node, list):
                node[int(last)] = value
            else:
                if last not in node:
                    raise KeyError(f"no field '{path}' in the problem description")
                node[last] = value
        return ImplantProblem.model_validate(d)
