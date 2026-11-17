"""Engine-neutral description of a single-event energy-deposition problem (architecture v1, amendment A8).

A SeeProblem says WHAT hits WHAT: a particle source (species, energy, tilt), a target (a bulk block with one
sensitive volume (SV) inside it and optional back-end-of-line layers on top), the physics to use, how to histogram
the energy each event leaves in the SV, and how many events to follow. No class here imports an engine.

Units are written in the field names: micrometres, MeV. Materials are Geant4 NIST material names ("G4_Si").
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

G4_MATERIAL = r"^G4_[A-Za-z0-9_]+$"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeeSource(_M):
    particle: Literal["proton", "alpha", "neutron", "e-", "gamma", "ion"]
    ion_Z: Optional[int] = Field(default=None, ge=1, le=118)
    ion_A: Optional[int] = None
    energy_MeV: float = Field(gt=0)              # kinetic energy of the whole particle
    tilt_deg: float = Field(default=0.0, ge=0, lt=90)

    @model_validator(mode="after")
    def _ion(self):
        if self.particle == "ion":
            if self.ion_Z is None or self.ion_A is None:
                raise ValueError("source: particle 'ion' needs ion_Z and ion_A")
            if self.ion_A < self.ion_Z:
                raise ValueError("source: ion_A must be >= ion_Z")
        elif self.ion_Z is not None or self.ion_A is not None:
            raise ValueError(f"source: ion_Z and ion_A are only for particle 'ion', not '{self.particle}'")
        return self


class BeolLayer(_M):
    name: str = Field(min_length=1)
    material: str = Field(pattern=G4_MATERIAL)
    thickness_um: float = Field(gt=0)


class SeeGeometry(_M):
    bulk_material: str = Field(default="G4_Si", pattern=G4_MATERIAL)   # the SV is of the same material
    bulk_width_um: float = Field(gt=0)
    bulk_thickness_um: float = Field(gt=0)
    sv_depth_um: float = Field(default=0.0, ge=0)                       # SV top below the bulk top
    sv_width_um: Tuple[float, float]
    sv_thickness_um: float = Field(gt=0)
    beol: List[BeolLayer] = []                                          # the beam side first

    @model_validator(mode="after")
    def _inside(self):
        for w in self.sv_width_um:
            if not 0 < w <= self.bulk_width_um:
                raise ValueError(f"geometry: SV width {w:g} um must be > 0 and <= the bulk width "
                                 f"{self.bulk_width_um:g} um")
        if self.sv_depth_um + self.sv_thickness_um > self.bulk_thickness_um:
            raise ValueError("geometry: the SV (depth + thickness) must lie inside the bulk")
        names = [b.name for b in self.beol]
        if len(set(names)) != len(names):
            raise ValueError("geometry: BEOL layer names must be unique")
        return self

    @property
    def beol_thickness_um(self) -> float:
        return float(sum(b.thickness_um for b in self.beol))


class SeePhysics(_M):
    em: Literal["option4", "option3", "livermore"] = "option4"
    hadronic: Literal["none", "ftfp_bert_hp"] = "none"


class EdepHistogram(_M):
    scale: Literal["lin", "log"] = "lin"
    bins: int = Field(default=2000, ge=10, le=20000)
    low_MeV: float = Field(default=0.0, ge=0)
    high_MeV: float

    @model_validator(mode="after")
    def _range(self):
        if not self.high_MeV > self.low_MeV:
            raise ValueError("histogram: high_MeV must be > low_MeV")
        if self.scale == "log" and not self.low_MeV > 0:
            raise ValueError("histogram: a log histogram needs low_MeV > 0")
        return self


class SeeStatistics(_M):
    n_events: int = Field(default=10000, ge=100, le=100_000_000)
    seed: int = Field(default=1, ge=0)


class SeeProblem(_M):
    spec_version: str = "1.0"
    kind: Literal["see"] = "see"
    source: SeeSource
    geometry: SeeGeometry
    physics: SeePhysics = Field(default_factory=SeePhysics)
    histogram: EdepHistogram
    statistics: SeeStatistics = Field(default_factory=SeeStatistics)

    def problem_id(self) -> str:
        """Content hash of the whole description (32 hex characters), as Problem.problem_id."""
        from simws.hub.canonical import job_id
        return job_id(self.model_dump(mode="json"))

    def with_changes(self, changes: Dict[str, object]) -> "SeeProblem":
        """A copy with dotted-path overrides, validated again:
        p.with_changes({"source.energy_MeV": 10.0, "statistics.seed": 2})"""
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
        return SeeProblem.model_validate(d)
