"""Engine-neutral problem description: see simws.spec.schema and docs/ARCHITECTURE_V1.md."""
from simws.spec.schema import (SPEC_VERSION, Analysis, Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium,
                               GradeTerm, MeshHints, ModelSpec, Numerics, Problem, Region, Track)

__all__ = ["SPEC_VERSION", "Analysis", "Box", "Contact", "DCSweep", "DeviceSpec", "DopingTerm", "Equilibrium",
           "GradeTerm", "MeshHints", "ModelSpec", "Numerics", "Problem", "Region", "Track"]
