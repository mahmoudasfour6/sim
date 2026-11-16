"""Engine-neutral problem description: see simws.spec.schema and docs/ARCHITECTURE_V1.md."""
from simws.spec.schema import (SPEC_VERSION, Analysis, Bias, Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium,
                               GradeTerm, MeshHints, ModelSpec, Numerics, Problem, Region, SmallSignal, Track)

__all__ = ["SPEC_VERSION", "Analysis", "Bias", "Box", "Contact", "DCSweep", "DeviceSpec", "DopingTerm", "Equilibrium",
           "GradeTerm", "MeshHints", "ModelSpec", "Numerics", "Problem", "Region", "SmallSignal", "Track"]
