"""Engine-neutral problem descriptions: see simws.spec.schema (devices), simws.spec.implant (ion implantation)
and docs/ARCHITECTURE_V1.md."""
from simws.spec.implant import Atom, Beam, ImplantModels, ImplantProblem, Layer, Statistics, Target
from simws.spec.schema import (SPEC_VERSION, Analysis, Bias, Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium,
                               GradeTerm, MeshHints, ModelSpec, Numerics, Problem, Region, SmallSignal, Track)


def family(problem) -> str:
    """The problem family (contract A7.1): "implant" for an ImplantProblem, "device" for a Problem."""
    return "implant" if isinstance(problem, ImplantProblem) else "device"


__all__ = ["SPEC_VERSION", "Analysis", "Bias", "Box", "Contact", "DCSweep", "DeviceSpec", "DopingTerm", "Equilibrium",
           "GradeTerm", "MeshHints", "ModelSpec", "Numerics", "Problem", "Region", "SmallSignal", "Track",
           "Atom", "Beam", "ImplantModels", "ImplantProblem", "Layer", "Statistics", "Target", "family"]
