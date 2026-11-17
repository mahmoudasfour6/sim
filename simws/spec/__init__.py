"""Engine-neutral problem descriptions: see simws.spec.schema (devices), simws.spec.implant (ion implantation),
simws.spec.see (single-event energy deposition) and docs/ARCHITECTURE_V1.md."""
from simws.spec.implant import Atom, Beam, ImplantModels, ImplantProblem, Layer, Statistics, Target
from simws.spec.see import (BeolLayer, EdepHistogram, SeeGeometry, SeePhysics, SeeProblem, SeeSource,
                             SeeStatistics)
from simws.spec.schema import (SPEC_VERSION, Analysis, Bias, Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium,
                               GradeTerm, MeshHints, ModelSpec, Numerics, Problem, Region, SmallSignal, Track)


def family(problem) -> str:
    """The problem family (contracts A7.1, A8.1): "implant" for an ImplantProblem, "see" for a SeeProblem,
    "device" for a Problem."""
    if isinstance(problem, ImplantProblem):
        return "implant"
    if isinstance(problem, SeeProblem):
        return "see"
    return "device"


__all__ = ["SPEC_VERSION", "Analysis", "Bias", "Box", "Contact", "DCSweep", "DeviceSpec", "DopingTerm", "Equilibrium",
           "GradeTerm", "MeshHints", "ModelSpec", "Numerics", "Problem", "Region", "SmallSignal", "Track",
           "Atom", "Beam", "ImplantModels", "ImplantProblem", "Layer", "Statistics", "Target",
           "BeolLayer", "EdepHistogram", "SeeGeometry", "SeePhysics", "SeeProblem", "SeeSource", "SeeStatistics",
           "family"]
