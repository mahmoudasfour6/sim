"""Physics models added to a DEVSIM drift-diffusion region (on top of simple_physics).

field_dependent_mobility   Caughey-Thomas velocity saturation, edge based:
                           mu = mu0 / (1 + (mu0 |E| / v_sat)^beta)^(1/beta)
impact_ionization          Van Overstraeten-de Man coefficients, alpha = a exp(-b/|E|), generation
                           G_ii = (alpha_n |Jn| + alpha_p |Jp|) / q, added to both continuity equations as an
                           edge-volume source (integrated over each edge's control volume).

|E| is the electrostatic field projected on each edge (ElectricField), the usual choice for edge-based
finite-volume schemes. All coefficients come from the material database (simws.materials).

Call order:
    sp.CreateSiliconPotentialOnly(...)                                   # defines ElectricField + derivatives
    mu_n, mu_p = field_dependent_mobility(dev, reg, mat)                 # returns edge-model names
    sp.CreateSiliconDriftDiffusion(dev, reg, mu_n=mu_n, mu_p=mu_p)
    impact_ionization(dev, reg, mat)                                     # after the currents exist
"""
from __future__ import annotations

from devsim.python_packages.model_create import CreateEdgeModel, CreateEdgeModelDerivatives

# Smooth magnitudes: |x| -> sqrt(x^2 + x0^2). A kink at zero (plain abs) makes Newton cycle where currents or
# fields change sign; the offsets (1 V/cm, 1e-12 A/cm^2) are far below any physically relevant value.
_EABS = "pow(ElectricField^2 + 1, 0.5)"
_JABS = "pow({0}^2 + 1e-24, 0.5)"


def field_dependent_mobility(device, region, mat, mu_n0="mu_n", mu_p0="mu_p"):
    """Create edge models mu_n_fd / mu_p_fd. mu_n0/mu_p0 may be parameters or edge models (e.g. mu_n_T from
    simws.coupling.electrothermal), so velocity saturation composes with lattice-temperature dependence."""
    vsat = mat.get("v_sat", "cm/s")
    out = []
    for carrier, mu0, beta in (("n", mu_n0, mat.get("ct_beta_n", "dimensionless")),
                               ("p", mu_p0, mat.get("ct_beta_p", "dimensionless"))):
        name = f"mu_{carrier}_fd"
        expr = f"{mu0} / pow(1 + pow({mu0} * {_EABS} / {vsat}, {beta}), {1.0 / beta})"
        CreateEdgeModel(device, region, name, expr)
        CreateEdgeModelDerivatives(device, region, name, expr, "Potential")
        out.append(name)
    return tuple(out)


def alpha_expressions(mat):
    an, bn = mat.get("ii_a_n", "1/cm"), mat.get("ii_b_n", "V/cm")
    apl, bpl = mat.get("ii_a_p_low", "1/cm"), mat.get("ii_b_p_low", "V/cm")
    aph, bph = mat.get("ii_a_p_high", "1/cm"), mat.get("ii_b_p_high", "V/cm")
    E0 = mat.get("ii_E0", "V/cm")
    alpha_n = f"({an} * exp(-{bn} / {_EABS}))"
    alpha_p = f"ifelse({_EABS} < {E0}, {apl} * exp(-{bpl} / {_EABS}), {aph} * exp(-{bph} / {_EABS}))"
    return alpha_n, alpha_p


def alpha_numpy(mat, E):
    """Same coefficients for post-processing (E in V/cm, numpy array). Returns (alpha_n, alpha_p) in 1/cm."""
    import numpy as np
    E = np.sqrt(np.asarray(E, float) ** 2 + 1.0)
    an = mat.get("ii_a_n", "1/cm") * np.exp(-mat.get("ii_b_n", "V/cm") / E)
    ap = np.where(E < mat.get("ii_E0", "V/cm"),
                  mat.get("ii_a_p_low", "1/cm") * np.exp(-mat.get("ii_b_p_low", "V/cm") / E),
                  mat.get("ii_a_p_high", "1/cm") * np.exp(-mat.get("ii_b_p_high", "V/cm") / E))
    return an, ap


def impact_ionization(device, region, mat):
    """Add avalanche generation to the electron and hole continuity equations (redefines both equations
    with the same names, variables and models as simple_physics, plus an edge-volume source)."""
    from devsim import equation

    alpha_n, alpha_p = alpha_expressions(mat)
    gii = (f"({alpha_n} * {_JABS.format('ElectronCurrent')} + {alpha_p} * {_JABS.format('HoleCurrent')})"
           " / ElectronCharge")
    CreateEdgeModel(device, region, "ImpactGeneration", gii)
    for v in ("Potential", "Electrons", "Holes"):
        CreateEdgeModelDerivatives(device, region, "ImpactGeneration", gii, v)
    for name, expr in (("ImpactSourceElectron", "+ElectronCharge * ImpactGeneration"),
                       ("ImpactSourceHole", "-ElectronCharge * ImpactGeneration")):
        CreateEdgeModel(device, region, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(device, region, name, expr, v)
    equation(device=device, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
             time_node_model="NCharge", edge_model="ElectronCurrent", variable_update="positive",
             node_model="ElectronGeneration", edge_volume_model="ImpactSourceElectron")
    equation(device=device, region=region, name="HoleContinuityEquation", variable_name="Holes",
             time_node_model="PCharge", edge_model="HoleCurrent", variable_update="positive",
             node_model="HoleGeneration", edge_volume_model="ImpactSourceHole")
