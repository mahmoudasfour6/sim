"""Channeling in (100) silicon from published analytic fits (architecture v1, contract A15).

Not a program: the engine evaluates Tasch et al. 1989 (boron: two Pearson densities, fitted to SIMS) or Suzuki et
al. 2000 (arsenic: joined half Gaussian with a channeling tail, fitted to tuned crystal-TRIM) for an ImplantProblem
on crystalline silicon, at the one pair of beam angles each paper covers. No ions are followed, so there is no
lateral spread and no statistics; the uncertainty is that of reading the papers' figures (A15.5).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from simws.engines.base import ImplantCapability, ImplantResult, Unsupported

FAMILY = "implant"
CODE = "published_fits"
NAME = "channeling"
ANGLE_TOL_DEG = 0.5
LOSS_LIMIT = 1e-4          # A15.6 R81: the target may cut off at most this share of the profile
FINE = 20                  # sub-points per depth bin for the bin average

CAPABILITY = ImplantCapability(
    engine=NAME, env="tcad", in_process=True,
    description="Channeling in (100) silicon from fits: boron (Tasch et al. 1989, dual Pearson, SIMS), arsenic "
                "(Suzuki et al. 2000, joined half Gaussian with tail, tuned Monte Carlo) and phosphorus (simws's fit to "
                "the SIMS of Ghante et al. 1995, A21)",
    structures=["crystalline_si_100"], potentials=["zbl", "kr_c", "moliere"],
    electronic_stopping=["lindhard_scharff", "empirical"], recoils=[False], max_layers=1, max_keV_per_amu=25.0,
    tested_keV_per_amu=(0.48, 5.81), tested_target_symbols=["Si"], tested_ion_symbols=["B", "As", "P"],
    notes="Boron 15-50 keV, 2e13-2e15 cm^-2, tilt 7 deg, rotation 30 deg; arsenic 40-200 keV, 1e11-1e16 cm^-2, "
          "tilt 7 deg, rotation 22 deg; phosphorus 15-180 keV at 1e14 (tilt 4, rotation 22) and tilt 10, rotation 0 "
          "(30-120 keV at 1e14, 1e13-8e15 at 50 keV); one layer of bare crystalline silicon; a dose is required. "
          "Depth profile only (no lateral spread). Unverified by rule: no second channeling engine.")

FLAG_FMT = ("implant:published_fit: {cite}; {conditions}; the models and statistics of the description are not "
            "used (A15)")
BUDGET_FLAG = ("budget:digitisation_only: the uncertainty is that of reading the paper's figures; the paper's own fit "
               "error is not included (A15.5)")


def _model(problem):
    """The fit for this ion; for phosphorus the measured condition matching the beam's angles (A21), or None."""
    from simws.radiation import channeling as ch
    m = ch.model(problem.beam.ion.symbol)
    if m is not None and hasattr(m, "select"):
        return m.select(problem.beam.tilt_deg, problem.beam.rotation_deg, ANGLE_TOL_DEG)
    return m


def unsupported(problem) -> List[str]:
    out = CAPABILITY.unsupported(problem)
    m = _model(problem)
    ion = problem.beam.ion
    if ion.symbol == "P" and round(ion.mass_amu) == 31 and m is None:
        out.append(f"angles: tilt {problem.beam.tilt_deg:g} deg, rotation {problem.beam.rotation_deg} deg; phosphorus "
                   f"is fitted at tilt 4 / rotation 22 and tilt 10 / rotation 0 only (A21.2; set beam.rotation_deg)")
        return out
    if m is None or (ion.symbol == "B" and round(ion.mass_amu) != 11) or (ion.symbol == "As" and round(ion.mass_amu) != 75) \
            or (ion.symbol == "P" and round(ion.mass_amu) != 31):
        out.append(f"ion: {ion.symbol} ({ion.mass_amu:g} u) has no channeling fit here (11B: Tasch 1989; "
                   f"75As: Suzuki 2000; 31P: simws's fit to Ghante 1995)")
        return out
    layers = problem.target.layers
    if len(layers) != 1 or set(layers[0].density_cm3) != {"Si"}:
        out.append("structures: the fits are for one layer of bare crystalline silicon")
    E = problem.beam.energy_keV
    if not m.energies[0] - 1e-9 <= E <= m.energies[1] + 1e-9:
        out.append(f"energy: {E:g} keV outside {m.energies[0]:g}-{m.energies[1]:g} keV for {m.ion} (A15.2)")
    d = problem.beam.dose_cm2
    if d is None:
        out.append("dose: the channeled share depends on the dose; give beam.dose_cm2")
    elif not m.doses[0] * (1 - 1e-9) <= d <= m.doses[1] * (1 + 1e-9):
        out.append(f"dose: {d:.3g} cm^-2 outside {m.doses[0]:.0e}-{m.doses[1]:.0e} for {m.ion} (A15.2)")
    if d is not None and hasattr(m, "valid") and m.valid(E, d):
        out.append(m.valid(E, d))
    if abs(problem.beam.tilt_deg - m.tilt_deg) > ANGLE_TOL_DEG:
        out.append(f"angles: tilt {problem.beam.tilt_deg:g} deg; the {m.ion} fit is for {m.tilt_deg:g} deg")
    rot = problem.beam.rotation_deg
    if rot is None or abs(rot - m.rotation_deg) > ANGLE_TOL_DEG:
        out.append(f"angles: rotation {rot} deg; the {m.ion} fit is for {m.rotation_deg:g} deg (set beam.rotation_deg)")
    if not out:                                     # A15.6 R81: the target must hold the profile (visible to plan())
        lost = share_beyond(m, problem)
        if lost > LOSS_LIMIT:
            out.append(f"structures: the target ({problem.target.thickness_um:g} um) cuts off {lost:.2e} of the "
                       f"profile (limit {LOSS_LIMIT:g}); make it thicker")
    return out


def share_beyond(m, problem) -> float:
    """Share of the [0, inf)-normalised profile deeper than the target, by a fine quadrature to 50 um."""
    T = problem.target.thickness_um
    x_in = np.linspace(0.0, T, 20001)
    x_out = np.linspace(T, max(50.0, 5 * T), 40001)
    d_in = np.asarray(m.profile(x_in, problem.beam.energy_keV, problem.beam.dose_cm2, 0)["density"], float)
    d_out = np.asarray(m.profile(x_out, problem.beam.energy_keV, problem.beam.dose_cm2, 0)["density"], float)
    out = float(np.trapezoid(d_out, x_out))
    total = float(np.trapezoid(d_in, x_in)) + out
    return max(0.0, out / total) if total > 0 else 1.0


def available() -> bool:
    return True


def _evaluate(m, problem, perturb: int):
    """Bin-averaged density (1/um per ion) on the problem's depth bins, the channeled share, the share lost beyond
    the target, and the mean and standard deviation of the profile within the target (um)."""
    T = problem.target.thickness_um
    b = problem.depth_bin_um
    nb = int(math.ceil(T / b - 1e-9))
    xf = (np.arange(nb * FINE) + 0.5) * (b / FINE)
    out = m.profile(xf, problem.beam.energy_keV, problem.beam.dose_cm2, perturb)
    dens = np.asarray(out["density"], float)
    binned = dens.reshape(nb, FINE).mean(axis=1)
    inside = float(np.sum(dens) * (b / FINE))      # bin-average quadrature; the share beyond is share_beyond()
    centres = (np.arange(nb) + 0.5) * b
    w = binned / binned.sum()
    rp = float((w * centres).sum())
    drp = float(math.sqrt((w * (centres - rp) ** 2).sum()))
    return binned, float(out["channeled_fraction"]), 1.0 - inside, rp, drp


def run(problem, workdir=None) -> ImplantResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    m = _model(problem)
    p0, fc0, _, rp0, drp0 = _evaluate(m, problem, 0)
    lost = share_beyond(m, problem)
    lo = _evaluate(m, problem, -1)
    hi = _evaluate(m, problem, +1)
    sig = 0.5 * np.abs(hi[0] - lo[0])
    b = problem.depth_bin_um
    centres = ((np.arange(p0.size) + 0.5) * b).tolist()
    diag = {"model": type(m).__name__.lower(), "source": m.cite, "share_beyond_target": lost,
            "Rp_um": rp0, "Rp_sigma_um": 0.5 * abs(hi[3] - lo[3]), "dRp_um": drp0,
            "dRp_sigma_um": 0.5 * abs(hi[4] - lo[4])}
    from simws.radiation import channeling as ch
    if m.ion == "P":                     # A21.9: not a published fit; the conditions go in the diagnostics
        flags = [ch.OWN_FIT_FLAG, ch.SIMS_FLAG]
        diag["conditions"] = m.conditions
    else:
        flags = [FLAG_FMT.format(cite=m.cite, conditions=m.conditions), BUDGET_FLAG]
    return ImplantResult(engine=NAME, problem_id=problem.problem_id(), n_ions=0, n_reflected=0, n_transmitted=0,
                         depth_um=[], lateral_um=[], profile_um=centres, profile_per_um=p0.tolist(),
                         profile_sigma_per_um=sig.tolist(), channeled_fraction=fc0,
                         channeled_fraction_sigma=0.5 * abs(hi[1] - lo[1]),
                         flags=flags, diagnostics=diag)
