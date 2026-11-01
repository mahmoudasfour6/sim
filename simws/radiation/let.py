"""LET <-> pair density <-> deposited charge, using the material database (density, pair energy)."""
from simws.units import Q_E


def pairs_per_cm(let_MeV_cm2_mg, mat):
    rho_mg_cm3 = mat.get("density", "mg/cm**3")
    return let_MeV_cm2_mg * 1e6 * rho_mg_cm3 / mat.get("pair_energy", "eV")


def charge_fC_per_um(let_MeV_cm2_mg, mat):
    return pairs_per_cm(let_MeV_cm2_mg, mat) * Q_E * 1e15 * 1e-4


def let_from_charge(fC_per_um, mat):
    return fC_per_um / charge_fC_per_um(1.0, mat)


def let_from_dedx(MeV_per_um, mat):
    return MeV_per_um * 1e4 / mat.get("density", "mg/cm**3")
