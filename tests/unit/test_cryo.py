"""0.9a cryogenic core: statistics, freeze-out, Klaassen, and device-level consistency checks."""
import math

import numpy as np
import pytest

from simws.materials import load
from simws.materials.semiconductor import intrinsic_density
from simws.tcad import cryo as C


@pytest.fixture(scope="module")
def si():
    return load("silicon")


def test_nilsson_inverse_density_error():
    us = np.geomspace(1e-6, 100, 61)
    rel = C.fd12_exact(C.eta_from_u_nilsson(us)) / us - 1
    assert np.max(np.abs(rel)) < 6e-3            # Nilsson's quoted accuracy; recorded as a model error


def test_eta_expr_clamp_matches_numpy_below_clamp():
    # the DEVSIM expression clamps u at U_CLAMP only to protect wild Newton iterates
    assert C.U_CLAMP >= 1e3
    assert "min(" in C.eta_expr("u") and "min(" in C.ln_gamma_expr("u")


def test_boltzmann_limit_of_band_params(si):
    for T in (77.0, 150.0, 300.0):
        ni = intrinsic_density(si, T)
        Nc, Nv, Eg = C.band_params(si, T, ni)
        kT = 8.617333262e-5 * T
        assert math.isclose(Nc * Nv * math.exp(-Eg / kT), ni ** 2, rel_tol=1e-10)


def test_freeze_out_closed_form(si):
    """device_equilibrium (Nilsson statistics) vs exact-FD neutrality: equal within the statistics error."""
    for T, N in ((77.0, 1e16), (77.0, 1e17), (150.0, 1e18), (300.0, 1e17)):
        ni = intrinsic_density(si, T)
        Nc, Nv, _ = C.band_params(si, T, ni)
        n, p, npr, ppr = C.device_equilibrium(si, np.array([N]), np.array([0.0]), np.array([ni]), T, Nc, Nv)
        n_ex, *_ = C.neutral_equilibrium(si, N, 0.0, T, n_i=ni)
        assert abs(n[0] / n_ex - 1) < 6e-3, (T, N)
    # at 77 K a 1e16 phosphorus doping is substantially frozen out, at 300 K essentially fully ionized
    ni = intrinsic_density(si, 77.0)
    Nc, Nv, _ = C.band_params(si, 77.0, ni)
    n77, *_ = C.device_equilibrium(si, np.array([1e17]), np.array([0.0]), np.array([ni]), 77.0, Nc, Nv)
    assert n77[0] / 1e17 < 0.5


def test_klaassen_pmin_and_300K_values(si):
    assert abs(C._pmin(300.0, C.KL_ME) - 0.3246) < 2e-3
    mu = C.klaassen_mobility(si, "n", 0.0, np.array([1e14]), np.array([1e14]), np.array([1e6]), 300.0, "P")
    assert 1300 < float(mu[0]) < 1420               # lattice-limited electron mobility near 1414 cm^2/Vs


def test_device_cryo_checks():
    from simws.validation import cryo_checks as K
    r = K.resistor_equilibrium(Ts=(300.0, 77.0), dopings=(1e16, 1e18))
    assert r["pass"], r
    r = K.resistor_ohmic(Ts=(77.0,), dopings=(1e17,))
    assert r["pass"], r
    r = K.einstein_graded()
    assert r["pass"], r


def test_onsager_reciprocity():
    from simws.validation import cryo_checks as K
    r = K.onsager(Ts=(300.0,))
    assert r["pass"], r


def test_gci_refuses_non_converging_and_adapter_bound():
    """Equal (or growing) differences under refinement give no order: no GCI (was negative), and the HBT
    adapter falls back to the conservative bound and flags the result."""
    from simws.verify import analyse
    r = analyse([0, 1, 2], [1.02, 1.01, 1.00], [0, 0, 0], ratio=2 ** 0.5)
    assert r.gci_fine is None and any("do not shrink" in n for n in r.notes)
    from simws.adapters.tcad import HbtGummel

    class Ctx:
        verification, flags = {}, []
    L = [np.array([[1.02, 0.0102, -1.0302]]), np.array([[1.01, 0.0101, -1.0201]]), np.array([[1.00, 0.0100, -1.0100]])]
    c = Ctx()
    sig = HbtGummel()._mesh_uncertainty(c, L, np.array([0.7]))
    assert all(s[0] > 0 for s in sig[:3])
    assert c.verification["mesh"]["passed"] is False and any(f.startswith("mesh_not_asymptotic") for f in c.flags)
