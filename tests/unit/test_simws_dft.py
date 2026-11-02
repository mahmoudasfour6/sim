"""Unit tests for simws 0.7: first-principles parameter machinery (no DFT code needed)."""
import math

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")


class _FakeBands:
    """Parabolic bands in place of pw.x: three valence bands at Gamma, Delta valleys at 0.85 X."""

    def __init__(self, s, m_v=(0.5, 0.25, 0.2), m_l=0.92, m_t=0.19, k0=0.85):
        from simws.dft.bandedge import HB2_2M
        self.s, self.soc, self.nocc = s, False, 4
        self.c = HB2_2M * (2 * math.pi / s.a_angstrom) ** 2
        self.m_v, self.m_l, self.m_t, self.k0 = m_v, m_l, m_t, k0

    def energies(self, k):
        k = np.atleast_2d(k)
        n = len(k)
        E = np.zeros((n, 8))
        k2 = (k ** 2).sum(axis=1)
        for j, m in zip((3, 2, 1), self.m_v):
            E[:, j] = -self.c * k2 / m
        E[:, 0] = -50
        perp = k[:, 1] ** 2 + k[:, 2] ** 2
        E[:, 4] = 1.0 + self.c * ((k[:, 0] - self.k0) ** 2 / self.m_l + perp / self.m_t)
        E[:, 5] = 1.0 + self.c * ((k[:, 0] - (2 - self.k0)) ** 2 / self.m_l + perp / self.m_t)
        E[:, 6:] = 20
        return np.sort(E, axis=1)


def test_thermal_dos_reproduces_parabolic_bands():
    from simws.dft.bandedge import NC_M1_300, Setup, thermal_dos
    s = Setup(pseudo="x", h_dos=0.0125)
    fb = _FakeBands(s)
    r = thermal_dos(fb, fb, T=(300.0,), E_cbm=1.0, E_vbm=0.0)
    Nc_exact = NC_M1_300 * 6 * math.sqrt(fb.m_l * fb.m_t ** 2)
    Nv_exact = NC_M1_300 * sum(m ** 1.5 for m in fb.m_v)
    assert r["Nc_cm3"][0] == pytest.approx(Nc_exact, rel=0.01)
    assert r["Nv_cm3"][0] == pytest.approx(Nv_exact, rel=0.01)


def test_grid_weights_count_every_state_once():
    from simws.dft.bandedge import _grid_conduction, _grid_valence
    k, w = _grid_valence(0.2, 0.02)
    n = int(round(0.2 / 0.02))
    g = np.arange(-n, n + 1) * 0.02
    full = sum(1 for a in g for b in g for c in g if a * a + b * b + c * c <= 0.04 + 1e-12)
    assert w.sum() == full
    kc, wc = _grid_conduction(0.6, 0.1, 0.02)
    planes = int(round(0.4 / 0.02))
    disc = sum(1 for a in np.arange(-5, 6) for b in np.arange(-5, 6) if (a * a + b * b) * 4e-4 <= 0.01 + 1e-12)
    assert wc.sum() == disc * (2 * planes + 1)            # mirror images through X counted, X plane once


def test_mass_fit_and_power_law():
    from simws.dft.bandedge import HB2_2M, _fit_curvature, _mass
    from simws.studies.dft_params import _power_law
    a, m = 5.431, 0.19
    dk = 0.01 * np.arange(-4, 5)
    E = HB2_2M * (2 * math.pi / a) ** 2 * dk ** 2 / m + 3.0 * dk ** 4
    _, c2 = _fit_curvature(dk, E)
    assert _mass(c2, a) == pytest.approx(m, rel=1e-9)
    T = np.array([200.0, 300.0, 400.0, 500.0])
    A, p, resid = _power_law(T, 2.8e19 * (T / 300) ** 1.62)
    assert A == pytest.approx(2.8e19, rel=1e-12) and p == pytest.approx(1.62, rel=1e-12) and resid < 1e-12


def test_aggregate_combines_functional_and_numerical_spread():
    from simws.studies.dft_params import aggregate
    a = aggregate({"PBE": {"m_l": 0.96}, "LDA": {"m_l": 0.94}, "PBEsol": {"m_l": 0.95}}, {"m_l": 0.003})
    assert a["m_l"]["value"] == pytest.approx(0.95)
    assert a["m_l"]["sigma"] == pytest.approx(math.hypot(0.01, 0.003))


def test_computed_profile_loading_and_intrinsic_density(tmp_path):
    import yaml
    from simws.materials import load
    from simws.tcad.physics import intrinsic_density
    doc = {"silicon": {"testdft": {
        "Nc_300": {"value": 3.0e19, "unit": "cm**-3", "sigma": 1e18, "ref": "t", "status": "computed"},
        "Nv_300": {"value": 2.0e19, "unit": "cm**-3", "sigma": 1e18, "ref": "t", "status": "computed"},
        "Nc_T_exponent": {"value": 1.6, "unit": "dimensionless", "ref": "t", "status": "computed"},
        "Nv_T_exponent": {"value": 1.8, "unit": "dimensionless", "ref": "t", "status": "computed"},
        "m_l": {"value": 0.95, "unit": "dimensionless", "sigma": 0.01, "ref": "t", "status": "computed"}}}}
    (tmp_path / "silicon.testdft.yaml").write_text(yaml.safe_dump(doc))
    m = load("silicon", profile="testdft", computed_dir=str(tmp_path))
    assert m.get("m_l", "dimensionless") == 0.95 and m.sigma("m_l") == 0.01
    assert "m_l" in m.computed() and "m_l" not in m.provisional()
    assert m.get("mu_n", "cm**2/V/s") == load("silicon").get("mu_n", "cm**2/V/s")      # base set kept
    T = 400.0
    Eg = 1.170 - 4.73e-4 * T * T / (T + 636)
    ref = math.sqrt(3e19 * (T / 300) ** 1.6 * 2e19 * (T / 300) ** 1.8) * math.exp(-Eg / (2 * 8.617333262e-5 * T))
    assert intrinsic_density(m, T) == pytest.approx(ref, rel=1e-12)
    with pytest.raises(KeyError):
        load("silicon", profile="nonexistent", computed_dir=str(tmp_path))


def test_critical_thickness_limits():
    from simws.dft.mlip import critical_thickness
    from simws.studies.dft_params import C_EXP, THETA_DISMUKES_A
    es, eg = C_EXP["Si"], C_EXP["Ge"]
    h = [critical_thickness(x, THETA_DISMUKES_A, es, eg) for x in (0.1, 0.2, 0.3)]
    assert h[0] > h[1] > h[2] > 0                             # thinner for more misfit
    assert 5 < h[1] < 20                                      # ~10 nm at x = 0.2 (Matthews-Blakeslee)
    pb = critical_thickness(0.2, THETA_DISMUKES_A, es, eg, "people-bean")
    assert pb > 3 * h[1]                                      # metastable limit well above equilibrium


def test_qe_input_and_xml_parsing(tmp_path):
    from simws.dft.qe import PW, Structure
    st = Structure.diamond("Si", 5.431)
    assert st.volume_A3 == pytest.approx(5.431 ** 3 / 4, rel=1e-12)
    txt = PW(st, {"Si": "Si.upf"}, functional="hse", nqx=(4, 4, 4), soc=True, tstress=True).text("/p")
    for frag in ("input_dft = 'hse'", "nqx1 = 4", "lspinorb = .true.", "tstress = .true.", "CELL_PARAMETERS alat"):
        assert frag in txt
    bands = PW(st, {"Si": "Si.upf"}, calculation="bands", klist=np.zeros((2, 3))).text("/p")
    assert "K_POINTS tpiba\n  2" in bands


def test_birch_murnaghan_exact_on_synthetic_curve():
    from simws.dft.qe import birch_murnaghan
    V0, E0, B0, Bp = 41.0, -200.0, 98 / 160.21766208, 4.2
    V = np.linspace(38, 44, 7)
    eta = (V0 / V) ** (2 / 3)
    E = E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))
    r = birch_murnaghan(V, E)
    assert r["V0_A3"] == pytest.approx(41.0, rel=1e-10) and r["B0_GPa"] == pytest.approx(98.0, rel=1e-8)
    assert r["B0p"] == pytest.approx(4.2, rel=1e-8)


def test_hbt_base_constants_match_device():
    pytest.importorskip("devsim")
    from simws.studies.dft_params import HBT_BASE
    from simws.tcad.hbt import HBTCase
    c = HBTCase()
    assert HBT_BASE == (c.ge_start, c.ge_peak, c.w_b * 1e3)
