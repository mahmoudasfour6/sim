"""simws.engines.charon_engine (contract section 4.5): capability, engine-specific refusals, deck writing and the
not-installed path. Charon is not installed on the build machine; nothing here runs Charon."""
import re
from pathlib import Path

import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d

NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _E():
    from simws.engines import charon_engine
    return charon_engine


def _numbers(text):
    return [float(s) for s in NUM.findall(text)]


def _has(nums, value, rel=1e-9):
    return any(abs(n - value) <= rel * abs(value) for n in nums)


def _charon_installed():
    from simws.config import path
    try:
        return path("toolchain.charon_exe").exists()
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ capability / refusals
def test_module_surface():
    E = _E()
    for f in ("CAPABILITY", "unsupported", "available", "run", "prepare", "collect"):
        assert hasattr(E, f), f
    assert E.CAPABILITY.engine == "charon" and E.CAPABILITY.in_process is False


def test_supported_problem_has_no_reasons(charon_problem):
    E = _E()
    p = charon_problem()
    assert E.unsupported(p) == []
    assert E.CAPABILITY.outside_tested_range(p) == []
    assert E.CAPABILITY.outside_tested_range(charon_problem(T=77.0)) == []        # bound included
    assert E.unsupported(charon_problem(T=77.0)) == []
    r = E.CAPABILITY.outside_tested_range(charon_problem(T=70.0))
    assert len(r) == 1 and r[0].startswith("temperature_K:")


def test_default_diode_refused_with_reasons():
    E = _E()
    r = E.unsupported(pn_diode_1d(V=[0.1, 0.2]))         # 4 um / 1 um, Boltzmann, no incomplete ionization
    assert "mesh: no Charon mesh for this geometry" in r
    assert "statistics: 'boltzmann' not in ['fermi_dirac']" in r
    assert "incomplete_ionization: 'off' not in ['constant_energy']" in r


def test_equilibrium_refused(charon_problem):
    p = charon_problem().with_changes({"analysis": {"kind": "equilibrium"}})
    assert "analyses: 'equilibrium' not in ['dc_sweep']" in _E().unsupported(p)


def test_unequal_lifetimes_refused(charon_problem):
    p = charon_problem().with_changes({"models.tau_p_s": 2e-7})
    assert _E().unsupported(p) == ["srh: Charon deck takes one lifetime for both carriers"]


@pytest.mark.parametrize("doping", [
    # three terms
    [{"species": "acceptor", "concentration_cm3": 1e17, "box": {"x": [None, 0.5]}},
     {"species": "donor", "concentration_cm3": 1e17, "box": {"x": [0.5, None]}},
     {"species": "donor", "concentration_cm3": 1e18, "box": {"x": [0.9, None]}}],
    # n side at x_min (donor first)
    [{"species": "donor", "concentration_cm3": 1e17, "box": {"x": [None, 0.5]}},
     {"species": "acceptor", "concentration_cm3": 1e17, "box": {"x": [0.5, None]}}],
    # one term only
    [{"species": "acceptor", "concentration_cm3": 1e17}],
])
def test_non_step_doping_refused(charon_problem, doping):
    p = charon_problem().with_changes({"device.doping": doping})
    assert "doping: Charon translator supports an abrupt two-term pn step only" in _E().unsupported(p)


def test_sweep_must_be_anode_only(charon_problem):
    E = _E()
    p1 = charon_problem().with_changes({"analysis.contact": "cathode"})
    assert any(s.startswith("analysis:") for s in E.unsupported(p1)), E.unsupported(p1)
    p2 = charon_problem().with_changes({"analysis.fixed_V": {"cathode": 0.0}})
    assert any(s.startswith("analysis:") for s in E.unsupported(p2)), E.unsupported(p2)
    p3 = charon_problem().with_changes({"analysis.tracking": [{"contact": "cathode"}]})
    assert any(s.startswith("analysis:") for s in E.unsupported(p3)), E.unsupported(p3)


def test_geometry_must_match_a_mesh(charon_problem):
    E = _E()
    p = charon_problem().with_changes({"device.regions.0.box.x": [0.0, 2.0]})
    assert "mesh: no Charon mesh for this geometry" in E.unsupported(p)
    q = charon_problem().with_changes({"device.doping.0.box.x": [None, 0.4], "device.doping.1.box.x": [0.4, None]})
    assert "mesh: no Charon mesh for this geometry" in E.unsupported(q)


def test_hbt_refused():
    r = _E().unsupported(npn_hbt_2d(VBE=[0.6]))
    assert "dimensions: 2 not in [1]" in r
    assert "grading_elements: 'Ge' not in []" in r


# ------------------------------------------------------------------------------------------------ availability / decks
def test_not_available_here():
    if _charon_installed():
        pytest.skip("Charon is installed on this machine")
    assert _E().available() is False


# Values chosen to differ from charon_xcheck's own defaults, so a translator that ignores the problem fails.
VALS = dict(NA=2e17, ND=5e16, mu_n=1100.0, mu_p=450.0, tau=3e-7, T=250.0, E_ion=0.044, N_crit=2e18)


def _deck_problem(charon_problem):
    return charon_problem(V=(0.0, 0.1, 0.2, 0.3), **VALS)


def test_prepare_writes_decks_with_problem_values(charon_problem, tmp_path):
    p = _deck_problem(charon_problem)
    files = _E().prepare(p, tmp_path)
    assert files and all(Path(f).exists() or (tmp_path / f).exists() for f in files), files
    decks = sorted(tmp_path.glob("*.inp"))
    assert len(decks) >= len(p.analysis.values_V), [d.name for d in decks]
    assert list(tmp_path.glob("runlist*.txt")), "no run list written"
    anode_V = set()
    for d in decks:
        text = d.read_text()
        nums = _numbers(text)
        for key in ("NA", "ND", "mu_n", "mu_p", "tau", "T", "E_ion", "N_crit"):
            assert _has(nums, VALS[key]), (d.name, key, VALS[key])
        assert "use Fermi Dirac" in text
        m = re.search(r"BC is ohmic for anode on silicon fixed at\s+(\S+)", text)
        assert m, d.name
        anode_V.add(round(float(m.group(1)), 9))
    assert set(round(v, 9) for v in p.analysis.values_V) <= anode_V, anode_V
    # every deck named in the run list exists
    for rl in tmp_path.glob("runlist*.txt"):
        for name in rl.read_text().split():
            assert (tmp_path / f"{name}.inp").exists() or (tmp_path / name).exists(), name


def test_prepare_srh_off_deck(charon_problem, tmp_path):
    p = charon_problem().with_changes({"models.srh": "off", "models.tau_n_s": None, "models.tau_p_s": None})
    assert _E().unsupported(p) == []
    _E().prepare(p, tmp_path)
    decks = list(tmp_path.glob("*.inp"))
    assert decks and all("srh recombination is off" in d.read_text() for d in decks)


def test_run_raises_not_available_after_prepare(charon_problem, tmp_path):
    if _charon_installed():
        pytest.skip("Charon is installed on this machine")
    from simws.engines.base import EngineNotAvailable
    p = _deck_problem(charon_problem)
    with pytest.raises(EngineNotAvailable):
        _E().run(p, tmp_path)
    assert list(tmp_path.glob("*.inp")), "run() must prepare the decks before refusing"


def test_run_refuses_unsupported_problem(tmp_path):
    from simws.engines.base import Unsupported
    with pytest.raises(Unsupported) as ei:
        _E().run(pn_diode_1d(V=[0.1]), tmp_path)
    assert "mesh: no Charon mesh for this geometry" in ei.value.reasons
