"""Contract amendment A14: damage (vacancy) profiles from both implant engines. Description (A14.2), engine output
(A14.3), adapter quantities and error bars (A14.4) and the acceptance checks R70 to R76. Written from
docs/ARCHITECTURE_V1.md (sections 2, 6, A2, A7, A12, A13, A14) only; the implementation files were not read.

R70's "checker reads one batch directly": the checker writes the RustBCA input for one batch itself (build_input of
the problem, with the A14.3 changes applied by the checker: track_recoils and track_displacements true, the atoms'
Ed, the batch's ion count and seed seed * 1000 + b), writes it as TOML with its own writer, runs RustBCA's
command-line program (SIMWS_RUSTBCA_BIN) and counts the displacement rows whose start lies in the target. RustBCA
gives every ion its own random stream (stream = ion index), so the result does not depend on num_chunks or threads.

Damage runs need the RustBCA command-line program (engine damage availability / SIMWS_RUSTBCA_BIN) or the Geant4
program (<SIMWS_G4SEE_PREFIX>/bin/g4implant); tests are skipped where they are not installed."""
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

UM = 1e-6
ADAPTERS = {"rustbca": "implant.rustbca", "g4implant": "implant.g4implant"}
DAMAGE_FLAG = "implant:damage_full_cascade: no recombination or annealing; an upper bound on surviving defects (A14.1)"
BATCHES = 20


def _implant(*a, **k):
    from simws.spec.library import implant_si
    return implant_si(*a, **k)


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _mod(name):
    from simws import engines
    return engines.module(name)


def _need(engine):
    if engine == "rustbca":
        pytest.importorskip("libRustBCA")
        mod = _mod("rustbca")
        ok = getattr(mod, "damage_available", lambda: False)()
        b = os.environ.get("SIMWS_RUSTBCA_BIN")
        if not (ok or (b and Path(b).exists())):
            pytest.skip("RustBCA command-line program not installed")
    else:
        pre = os.environ.get("SIMWS_G4SEE_PREFIX")
        if not ((pre and (Path(pre) / "bin" / "g4implant").exists()) or _mod("g4implant").available()):
            pytest.skip("g4implant is not installed")


def _dmg(ion, E=50.0, Ed=15.0, **k):
    return _implant(ion, E, n_ions=k.pop("n_ions", 2000), displacement_eV=Ed, **k)


# ------------------------------------------------------------------------------------------------ capture fixture
@pytest.fixture(scope="module")
def captured():
    """Wraps both engine modules' run() so that the ImplantResult behind every hub submission of this module is kept
    (key: engine, problem id). The adapter is otherwise unchanged."""
    from simws import engines
    from simws.spec import ImplantProblem
    store = {}
    mp = pytest.MonkeyPatch()
    for name in ADAPTERS:
        mod = engines.module(name)

        def spy(problem, *a, _real=mod.run, _name=name, **k):
            res = _real(problem, *a, **k)
            p = problem if isinstance(problem, ImplantProblem) else ImplantProblem.model_validate(problem)
            store[(_name, p.problem_id())] = res
            return res
        mp.setattr(mod, "run", spy)
    yield store
    mp.undo()


@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("a14_damage") / "store", mode="inprocess")


_TIMES = {}


def run_damage(rhub, captured, engine, p):
    """(hub result, ImplantResult) for a damage problem; one engine run per problem in this module."""
    _need(engine)
    t = time.time()
    r = rhub.submit(ADAPTERS[engine], params=_params(p))
    _TIMES.setdefault((engine, p.problem_id()), time.time() - t)
    assert r.status == "ok", (engine, r.status, r.message)
    res = captured.get((engine, p.problem_id()))
    assert res is not None, "engine result was not captured (cache hit from another module?)"
    N = p.statistics.n_ions
    assert len(res.depth_um) + res.n_reflected + res.n_transmitted == N
    return r, res


def _vpi(r):
    q = r.q("implant.vacancies_per_ion")
    return float(q.value), float(q.uncertainty.sigma)


# ------------------------------------------------------------------------------------------------ A14.2 description
def test_description():
    import pydantic
    from simws.spec import Atom
    a = Atom(symbol="Si", Z=14, mass_amu=28.085, cutoff_eV=1.5)
    assert a.displacement_eV is None
    assert "displacement_eV" not in a.model_dump(mode="json")
    assert Atom(symbol="Si", Z=14, mass_amu=28.085, cutoff_eV=1.5, displacement_eV=15.0).displacement_eV == 15.0
    for bad in (0.0, -1.0):
        with pytest.raises(pydantic.ValidationError):
            Atom(symbol="Si", Z=14, mass_amu=28.085, cutoff_eV=1.5, displacement_eV=bad)
    p = _implant("P", 50, displacement_eV=15.0)
    si = [x for x in p.target.atoms if x.symbol == "Si"][0]
    assert si.displacement_eV == 15.0 and p.models.recoils is True
    q = _implant("P", 50)
    assert q.models.recoils is False and all(x.displacement_eV is None for x in q.target.atoms)
    assert "displacement_eV" not in json.dumps(q.model_dump(mode="json"))


def test_R75_ids_unchanged():
    assert _implant("P", 50).problem_id() == "5cfbc9d5e496882805061234f5349fe1"
    assert _implant("B", 100).problem_id() == "7141925e1068b3eda07be7f312f0690d"
    assert _implant("P", 50, displacement_eV=15.0).problem_id() != _implant("P", 50).problem_id()


def _a74_input(problem, threads=2):
    """RustBCA input written from A7.4 (default mode, no recoils), not from build_input."""
    b, t = problem.beam, problem.target
    th = math.radians(b.tilt_deg)
    syms = [a.symbol for a in t.atoms]
    pot = {"zbl": "ZBL", "kr_c": "KR_C", "moliere": "MOLIERE"}[problem.models.potential]
    return {
        "options": {"name": "checker", "track_recoils": False, "electronic_stopping_mode": "LOW_ENERGY_NONLOCAL",
                    "interaction_potential": [[pot]], "scattering_integral": [["MENDENHALL_WELLER"]],
                    "mean_free_path_model": "LIQUID", "seed": problem.statistics.seed, "num_chunks": 1,
                    "num_threads": threads},
        "particle_parameters": {"length_unit": "MICRON", "energy_unit": "EV", "mass_unit": "AMU",
                                "N": [problem.statistics.n_ions], "Z": [b.ion.Z], "m": [b.ion.mass_amu],
                                "Ec": [b.ion.cutoff_eV], "Es": [b.ion.surface_binding_eV],
                                "E": [b.energy_keV * 1000.0], "pos": [[0.0, 0.0, 0.0]],
                                "dir": [[math.cos(th), math.sin(th), 0.0]]},
        "geometry_input": {"length_unit": "MICRON", "layer_thicknesses": [l.thickness_um for l in t.layers],
                           "densities": [[l.density_cm3.get(s, 0.0) * 1e-12 for s in syms] for l in t.layers],
                           "electronic_stopping_correction_factors": [1.0 for _ in t.layers]},
        "material_parameters": {"energy_unit": "EV", "mass_unit": "AMU", "Z": [a.Z for a in t.atoms],
                                "m": [a.mass_amu for a in t.atoms], "Ec": [a.cutoff_eV for a in t.atoms],
                                "Es": [a.surface_binding_eV for a in t.atoms],
                                "Eb": [a.bulk_binding_eV for a in t.atoms]},
    }


def test_R75_range_only_unchanged(tmp_path):
    """The range-only RustBCA result (in process) is the one of the A7.4 input: depth_um identical."""
    lib = pytest.importorskip("libRustBCA")
    p = _implant("P", 50, n_ions=2000)
    res = _mod("rustbca").run(p, tmp_path)
    out = lib.rustbca_local_py(_a74_input(p), "1D")
    ref = [x for x, imp in zip(out["x"], out["implanted"]) if imp]
    assert res.depth_um == ref
    assert DAMAGE_FLAG not in res.flags


# ------------------------------------------------------------------------------------------------ R76 refusals
def _two_species(Ed_si=15.0, Ed_o=20.0):
    from simws.spec import ImplantProblem
    d = _implant("P", 50, n_ions=2000, displacement_eV=Ed_si).model_dump(mode="json")
    o = {"symbol": "O", "Z": 8, "mass_amu": 15.999, "cutoff_eV": 2.0, "surface_binding_eV": 2.0}
    if Ed_o is not None:
        o["displacement_eV"] = Ed_o
    d["target"]["atoms"].append(o)
    d["target"]["layers"] = [{"name": "oxide", "thickness_um": 0.02, "density_cm3": {"Si": 2.3e22, "O": 4.6e22}},
                             {"name": "si", "thickness_um": 1.98, "density_cm3": {"Si": 4.994e22}}]
    return ImplantProblem.model_validate(d)


def _reasons(engine, p):
    from simws import engines
    r = engines.unsupported(engine, p)
    print(f"\n[R76 {engine}] {r}")
    return r


@pytest.mark.parametrize("engine", ["rustbca", "g4implant"])
def test_R76_recoils_need_displacement(engine):
    r = _reasons(engine, _implant("P", 50).with_changes({"models.recoils": True}))
    assert r and any(s.startswith("recoils:") for s in r), r
    assert _reasons(engine, _dmg("P")) == []                         # a valid damage problem is supported


def test_R76_rustbca():
    r = _reasons("rustbca", _dmg("B").with_changes({"models.electronic_stopping": "empirical"}))
    assert r and any(s.startswith("electronic_stopping:") for s in r), r
    r = _reasons("rustbca", _dmg("P", Ed=1.0))                       # Ed 1.0 below the silicon cutoff 1.5 eV
    assert r and any(s.startswith("recoils:") for s in r), r
    r = _reasons("rustbca", _two_species(Ed_o=None))                 # O used in a layer, no displacement_eV
    assert r and any(s.startswith("recoils:") for s in r), r
    assert _reasons("rustbca", _two_species()) == []                 # two species are fine for RustBCA
    r = _reasons("rustbca", _dmg("P").with_changes({"statistics.seed": 2_000_000}))
    assert r, "A14.3: the seed must be below 2 000 000 (no reason prefix is fixed by the contract)"
    assert _reasons("rustbca", _dmg("P").with_changes({"statistics.seed": 1_999_999})) == []


def test_R76_g4implant():
    r = _reasons("g4implant", _two_species())
    assert r and any(s.startswith("recoils:") for s in r), r
    r = _reasons("g4implant", _dmg("P", Ed=9.9))
    assert r and any(s.startswith("recoils:") for s in r), r
    assert _reasons("g4implant", _dmg("P", Ed=10.0)) == []
    r = _reasons("g4implant", _dmg("P").with_changes({"target.atoms": [
        dict(_dmg("P").target.atoms[0].model_dump(mode="json"), bulk_binding_eV=0.5)]}))
    assert r and any(s.startswith("recoils:") for s in r), r
    # "empirical" with recoils is refused only by RustBCA (A14.3); Geant4 states no such rule
    print(f"[R76 g4implant] empirical + recoils: "
          f"{_reasons('g4implant', _dmg('B').with_changes({'models.electronic_stopping': 'empirical'}))}")


def test_R76_through_hub(hub):
    for eng, p, pre in (("rustbca", _dmg("P", Ed=1.0), "recoils:"),
                        ("g4implant", _dmg("P", Ed=9.9), "recoils:"),
                        ("rustbca", _dmg("B").with_changes({"models.electronic_stopping": "empirical"}),
                         "electronic_stopping:")):
        r = hub.submit(ADAPTERS[eng], params=_params(p))
        assert r.status == "invalid_input" and pre in r.message, (eng, r.status, r.message)


# ------------------------------------------------------------------------------------------------ engine output, adapter
def _check_engine_output(p, res, engine):
    N = p.statistics.n_ions
    nb = math.ceil(p.target.thickness_um / p.depth_bin_um)
    assert res.vacancy_bin_um == p.depth_bin_um
    assert len(res.vacancy_batches) == BATCHES and all(len(b) == nb for b in res.vacancy_batches)
    ions = list(res.vacancy_batch_ions)
    assert len(ions) == BATCHES and sum(ions) == N and max(ions) - min(ions) <= 1, ions
    if engine == "g4implant":                                         # event i in batch floor(20 i / N)
        exp = np.bincount([(BATCHES * i) // N for i in range(N)], minlength=BATCHES)
        assert ions == exp.tolist()
    assert "vacancies_outside" in res.diagnostics
    assert all(c >= 0 for b in res.vacancy_batches for c in b)


def expected_damage(p, res):
    """A14.4 from the batch data: (value, sigma) of vacancies per ion, profile, mean depth (pooled and per-batch)."""
    N = p.statistics.n_ions
    C = np.asarray(res.vacancy_batches, float)                       # B x nbins
    n = np.asarray(res.vacancy_batch_ions, float)
    b_m = p.depth_bin_um * UM
    centres = (np.arange(C.shape[1]) + 0.5) * b_m
    se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v)))      # standard error over the batches
    vpi_b = C.sum(1) / n
    prof_b = C / (n[:, None] * b_m)
    dm_b = (C * centres).sum(1) / C.sum(1)
    tot = C.sum(0)
    return {"vpi": (tot.sum() / N, se(vpi_b)),
            "profile": (tot / (N * b_m), np.std(prof_b, axis=0, ddof=1) / math.sqrt(C.shape[0]), centres),
            "depth_pooled": float((tot * centres).sum() / tot.sum()), "depth_batch_mean": float(dm_b.mean()),
            "depth_se": se(dm_b)}


def _check_adapter(r, p, res):
    e = expected_damage(p, res)
    assert DAMAGE_FLAG in r.flags, r.flags
    q = r.q("implant.vacancies_per_ion")
    assert q.unit == "dimensionless" and q.axes == []
    assert q.value == pytest.approx(e["vpi"][0], rel=1e-9)
    assert q.uncertainty.sigma == pytest.approx(e["vpi"][1], rel=1e-9)
    assert q.uncertainty.components.get("statistical") == pytest.approx(q.uncertainty.sigma, rel=1e-12)
    pr = r.q("implant.vacancy_profile")
    assert pr.unit == "1/m" and len(pr.axes) == 1 and pr.axes[0].name == "depth" and pr.axes[0].unit == "m"
    assert np.allclose(pr.axes[0].values, e["profile"][2], rtol=1e-12, atol=0)
    assert np.allclose(pr.value, e["profile"][0], rtol=1e-9, atol=0)
    sig = np.asarray(pr.uncertainty.sigma, float)
    assert np.allclose(sig, e["profile"][1], rtol=1e-9, atol=1e-30 * 0 + 1e-300)
    dm = r.q("implant.vacancy_depth_mean")
    assert dm.unit == "m"
    which = ("pooled" if dm.value == pytest.approx(e["depth_pooled"], rel=1e-9) else
             "mean of batch means" if dm.value == pytest.approx(e["depth_batch_mean"], rel=1e-9) else None)
    print(f"\n[A14.4] vacancy_depth_mean {dm.value * 1e9:.3f} nm: pooled {e['depth_pooled'] * 1e9:.3f}, mean of batch "
          f"means {e['depth_batch_mean'] * 1e9:.3f} -> adapter uses: {which}; sigma {dm.uncertainty.sigma * 1e9:.4f} "
          f"vs batch SE {e['depth_se'] * 1e9:.4f} nm")
    assert which is not None
    assert dm.uncertainty.sigma == pytest.approx(e["depth_se"], rel=1e-9)
    return e


# ------------------------------------------------------------------------------------------------ R70
def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(str(k))} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def to_toml(d):
    """The checker's own TOML writer: one table per top-level key, inline values below."""
    lines = []
    for sec, body in d.items():
        lines.append(f"[{sec}]")
        for k, v in body.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    return "\n".join(lines)


def _batch_input(p, n_b, b, threads=2):
    inp = json.loads(json.dumps(_mod("rustbca").build_input(p), default=float))
    o = inp["options"]
    o.update({"name": "chk_", "track_recoils": True, "track_displacements": True,
              "seed": p.statistics.seed * 1000 + b, "num_threads": threads,
              "num_chunks": max(1, math.ceil(n_b / (2 * threads)))})
    inp["particle_parameters"]["N"] = [n_b]
    inp["material_parameters"]["Ed"] = [a.displacement_eV for a in p.target.atoms]
    return inp


def _read_batch_directly(p, n_b, b, folder):
    exe = os.environ.get("SIMWS_RUSTBCA_BIN") or _mod("rustbca").cli()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.toml").write_text(to_toml(_batch_input(p, n_b, b)))
    cp = subprocess.run([exe, "1D", "input.toml"], cwd=folder, capture_output=True, text=True, timeout=600)
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    rows = np.loadtxt(folder / "chk_displacements.output", delimiter=",", ndmin=2)
    return rows                                                       # m, Z, E, x0, y0, z0, x, y, z


def _hist_start(p, x0):
    """Start depths in the target on the problem's bins (an edge goes to the deeper bin; the last bin takes its
    end), as A7.9 states for the range profile."""
    T, bw = p.target.thickness_um, p.depth_bin_um
    nb = math.ceil(T / bw)
    idx = np.minimum(np.floor(x0 / bw).astype(int), nb - 1)
    return np.bincount(idx, minlength=nb)


@pytest.mark.slow
def test_R70_rustbca_one_batch_read_directly(rhub, captured, tmp_path):
    _need("rustbca")
    p = _dmg("P")
    r, res = run_damage(rhub, captured, "rustbca", p)
    _check_engine_output(p, res, "rustbca")
    j = 0
    n_b = res.vacancy_batch_ions[j]
    engine_count = int(sum(res.vacancy_batches[j]))
    T = p.target.thickness_um
    found = {}
    for b in (j, j + 1):                                             # A14.3 does not say whether b counts from 0 or 1
        rows = _read_batch_directly(p, n_b, b, tmp_path / f"seed{b}")
        x0 = rows[:, 3]
        inside = (x0 >= 0.0) & (x0 <= T)
        found[b] = (int(inside.sum()), int((~inside).sum()), _hist_start(p, x0[inside]))
        print(f"\n[R70 rustbca] batch {j} ({n_b} ions) with seed {p.statistics.seed * 1000 + b}: displacement rows "
              f"{len(rows)}, start in target {found[b][0]}, outside {found[b][1]}; engine batch count {engine_count}")
    match = [b for b, f in found.items() if f[0] == engine_count]
    assert match, (engine_count, {b: f[0] for b, f in found.items()})
    b = match[0]
    print(f"[R70 rustbca] the engine's batch {j} is the run with seed seed*1000+{b} (batches counted from {b - j})")
    assert found[b][2].tolist() == [int(c) for c in res.vacancy_batches[j]]


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["rustbca", "g4implant"])
def test_R70_counting_and_adapter(rhub, captured, engine):
    p = _dmg("P")
    r, res = run_damage(rhub, captured, engine, p)
    _check_engine_output(p, res, engine)
    N = p.statistics.n_ions
    total = float(np.sum(res.vacancy_batches))
    vpi = r.q("implant.vacancies_per_ion").value
    prof = np.asarray(r.q("implant.vacancy_profile").value, float)
    print(f"\n[R70 {engine}] P 50 keV Ed 15: total vacancies {total:.0f}, vacancies/ion x N {vpi * N:.6f}, profile sum "
          f"x bin x N {prof.sum() * p.depth_bin_um * UM * N:.6f}; outside {res.diagnostics.get('vacancies_outside')}; "
          f"run time {_TIMES.get((engine, p.problem_id()), float('nan')):.1f} s")
    assert vpi * N == pytest.approx(total, rel=1e-12)
    assert prof.sum() * p.depth_bin_um * UM * N == pytest.approx(total, rel=1e-9)
    _check_adapter(r, p, res)
    print(f"[R70 {engine}] flags: {r.flags}")


def test_dose_gives_vacancy_concentration(hub, captured):
    """A14.4: with a dose, implant.vacancy_concentration (1/m^3) = dose (m^-2) x vacancy profile (as A7.9)."""
    _need("rustbca")
    p = _dmg("P", n_ions=200, dose_cm2=1e14)
    r = hub.submit(ADAPTERS["rustbca"], params=_params(p))
    assert r.status == "ok", r.message
    c, pr = r.q("implant.vacancy_concentration"), r.q("implant.vacancy_profile")
    assert c.unit in ("1/m**3", "1/m^3", "m**-3")
    assert np.allclose(c.value, 1e14 * 1e4 * np.asarray(pr.value), rtol=1e-12, atol=0)
    assert np.allclose(c.uncertainty.sigma, 1e14 * 1e4 * np.asarray(pr.uncertainty.sigma), rtol=1e-12, atol=0)
    r0 = hub.submit(ADAPTERS["rustbca"], params=_params(_dmg("P", n_ions=200)))
    assert "implant.vacancy_concentration" not in [q.name for q in r0.quantities]
    rr = hub.submit(ADAPTERS["rustbca"], params=_params(_implant("P", 50, n_ions=200)))
    assert not any(q.name.startswith("implant.vacanc") for q in rr.quantities) and DAMAGE_FLAG not in rr.flags


# ------------------------------------------------------------------------------------------------ R71, R72
# Marked after the run (section 2); strict, so a change is noticed.
_R71_FAIL = {"B": "recorded FAIL A14 R71 g4implant B 50 keV: vacancies/ion Ed 15 604.28, Ed 30 438.52, ratio 1.378 (rerun after the A13.7 fix) "
                  "outside [1.6, 2.4]",
             "P": "recorded FAIL A14 R71 g4implant P 50 keV: vacancies/ion Ed 15 1070.17, Ed 30 782.99, ratio 1.367 (rerun after the A13.7 fix) "
                  "outside [1.6, 2.4]"}


@pytest.mark.slow
@pytest.mark.parametrize("engine,ion", [
    pytest.param(e, i, id=f"{i}-{e}", marks=[pytest.mark.xfail(strict=True, reason=_R71_FAIL[i])] if e == "g4implant"
                 else []) for i in ("B", "P") for e in ("rustbca", "g4implant")])
def test_R71_ed_scaling(rhub, captured, engine, ion):
    r15, _ = run_damage(rhub, captured, engine, _dmg(ion, Ed=15.0))
    r30, _ = run_damage(rhub, captured, engine, _dmg(ion, Ed=30.0))
    (a, sa), (b, sb) = _vpi(r15), _vpi(r30)
    ratio = a / b
    print(f"\n[R71 {engine} {ion} 50 keV] vacancies/ion Ed 15: {a:.2f} +- {sa:.2f}, Ed 30: {b:.2f} +- {sb:.2f}; "
          f"ratio {ratio:.4f} (band [1.6, 2.4])")
    assert 1.6 <= ratio <= 2.4


def n_nrt(Z1, A1, E_eV, Ed, Z2=14, A2=28.085):
    """A14.5 R72: NRT with the ion as the primary."""
    E_L = 30.724 * Z1 * Z2 * math.sqrt(Z1 ** (2 / 3) + Z2 ** (2 / 3)) * (A1 + A2) / A2
    eps = E_eV / E_L
    k = 0.1337 * Z1 ** (1 / 6) * math.sqrt(Z1 / A1)
    g = 3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps
    T_dam = E_eV / (1 + k * g)
    return 0.8 * T_dam / (2 * Ed)


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["rustbca", "g4implant"])
@pytest.mark.parametrize("ion", ["B", "P"])
def test_R72_nrt_band(rhub, captured, engine, ion):
    p = _dmg(ion)
    r, _ = run_damage(rhub, captured, engine, p)
    a = p.beam.ion
    nrt = n_nrt(a.Z, a.mass_amu, p.beam.energy_keV * 1e3, 15.0)
    v, s = _vpi(r)
    print(f"\n[R72 {engine} {ion} 50 keV] vacancies/ion {v:.2f} +- {s:.2f}; N_NRT {nrt:.2f}; ratio {v / nrt:.4f} "
          f"(band [0.8, 3.0]); mean depth {r.q('implant.vacancy_depth_mean').value * 1e9:.2f} nm")
    assert 0.8 <= v / nrt <= 3.0


# ------------------------------------------------------------------------------------------------ R73
@pytest.mark.slow
@pytest.mark.parametrize("ion", ["B", pytest.param("P", marks=[pytest.mark.xfail(
    strict=True, reason="recorded FAIL A14 R73 P 50 keV: vacancy_depth_mean Geant4 60.27 nm against RustBCA 52.30 nm "
    "(+15.2 %, gate 15 %, rerun after the A13.7 fix); vacancies/ion -13.9 % within 20 %")])])
def test_R73_cross_code(rhub, captured, ion):
    from simws import compare
    p = _dmg(ion)
    rb, _ = run_damage(rhub, captured, "rustbca", p)
    g4, _ = run_damage(rhub, captured, "g4implant", p)
    ok = {}
    for name, tol in (("implant.vacancies_per_ion", 0.20), ("implant.vacancy_depth_mean", 0.15)):
        c = compare.compare_results(rhub, g4.id, name, rb.id, tolerance=tol, metric="relative")
        compare.record(rhub, c)
        a, b = g4.q(name), rb.q(name)
        print(f"\n[R73 {ion} 50 keV] {name}: Geant4 {a.value:.5g} +- {a.uncertainty.sigma:.2g}, RustBCA {b.value:.5g} +- "
              f"{b.uncertainty.sigma:.2g}; Geant4/RustBCA - 1 = {a.value / b.value - 1:+.4f} (tolerance {tol}); "
              f"passed={c.passed} independent={c.independent}")
        assert c.independent is True
        ok[name] = c.passed
    assert all(ok.values()), ok


# ------------------------------------------------------------------------------------------------ R74
@pytest.mark.slow
def test_R74_memory(tmp_path):
    _need("rustbca")
    env = dict(os.environ)
    env.pop("SIMWS_THREADS", None)                                   # the default thread count
    code = textwrap.dedent(f"""
        import json, resource, time
        from simws import engines
        from simws.spec.library import implant_si
        p = implant_si("P", 50.0, n_ions=2000, displacement_eV=15.0)
        t = time.time()
        res = engines.module("rustbca").run(p, {str(tmp_path / 'w')!r})
        dt = time.time() - t
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        print(json.dumps({{"maxrss_kB": ru.ru_maxrss, "seconds": dt, "vac": float(sum(map(sum, res.vacancy_batches)))}}))
    """)
    cp = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1800,
                        cwd=str(Path(__file__).resolve().parents[2]))
    assert cp.returncode == 0, cp.stderr[-3000:]
    d = json.loads(cp.stdout.strip().splitlines()[-1])
    peak = d["maxrss_kB"] * 1024
    print(f"\n[R74] RustBCA damage run P 50 keV, 2000 ions, default threads: peak child RSS {peak / 1e6:.1f} MB "
          f"({peak / 2 ** 20:.1f} MiB), run time {d['seconds']:.1f} s, vacancies {d['vac']:.0f}")
    assert peak < 1.5e9
