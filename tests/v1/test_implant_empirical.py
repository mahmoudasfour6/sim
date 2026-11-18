"""Contract amendment A12: electronic stopping from measured data for RustBCA ("empirical"). Acceptance checks R16e,
R16s (reported), R17, R18 and R19, plus the description, translator, flag and capability rules of A12.4 / A12.5.
Written from docs/ARCHITECTURE_V1.md (sections 2, A7, A12) only; the implementation files were not read.

R17 recomputes the factor from simws/data/stopping/iaea_<ion>_si.csv with the rule of A12.2, calling RustBCA's own
`electronic_stopping_cross_sections` (A12.2b) and nothing from simws.radiation.empirical_stopping, and only at the
end compares with `entry(ion)`. R16e / R16s use the R16 table of A7.7 and its 25 % gate, unchanged."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

ADAPTER = "implant.rustbca"
UM = 1e-6
DATA = Path(__file__).resolve().parents[2] / "simws" / "data" / "stopping"
AMU_KG = 1.66053906660e-27             # RustBCA 3.0.0 consts.rs AMU (CODATA 2018), used for the u -> mg conversion
SI_MASS_U = 28.0855                    # A12.2(b)
E_MAX_MEV_U = 0.010                    # A12.2(a)
ISOTOPE = {"B": (5, 11.009305), "P": (15, 30.973762), "As": (33, 74.921595)}   # A7.2 table (Z, isotope mass)


def _implant(*a, **k):
    from simws.spec.library import implant_si
    return implant_si(*a, **k)


def _emp(p):
    return p.with_changes({"models.electronic_stopping": "empirical"})


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _ls_eV_m2(Z1, E_eV, M1):
    """A12.2(b): RustBCA 3.0.0's own electronic_stopping_cross_sections(Z1, 14, E, M1, 1.0, 1.0)[0], eV m^2."""
    lib = pytest.importorskip("libRustBCA")
    return lib.electronic_stopping_cross_sections(float(Z1), 14.0, float(E_eV), float(M1), 1.0, 1.0)[0]


def _ls_MeV_cm2_mg(Z1, E_eV, M1):
    """eV m^2 per atom -> MeV/(mg/cm^2) with the silicon atom mass 28.0855 u (A12.2b)."""
    m_mg = SI_MASS_U * AMU_KG * 1e6                                   # kg -> mg
    return _ls_eV_m2(Z1, E_eV, M1) * 1e4 / m_mg * 1e-6                # m^2 -> cm^2, eV -> MeV


# ------------------------------------------------------------------------------------------------ A12.2 by hand
def _rows_passing(ion):
    """A12.2(a) applied to the CSV copied from the IAEA database."""
    out = []
    with open(DATA / f"iaea_{ion.lower()}_si.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["projectile_name"] != ion or row["target_name"] != "Si" or row["phase"] != "s":
                continue
            if row["energy_unit"] != "MeV/u" or float(row["energy"]) > E_MAX_MEV_U:
                continue
            if row["ref_id"] == "SRIM2013":
                continue
            assert row["stopping_unit_converted"] == "MeV/(mg/cm2)", row
            out.append(row)
    return out


def _ck(rs, es):
    rs, es = np.asarray(rs, float), np.asarray(es, float)
    w = 1.0 / (rs * es) ** 2
    ck = float(np.sum(w * rs) / np.sum(w))
    chi2 = float(np.sum(w * (rs - ck) ** 2))
    n = rs.size
    birge = math.sqrt(chi2 / (n - 1)) if n > 1 else float("nan")
    u = math.sqrt(1.0 / np.sum(w)) * max(1.0, birge)
    return ck, u, chi2, birge


def rule(ion):
    """A12.2 (a)-(e) from the CSV text alone. None when (d) gives no entry."""
    Z1 = ISOTOPE[ion][0]
    rows = _rows_passing(ion)
    pubs = sorted({r["ref_id"] for r in rows})
    if len(rows) < 3 or len(pubs) < 2:
        return None
    r, e, eku, ref = [], [], [], []
    for row in rows:
        M1 = float(row["ion_isotope"])
        E_eV = float(row["energy"]) * M1 * 1e6
        r.append(float(row["stopping_power_converted"]) / _ls_MeV_cm2_mg(Z1, E_eV, M1))
        pe = row["percentage_error"].strip()
        e.append(float(pe) / 100.0 if pe else 0.10)
        eku.append(float(row["energy"]) * 1000.0)                    # E / M1 in keV/u
        ref.append(row["ref_id"])
    ck, u, chi2, birge = _ck(r, e)
    loo = {}
    for p_ in pubs:
        keep = [i for i in range(len(r)) if ref[i] != p_]
        loo[p_] = _ck([r[i] for i in keep], [e[i] for i in keep])[0]
    slope = float(np.polyfit(np.asarray(eku), np.asarray(r), 1)[0])
    return {"ck": ck, "u": u, "n": len(rows), "publications": pubs, "chi2": chi2, "birge": birge, "loo": loo,
            "slope": slope, "r": r, "E_keV_u": eku, "rows_per_pub": {p_: ref.count(p_) for p_ in pubs}}


# ------------------------------------------------------------------------------------------------ R17
def test_R17_rule_repeated():
    pytest.importorskip("libRustBCA")
    from simws.radiation import empirical_stopping as es
    mine = rule("B")
    assert mine is not None
    print(f"\n[R17 B] recomputed: ck={mine['ck']:.12f} u={mine['u']:.12f} n={mine['n']} pubs={mine['publications']} "
          f"rows/pub={mine['rows_per_pub']} chi2={mine['chi2']:.4f} Birge={mine['birge']:.4f} "
          f"E range {min(mine['E_keV_u']):.3f}-{max(mine['E_keV_u']):.3f} keV/u")
    print(f"[R17 B] leave one publication out: { {k: round(v, 6) for k, v in mine['loo'].items()} }, "
          f"slope={mine['slope']:.6f} per keV/u")
    # the numbers A12.3 states, to the decimals it states (a consistency check on the text)
    assert round(mine["ck"], 3) == 1.381 and round(mine["u"], 3) == 0.195
    assert mine["n"] == 11 and len(mine["publications"]) == 4
    assert round(mine["chi2"]) == 1190 and round(mine["birge"], 1) == 10.9
    assert round(min(mine["E_keV_u"]), 2) == 0.40 and round(max(mine["E_keV_u"]), 2) == 9.09
    assert sorted(round(v, 3) for v in mine["loo"].values()) == sorted([1.761, 1.373, 1.380, 1.072])
    assert round(mine["slope"], 3) == -0.073
    # now, and only now, the module's entry
    ent = es.entry("B")
    assert ent is not None
    print(f"[R17 B] entry('B'): ck={ent['ck']:.12f} u={ent['u']:.12f} n={ent['n']} pubs={ent['publications']}")
    rel_ck = abs(ent["ck"] / mine["ck"] - 1)
    rel_u = abs(ent["u"] / mine["u"] - 1)
    print(f"[R17 B] relative differences: ck {rel_ck:.3e}, u {rel_u:.3e}")
    assert rel_ck <= 1e-9 and rel_u <= 1e-9
    assert ent["n"] == mine["n"]
    assert sorted(ent["publications"]) == mine["publications"]
    assert es.entry("P") is None and es.entry("As") is None
    # the rule itself gives no entry for P and As (A12.3: P no row, As one publication)
    assert rule("P") is None and rule("As") is None
    print(f"[R17] rows passing (a): P {len(_rows_passing('P'))}, As {len(_rows_passing('As'))} from "
          f"{sorted({r['ref_id'] for r in _rows_passing('As')})}")
    assert len(_rows_passing("P")) == 0
    assert len(_rows_passing("As")) == 6 and len({r["ref_id"] for r in _rows_passing("As")}) == 1


def test_R17_reported_items_of_entry():
    """A12.2(e) is reported with the entry: leave-one-publication-out values and the slope (1e-9 as R17)."""
    pytest.importorskip("libRustBCA")
    from simws.radiation import empirical_stopping as es
    mine, ent = rule("B"), es.entry("B")
    loo = ent["leave_one_out"]
    assert sorted(loo) == mine["publications"]
    for k, v in mine["loo"].items():
        assert loo[k] == pytest.approx(v, rel=1e-9), k
    assert ent["slope_per_keV_u"] == pytest.approx(mine["slope"], rel=1e-9)
    assert ent["birge_ratio"] == pytest.approx(mine["birge"], rel=1e-9)
    assert len(es.selected("B")) == mine["n"]
    assert es.selected("P") == [] or len(es.selected("P")) == 0


# ------------------------------------------------------------------------------------------------ R18
@pytest.mark.parametrize("ion", ["B", "P", "As"])
@pytest.mark.parametrize("E_keV", [1.0, 10.0, 100.0])
def test_R18_ls_formula(ion, E_keV):
    lib = pytest.importorskip("libRustBCA")
    from simws.radiation import empirical_stopping as es
    Z1, M1 = ISOTOPE[ion]
    ref = lib.electronic_stopping_cross_sections(float(Z1), 14.0, E_keV * 1e3, M1, 1.0, 1.0)[0]
    got = es.lindhard_scharff_cross_section_eV_m2(Z1, 14, E_keV * 1e3, M1)
    assert ref > 0
    assert abs(got / ref - 1) <= 1e-12, (got, ref)


@pytest.mark.parametrize("ion", ["B", "P", "As"])
def test_ls_mass_stopping_conversion(ion):
    """lindhard_scharff_MeV_cm2_mg is the eV m^2 value converted with 28.0855 u (A12.2b) (the
    value of the atomic mass unit is not fixed by the contract; CODATA 2018 and 2022 differ by 1.4e-9, so 1e-8)."""
    pytest.importorskip("libRustBCA")
    from simws.radiation import empirical_stopping as es
    Z1, M1 = ISOTOPE[ion]
    for E in (1e3, 1e4, 1e5):
        assert es.lindhard_scharff_MeV_cm2_mg(Z1, M1, E) == pytest.approx(_ls_MeV_cm2_mg(Z1, E, M1), rel=1e-8)


# ------------------------------------------------------------------------------------------------ A12.4 description
def test_A124_ids_unchanged_and_literal():
    import pydantic
    assert _implant("P", 50).problem_id() == "5cfbc9d5e496882805061234f5349fe1"
    assert _implant("B", 100).problem_id() == "7141925e1068b3eda07be7f312f0690d"
    assert _implant("P", 50).models.electronic_stopping == "lindhard_scharff"
    p = _emp(_implant("B", 50))
    assert p.models.electronic_stopping == "empirical"
    assert p.problem_id() != _implant("B", 50).problem_id()
    with pytest.raises(pydantic.ValidationError):
        _implant("B", 50).with_changes({"models.electronic_stopping": "srim"})


# ------------------------------------------------------------------------------------------------ A12.5 translator
def test_build_input_factors():
    """Default mode: every correction factor 1.0 (R19); empirical: every layer's factor is the entry's ck, mode
    LOW_ENERGY_NONLOCAL as before (A12.5)."""
    from simws import engines
    from simws.radiation import empirical_stopping as es
    from simws.spec import ImplantProblem
    mod = engines.module("rustbca")
    for p in (_implant("B", 50), _implant("P", 50), _implant("As", 50)):
        g = mod.build_input(p)["geometry_input"]
        assert np.all(np.asarray(g["electronic_stopping_correction_factors"], float) == 1.0)
    d = _implant("B", 50).model_dump(mode="json")
    lay = d["target"]["layers"][0]
    d["target"]["layers"] = [dict(lay, name="top", thickness_um=0.03), dict(lay, name="bulk", thickness_um=1.97)]
    two = _emp(ImplantProblem.model_validate(d))
    ck = es.entry("B")["ck"]
    for p in (_emp(_implant("B", 50)), two):
        inp = mod.build_input(p)
        f = np.asarray(inp["geometry_input"]["electronic_stopping_correction_factors"], float).ravel()
        assert f.size >= 1
        assert np.all(f == ck), f
        assert inp["options"]["electronic_stopping_mode"] == "LOW_ENERGY_NONLOCAL"


def _oxide_on_si(ion="B", E=50.0):
    from simws.spec import ImplantProblem
    d = _implant(ion, E).model_dump(mode="json")
    d["target"]["atoms"].append({"symbol": "O", "Z": 8, "mass_amu": 15.999, "cutoff_eV": 2.0,
                                 "surface_binding_eV": 2.0})
    d["target"]["layers"] = [{"name": "oxide", "thickness_um": 0.02, "density_cm3": {"Si": 2.3e22, "O": 4.6e22}},
                             {"name": "si", "thickness_um": 1.98, "density_cm3": {"Si": 4.994e22}}]
    return ImplantProblem.model_validate(d)


def test_R19_refusals(tmp_path):
    from simws import engines
    from simws.engines.base import EngineNotAvailable, Unsupported
    mod = engines.module("rustbca")
    assert engines.unsupported("rustbca", _emp(_implant("B", 50))) == []
    assert engines.unsupported("rustbca", _emp(_implant("B", 100))) == []
    cases = {"P": _emp(_implant("P", 50)), "As": _emp(_implant("As", 50)), "B+oxide": _emp(_oxide_on_si())}
    assert engines.unsupported("rustbca", _oxide_on_si()) == []          # the oxide is fine without "empirical"
    for name, p in cases.items():
        r = engines.unsupported("rustbca", p)
        print(f"\n[R19 {name}] {r}")
        assert r and all(s.startswith("electronic_stopping:") for s in r), (name, r)
        try:
            with pytest.raises(Unsupported):
                mod.run(p.with_changes({"statistics.n_ions": 100}), tmp_path / name)
        except EngineNotAvailable:
            pytest.fail("EngineNotAvailable raised for an unsupported problem")


def test_R19_refusals_through_hub(hub):
    for p in (_emp(_implant("P", 50, n_ions=100)), _emp(_implant("As", 50, n_ions=100)),
              _emp(_oxide_on_si()).with_changes({"statistics.n_ions": 100})):
        r = hub.submit(ADAPTER, params=_params(p))
        assert r.status == "invalid_input", (r.status, r.message)
        assert "unsupported by rustbca: " in r.message and "electronic_stopping:" in r.message, r.message


def _a74_input(problem, threads=2):
    """The RustBCA input dictionary written from the text of A7.4 (not from build_input), default mode."""
    b, t = problem.beam, problem.target
    pot = {"zbl": "ZBL", "kr_c": "KR_C", "moliere": "MOLIERE"}[problem.models.potential]
    th = math.radians(b.tilt_deg)
    syms = [a.symbol for a in t.atoms]
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


def direct_depths(problem):
    """Depths (um) of the implanted ions, in launch order, from libRustBCA called with the A7.4 dictionary."""
    lib = pytest.importorskip("libRustBCA")
    out = lib.rustbca_local_py(_a74_input(problem), "1D")
    return [x for x, imp in zip(out["x"], out["implanted"]) if imp]


def test_R19_default_mode_unchanged(tmp_path):
    """The default mode is the pre-A12 run: same seed and input dictionary (A7.4 written by the checker), so the
    same depth_um. Boron is used because it is the ion A12 changes."""
    pytest.importorskip("libRustBCA")
    from simws import engines
    for p in (_implant("B", 50, n_ions=2000), _implant("P", 50, n_ions=2000)):
        res = engines.module("rustbca").run(p, tmp_path / p.beam.ion.symbol)
        ref = direct_depths(p)
        assert len(res.depth_um) == len(ref) and res.depth_um == ref
        assert not any(f.startswith("implant:empirical_stopping:") for f in res.flags), res.flags


# ------------------------------------------------------------------------------------------------ A12.5 flag, ck override
def _flag_text(ent):
    return (f"implant:empirical_stopping: Lindhard-Scharff x {ent['ck']:.3f} +- {ent['u']:.3f} from {ent['n']} "
            f"measured points (IAEA database, A12); the data sets disagree (Birge ratio {ent['birge']:.1f})")


def test_empirical_run_flag_and_diagnostics(hub, tmp_path):
    pytest.importorskip("libRustBCA")
    from simws import engines
    mine = rule("B")
    p = _emp(_implant("B", 50, n_ions=2000))
    res = engines.module("rustbca").run(p, tmp_path / "e")
    flag = _flag_text(mine)
    print(f"\n[A12.5] expected flag: {flag!r}\n[A12.5] engine flags: {res.flags}")
    assert flag in res.flags
    assert not any(f.startswith("implant:known_failure:") for f in res.flags), res.flags    # BORON_FLAG not set
    assert res.diagnostics["stopping_ck"] == pytest.approx(mine["ck"], rel=1e-9)
    assert res.diagnostics["stopping_ck_u"] == pytest.approx(mine["u"], rel=1e-9)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert flag in r.flags and not any(f.startswith("implant:known_failure:") for f in r.flags)
    # the default mode still carries the A7.9 boron flag
    rd = hub.submit(ADAPTER, params=_params(_implant("B", 50, n_ions=2000)))
    assert any(f.startswith("implant:known_failure:") for f in rd.flags), rd.flags
    assert rd.q("implant.Rp").value > r.q("implant.Rp").value         # a larger stopping gives a shallower range


def test_ck_override_one_equals_default(tmp_path):
    """run(problem, ck=...) replaces the entry's factor (A12.5); ck = 1.0 is the Lindhard-Scharff run itself, so
    with the same seed the depths are those of the default mode."""
    pytest.importorskip("libRustBCA")
    from simws import engines
    mod = engines.module("rustbca")
    p = _implant("B", 50, n_ions=2000)
    a = mod.run(_emp(p), tmp_path / "a", ck=1.0)
    b = mod.run(p, tmp_path / "b")
    assert a.depth_um == b.depth_um
    assert a.diagnostics["stopping_ck"] == 1.0


def test_capability_by_stopping_rule():
    """ImplantCapability.tested_ion_symbols_by_stopping: when the problem's electronic_stopping is a key, that list
    is used instead of tested_ion_symbols (A12.5)."""
    from simws.engines.base import ImplantCapability
    base = dict(engine="toy", env="toyenv", description="test sheet", in_process=True, structures=["amorphous"],
                potentials=["zbl"], electronic_stopping=["lindhard_scharff", "empirical"], recoils=[False],
                max_layers=20, max_keV_per_amu=25.0, tested_keV_per_amu=(1.0, 10.0), tested_target_symbols=["Si"],
                tested_ion_symbols=["P"])
    assert ImplantCapability(**base).tested_ion_symbols_by_stopping == {}
    c = ImplantCapability(**base, tested_ion_symbols_by_stopping={"empirical": ["B"]})
    assert c.outside_tested_range(_emp(_implant("B", 50))) == []
    r = c.outside_tested_range(_emp(_implant("P", 50)))
    assert len(r) == 1 and r[0].startswith("ion:"), r
    r = c.outside_tested_range(_implant("B", 50))
    assert len(r) == 1 and r[0].startswith("ion:"), r
    assert c.outside_tested_range(_implant("P", 50)) == []


# ------------------------------------------------------------------------------------------------ R16e, R16s
R16E = [("B", 50.0, 0.1608, 0.0504), ("B", 100.0, 0.2994, 0.0710)]


@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("a12_acceptance") / "store", mode="inprocess")


def _gate(r, ion, E, rp_ref, drp_ref, label):
    ok = {}
    for name, ref in (("implant.Rp", rp_ref), ("implant.dRp", drp_ref)):
        q = r.q(name)
        v, s = q.value / UM, q.uncertainty.sigma / UM
        ok[name] = abs(v - ref) <= 0.25 * ref
        print(f"\n[{label} {ion} {E:g} keV] {name}: {v * 1e3:.2f} +- {s * 1e3:.2f} nm, table {ref * 1e3:.1f} nm, "
              f"{v / ref - 1:+.4f} ({'within' if ok[name] else 'OUTSIDE'} 25 %)")
    return ok


def _r16e_ok(rhub, ion, E):
    r = rhub.submit(ADAPTER, params=_params(_emp(_implant(ion, E))))
    assert r.status == "ok", (r.status, r.message)
    return r


@pytest.mark.slow
@pytest.mark.parametrize("ion,E,rp_ref,drp_ref", [pytest.param(*c, id=f"{c[0]}{int(c[1])}keV") for c in R16E])
def test_R16e_held_out_table(rhub, ion, E, rp_ref, drp_ref):
    pytest.importorskip("libRustBCA")
    r = _r16e_ok(rhub, ion, E)
    assert any(f.startswith("implant:empirical_stopping:") for f in r.flags)
    ok = _gate(r, ion, E, rp_ref, drp_ref, "R16e")
    assert all(ok.values()), ok


# Marked after the run (section 2): all four R16e checks passed, but the capability still has
# tested_ion_symbols_by_stopping == {} (the A12.6 consequence has not been applied). Strict, so the update is noticed.
# amended by the lead: the A12.6 consequence is applied (A12.8), so the mark is removed
@pytest.mark.slow
def test_R16e_capability_follows_result(rhub):
    """A12.6: only if all four R16e checks pass does the capability get {"empirical": ["B"]}."""
    pytest.importorskip("libRustBCA")
    from simws import engines
    allpass = True
    for ion, E, rp_ref, drp_ref in R16E:
        allpass &= all(_gate(_r16e_ok(rhub, ion, E), ion, E, rp_ref, drp_ref, "R16e/capability").values())
    c = engines.capability("rustbca")
    print(f"\n[R16e] all four pass: {allpass}; capability tested_ion_symbols_by_stopping = "
          f"{c.tested_ion_symbols_by_stopping}")
    assert c.tested_ion_symbols_by_stopping == ({"empirical": ["B"]} if allpass else {})


@pytest.mark.slow
def test_R16s_sensitivity_reported(tmp_path):
    """Reported, no gate: ck - u and ck + u for the two boron problems (engine run with ck given, A12.5)."""
    pytest.importorskip("libRustBCA")
    from simws import engines
    mine = rule("B")
    mod = engines.module("rustbca")
    for ion, E, rp_ref, drp_ref in R16E:
        p = _emp(_implant(ion, E))
        line = []
        rps = {}
        for lab, ck in (("ck-u", mine["ck"] - mine["u"]), ("ck", mine["ck"]), ("ck+u", mine["ck"] + mine["u"])):
            x = np.asarray(mod.run(p, tmp_path / f"{E}{lab}", ck=ck).depth_um)
            rps[lab] = x.mean()
            line.append(f"{lab}={ck:.3f}: Rp {x.mean() * 1e3:.2f} nm ({x.mean() / rp_ref - 1:+.3f}), "
                        f"dRp {x.std(ddof=1) * 1e3:.2f} nm ({x.std(ddof=1) / drp_ref - 1:+.3f})")
        spread = (rps["ck-u"] - rps["ck+u"]) / rps["ck"]
        print(f"\n[R16s {ion} {E:g} keV] " + "; ".join(line) + f"; Rp spread (ck-u minus ck+u) / Rp(ck) = "
              f"{spread:+.3f}")
