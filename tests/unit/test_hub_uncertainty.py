"""Unit tests for simws 0.8: correlated ensembles, budgets, guards, verification records, MMS Poisson order."""
import math

import numpy as np
import pytest

pytest.importorskip("pydantic")

from simws.hub import guards as G  # noqa: E402
from simws.hub import uncertainty as U  # noqa: E402
from simws.hub import verification as V  # noqa: E402
from simws.hub.schema import Quantity  # noqa: E402


def test_member_ids():
    m = U.member_id(sample=3, functional="PBE")
    assert m == "functional=PBE|sample=3"
    assert U.parse_member(m) == {"functional": "PBE", "sample": "3"}
    assert U.sub_member(m, "functional") == "functional=PBE"
    assert U.member_id() == "base"
    with pytest.raises(ValueError):
        U.member_id(x="a|b")


def _ens(name, vals, unit="A"):
    return U.Ensemble.from_quantities(
        [Quantity(name=name, value=v, unit=unit, member=U.member_id(sample=i)) for i, v in enumerate(vals)])


def test_correlated_propagation_beats_independent():
    rng = np.random.default_rng(1)
    common = rng.normal(1.0, 0.1, 50)                     # shared (e.g. mobility) factor in both currents
    ic, ib = _ens("IC", 1e-3 * common), _ens("IB", 1e-5 * common)
    beta = U.propagate(lambda IC, IB: IC / IB, "beta", **{"IC": ic, "IB": ib})
    assert np.allclose(beta.stack(), 100.0)               # correlation cancels exactly
    assert U.correlation(ic, ib) == pytest.approx(1.0)
    shuffled = _ens("IB", 1e-5 * rng.permutation(common))
    b2 = U.propagate(lambda IC, IB: IC / IB, **{"IC": ic, "IB": shuffled})
    assert b2.std() > 5.0                                 # breaking the pairing inflates the spread
    with pytest.raises(ValueError):
        U.propagate(lambda a, b: a, a=ic, b=_ens("x", [1.0, 2.0]))


def test_model_form_and_budget():
    fun = U.Ensemble.from_quantities([Quantity(name="Eg", value=v, unit="eV", member=U.member_id(functional=f))
                                      for f, v in (("LDA", 0.50), ("PBE", 0.58), ("PBEsol", 0.54))])
    assert fun.range_sigma() == pytest.approx(0.08 / math.sqrt(12))
    assert set(fun.group("functional")) == {"LDA", "PBE", "PBEsol"}
    u = U.budget(1.12, numerical=0.003, parametric=0.004, model_form=float(fun.range_sigma()))
    assert u.sigma == pytest.approx(math.sqrt(0.003 ** 2 + 0.004 ** 2 + float(fun.range_sigma()) ** 2))
    assert U.dominant(u) == "model_form"
    assert U.missing_components(U.budget(1.0, numerical=0.1)) == ["parametric", "model_form"]
    assert U.budget(1.0).kind == "none"
    with pytest.raises(ValueError):
        U.budget(1.0, numerical=-1.0)
    q = fun.to_quantity()
    assert q.uncertainty.kind == "ensemble" and len(q.uncertainty.members) == 3


def test_verification_records():
    # f(h) = 1 + h^2 sampled at h = 1, 1/2, 1/4: order 2, Richardson 1
    rec = V.solution_from_values([0, 1, 2], [2.0, 1.25, 1.0625])
    assert rec["order"] == pytest.approx(2.0) and rec["richardson"] == pytest.approx(1.0)
    assert rec["sigma_num"] == pytest.approx(1.25 * (0.1875 / 1.0625) / 3 * 1.0625 / 2)
    assert V.code(1.98, 2.0)["passed"] and not V.code(1.5, 2.0)["passed"]


def test_guards():
    assert G.positive("n", [1, 2]) == [] and G.positive("n", [1, 0])
    assert G.monotonic("I", [1, 2, 2]) == [] and G.monotonic("I", [1, 2, 2], strict=True)
    assert G.mass_action([1e16], [1e4], 1e10) == [] and G.mass_action([1e16], [1e5], 1e10)
    assert G.kirchhoff([[1e-3], [1e-5], [-1.01e-3]]) == []
    assert G.kirchhoff([[1e-3], [1e-5], [-1.0e-3]])
    V_ = np.linspace(0.5, 0.8, 7)
    assert G.ideality("IC", V_, 1e-15 * np.exp(V_ / 0.025852)) == []
    assert G.ideality("IC", V_, 1e-15 * np.exp(V_ / (3 * 0.025852)))


def test_mms_poisson_second_order():
    pytest.importorskip("devsim")
    from simws.tcad.mms import poisson_order
    r = poisson_order(levels=3, n0=20)
    assert r["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert r["err_max_V"][-1] < 1e-7
