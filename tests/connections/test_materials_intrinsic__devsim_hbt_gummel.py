"""materials.intrinsic -> devsim.hbt_gummel: the n_i the device solves with is the one the material node reports."""
import pytest

pytest.importorskip("devsim")


@pytest.mark.parametrize("T", [250.0, 350.0])
def test_ni_consistent(hub, T):
    from devsim import get_parameter
    from simws.tcad.hbt import HBTCase, build
    r = hub.submit("materials.intrinsic", params={"material": "silicon", "profile": None, "T": ([T], "K")})
    ni_node = r.q("mat.n_i").to("cm**-3")[0]
    build(HBTCase(T=T, ge_peak=0.0))
    ni_dev = get_parameter(device="d", region="si", name="n_i")
    assert ni_dev == pytest.approx(ni_node, rel=1e-12)
