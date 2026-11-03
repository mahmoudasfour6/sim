"""radiation.tid_mos -> devsim.mos_cv: trapped charge handed over in canonical units (1/m^2) reproduces the
flat-band shift the TID adapter reports. Negative irradiation bias: no interface traps, holes near the gate."""
import numpy as np
import pytest

pytest.importorskip("devsim")


def test_trapped_charge_handover(hub):
    t_ox, NA = 20.0, 1e17
    r = hub.submit("radiation.tid_mos", params={"t_ox": (t_ox, "nm"), "V_irr": (-2.0, "V"), "dose": ([0, 1e4], "Gy"),
                                                "NA": (NA, "cm**-3")})
    assert r.status == "ok"
    assert r.q("tid.N_it").array()[-1] == 0.0
    N_ot_cm2 = r.q("tid.N_ot").to("1/cm**2")[-1]
    V = (np.linspace(-0.6, 0.4, 41), "V")          # V_fb is interpolated where psi_s = 0: needs a fine grid
    base = {"t_ox_nm": t_ox, "NA": NA}
    c0 = hub.submit("devsim.mos_cv", params={"case": base, "V": V})
    c1 = hub.submit("devsim.mos_cv", params={"case": {**base, "oxide": [[t_ox - 1.0, N_ot_cm2]]}, "V": V})
    dVfb = c1.q("mos.V_fb").value - c0.q("mos.V_fb").value
    assert dVfb == pytest.approx(r.q("tid.dV_fb").value[-1], abs=2e-4)
