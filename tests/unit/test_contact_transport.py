"""0.9 closure: Schottky transport with the image force inside the transmission and Fermi-Dirac supply."""
from simws.validation import contact_transport_checks as C


def test_thermionic_limit():
    assert C.te_limit()["pass"]


def test_fd_step_exact():
    assert C.fd_step_exact()["pass"]


def test_zero_bias_detailed_balance():
    assert C.zero_bias()["pass"]


def test_wkb_triangle():
    assert C.wkb_triangle()["pass"]


def test_fermi_dirac_inversion():
    assert C.degenerate_xi()["pass"]


def test_image_lowering_close_to_closed_form():
    for r in C.image_lowering()["rows"]:
        assert abs(r["rel"]) < 0.02
