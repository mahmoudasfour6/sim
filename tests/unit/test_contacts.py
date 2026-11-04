"""0.9b Schottky contacts: thermionic emission, image-force lowering, tunnelling (simws.validation.contact_checks)."""
from simws.validation import contact_checks as K


def test_thermionic_emission_closed_form():
    r = K.te_closed_form()
    assert r["pass"], r


def test_detailed_balance_with_lowering_and_tunnelling():
    r = K.detailed_balance()
    assert r["pass"], r


def test_image_force_lowering_closed_form():
    r = K.image_force()
    assert r["pass"], r


def test_tunnelling_quadrature_and_device():
    assert K.tunnel_quadrature()["pass"]
    r = K.tunnel_device()
    assert r["pass"], r


def test_padovani_stratton_tfe_regime_within_factor_two():
    lo, hi = K.padovani_stratton()["tfe_regime_ratio_range"]
    assert 0.5 < lo and hi < 2.0
