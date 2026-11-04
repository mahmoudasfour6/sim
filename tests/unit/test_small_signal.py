"""0.9b small-signal AC and noise (simws.validation.small_signal_checks); the HBT f_T gate runs in the 0.9b gate."""
from simws.validation import small_signal_checks as K


def test_depletion_capacitance():
    r = K.depletion_capacitance()
    assert r["pass"], r


def test_shot_noise_of_ideal_diode():
    r = K.shot_noise()
    assert r["pass"], r


def test_nyquist_noise_2d():
    r = K.nyquist_2d()
    assert r["pass"], r
