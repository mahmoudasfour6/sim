"""Contract amendment A6.3 (docs/ARCHITECTURE_V1.md): engine-neutral two-port figures, simws.analysis.two_port.

Written from the contract text alone, without reading the implementation. Every expected value is derived in the
test from the formulas of A6.3 (which are those of the legacy simws.tcad.small_signal.figures).

Call forms. A6.3 fixes `f_t_single_pole(f, |h21|, beta0)` and `f_max(f, U)`. For `h21` and `mason_u` it says only
"with complex y11, y12, y21, y22": the checker's GUESS is the literal one, four complex positional arguments in the
order y11, y12, y21, y22. The two wrappers below are the only place that guess lives.
"""
import cmath
import math

import pytest

two_port = pytest.importorskip("simws.analysis.two_port")

RTOL = 1e-12          # pure arithmetic: only the order of floating-point operations can differ


def H21(y11, y12, y21, y22):
    return two_port.h21(y11, y12, y21, y22)                 # guessed call form (see module docstring)


def U(y11, y12, y21, y22):
    return two_port.mason_u(y11, y12, y21, y22)             # guessed call form (see module docstring)


def _close(a, b, rtol=RTOL):
    return abs(a - b) <= rtol * abs(b)


# ------------------------------------------------------------------------------------------------ h21
@pytest.mark.parametrize("y", [
    (0.04 + 0.8j, -0.01j, 4.0 - 0.01j, 1e-3 + 0.01j),
    (1.0 + 0j, 0j, 2.0 + 0j, 1.0 + 0j),
    (3e-2 - 2e-3j, 1e-4 + 1e-4j, -7.0 + 0.5j, 2e-3 + 0j),
])
def test_h21_is_y21_over_y11(y):
    y11, y12, y21, y22 = y
    h = complex(H21(y11, y12, y21, y22))
    assert _close(h, y21 / y11), (h, y21 / y11)


def test_h21_single_pole_transistor():
    """Hybrid-pi without feedback: y11 = g_pi + j w C, y21 = g_m -> h21 = beta0 / (1 + j f / f_beta) with
    beta0 = g_m / g_pi and f_beta = g_pi / (2 pi C)."""
    g_m, beta0, f_beta = 4.0, 100.0, 2.0e8
    g_pi = g_m / beta0
    C = g_pi / (2 * math.pi * f_beta)
    for f in (1e6, 2e8, 5e9):
        y11 = complex(g_pi, 2 * math.pi * f * C)
        h = complex(H21(y11, 0j, complex(g_m, 0.0), complex(1e-3, 0.0)))
        assert _close(h, beta0 / (1 + 1j * f / f_beta), 1e-12)


# ------------------------------------------------------------------------------------------------ f_t_single_pole
# Single pole: |h21|^2 = beta0^2 / (1 + (f/f_beta)^2), so (beta0/|h21|)^2 - 1 = (f/f_beta)^2 and the formula gives
# beta0 * f / (f / f_beta) = beta0 * f_beta at every f. Frequencies with f/f_beta >= 0.1 only: for f << f_beta the
# formula itself subtracts two nearly equal numbers ((beta0/|h21|)^2 - 1 ~ (f/f_beta)^2), which costs ~eps/(f/f_beta)^2
# of relative accuracy whatever the implementation (1e-12 needs f/f_beta >~ 0.02; 0.1 leaves margin).
@pytest.mark.parametrize("beta0,f_beta", [(100.0, 2.0e8), (86.0, 3.3e9), (2.5, 1.0e6)])
@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1e4])
def test_f_t_single_pole_exact_for_single_pole(beta0, f_beta, ratio):
    f = ratio * f_beta
    h_abs = abs(beta0 / (1 + 1j * f / f_beta))
    fT = two_port.f_t_single_pole(f, h_abs, beta0)
    assert fT is not None
    assert _close(float(fT), beta0 * f_beta), (fT, beta0 * f_beta)


def test_f_t_single_pole_hand_value():
    """f = 1e9, |h21| = 3, beta0 = 5: (5/3)^2 - 1 = 16/9, sqrt = 4/3, f_T = 5e9 / (4/3) = 3.75e9."""
    assert _close(float(two_port.f_t_single_pole(1e9, 3.0, 5.0)), 3.75e9)


@pytest.mark.parametrize("h_abs,beta0", [(5.0, 5.0), (6.0, 5.0), (100.0, 1.0)])
def test_f_t_single_pole_none_unless_beta0_above_h21(h_abs, beta0):
    assert two_port.f_t_single_pole(1e9, h_abs, beta0) is None


# ------------------------------------------------------------------------------------------------ mason_u
def _u_hand(y11, y12, y21, y22):
    den = 4 * (y11.real * y22.real - y12.real * y21.real)
    return math.inf if den <= 0 else abs(y21 - y12) ** 2 / den


def test_mason_u_hand_value():
    """y11 = 1 + 1j, y12 = 0.1 - 0.2j, y21 = 3 + 0.5j, y22 = 0.5 + 0.1j:
    y21 - y12 = 2.9 + 0.7j, |.|^2 = 8.41 + 0.49 = 8.90; den = 4 (1 * 0.5 - 0.1 * 3) = 0.8; U = 11.125."""
    u = float(U(1 + 1j, 0.1 - 0.2j, 3 + 0.5j, 0.5 + 0.1j))
    assert _close(u, 11.125)


@pytest.mark.parametrize("y", [
    (0.04 + 0.8j, -0.01j, 4.0 - 0.01j, 1e-3 + 0.01j),
    (3e-2 - 2e-3j, 1e-4 + 1e-4j, -7.0 + 0.5j, 2e-3 + 0j),
    (2.0 + 0j, -0.5 + 0.1j, 0.3 + 0j, 1.0 + 3j),
])
def test_mason_u_formula(y):
    assert _close(float(U(*y)), _u_hand(*y))


def test_mason_u_unilateral():
    """y12 = 0: U = |y21|^2 / (4 Re y11 Re y22)."""
    y11, y21, y22 = 0.04 + 0.3j, 4.0 - 0.2j, 2e-3 + 0.05j
    assert _close(float(U(y11, 0j, y21, y22)), abs(y21) ** 2 / (4 * y11.real * y22.real))


def test_mason_u_hybrid_pi():
    """Hybrid-pi with C_mu: y12 = -j w C_mu, y21 = g_m - j w C_mu, so y21 - y12 = g_m and Re y12 = 0:
    U = g_m^2 / (4 g_pi g_o) at every frequency."""
    g_m, g_pi, g_o, C_pi, C_mu = 4.0, 0.04, 1e-3, 1.3e-11, 1e-12
    for f in (1e6, 1e9, 3e10):
        w = 2 * math.pi * f
        y = (complex(g_pi, w * (C_pi + C_mu)), complex(0, -w * C_mu), complex(g_m, -w * C_mu),
             complex(g_o, w * C_mu))
        assert _close(float(U(*y)), g_m ** 2 / (4 * g_pi * g_o))


def test_mason_u_invariants():
    """Derived from the formula: U depends on y21 - y12 and on the real parts only, so (a) a lossless (purely
    imaginary) feedback admittance jB between the ports (y11, y22 += jB; y12, y21 -= jB) leaves it unchanged,
    (b) lossless shunts jB1 at port 1 and jB2 at port 2 leave it unchanged, (c) scaling every y by k > 0 leaves it
    unchanged (numerator and denominator both scale by k^2)."""
    y = (0.04 + 0.8j, 1e-6 - 0.01j, 4.0 - 0.01j, 1e-3 + 0.01j)       # den = 4 (4e-5 - 4e-6) > 0
    u0 = float(U(*y))
    assert math.isfinite(u0) and u0 > 0
    B = 0.37
    fb = (y[0] + 1j * B, y[1] - 1j * B, y[2] - 1j * B, y[3] + 1j * B)
    assert _close(float(U(*fb)), u0, 1e-11)
    sh = (y[0] + 2.1j, y[1], y[2], y[3] - 0.7j)
    assert _close(float(U(*sh)), u0, 1e-11)
    for k in (1e-3, 7.0, 1e4):
        assert _close(float(U(*(k * v for v in y))), u0, 1e-11)


@pytest.mark.parametrize("y", [
    (1.0 + 0j, 0j, 2.0 + 0j, 0j),                              # Re y22 = 0: denominator 0
    (1.0 + 0j, 0j, 2.0 + 1j, -1e-3 + 0j),                      # Re y22 < 0: denominator < 0
    (1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j),                  # Re y11 Re y22 = Re y12 Re y21: exactly 0
    (1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 1.0 + 0j),                  # 1 - 6 < 0
])
def test_mason_u_inf_when_denominator_not_positive(y):
    assert math.isinf(float(U(*y))) and float(U(*y)) > 0


def test_mason_u_two_terminal_is_inf():
    """A two-terminal device seen as a two-port (y11 = y22 = -y12 = -y21): numerator |y21 - y12|^2 = 0 and
    denominator 4 (G^2 - G^2) = 0, so the contract's rule gives inf (denominator <= 0 wins)."""
    y = 1e-3 + 2e-2j
    assert math.isinf(float(U(y, -y, -y, y)))


# ------------------------------------------------------------------------------------------------ f_max
@pytest.mark.parametrize("f,u", [(1e9, 4.0), (2.5e8, 1234.5), (1e10, 0.25)])
def test_f_max(f, u):
    assert _close(float(two_port.f_max(f, u)), f * math.sqrt(u))


@pytest.mark.parametrize("u", [math.inf, 0.0, -1.0, -math.inf, math.nan])
def test_f_max_none_unless_finite_positive(u):
    assert two_port.f_max(1e9, u) is None


def test_figures_chain_matches_legacy_figures():
    """The legacy simws.tcad.small_signal.figures (pure arithmetic on a Y dict, no engine call) on the same
    numbers: h21, single-pole f_T, U and f_max agree to RTOL. The legacy keys Y[(i, j)] with 1 = base, 2 = collector."""
    from simws.tcad.small_signal import figures
    y = (0.04 + 0.08j, -0.001j, 4.0 - 0.001j, 1e-3 + 0.001j)
    Y = {(1, 1): y[0], (1, 2): y[1], (2, 1): y[2], (2, 2): y[3]}
    f, beta0 = 3e9, 99.0
    leg = figures(Y, f, beta0)
    h = abs(complex(H21(*y)))
    assert _close(h, leg["h21"])
    assert _close(float(two_port.f_t_single_pole(f, h, beta0)), leg["fT_single_pole"])
    u = float(U(*y))
    assert _close(u, leg["U"])
    assert _close(float(two_port.f_max(f, u)), leg["fmax_est"])
    assert cmath.isclose(complex(H21(*y)), y[2] / y[0], rel_tol=RTOL)
