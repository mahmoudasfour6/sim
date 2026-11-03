"""Cryogenic core physics (simws 0.9a): incomplete ionization, Fermi-Dirac statistics, Klaassen mobility.

Implemented from the original papers; Charon (Sandia) serves as the independent numerical cross-check.

Incomplete ionization (Altermatt, Schenk, Heiser & Green, SISPAD 2006; J. Appl. Phys. 100, 113714/113715, 2006)
    E_dop(N) = E_dop0 / (1 + (N/N_ref)^c)          screening lowers the binding energy at high doping
    b(N)     = 1 / (1 + (N/N_b)^d)                   fraction of states still bound (b -> 0: impurity band merges)
    n1       = N_c(T) exp(-E_dop/kT)
    N_D+/N_D = 1 - b n'/(n' + g n1),   n' = n/gamma_n (= N_c exp(eta_n); with Boltzmann statistics n' = n)
  model "charon": b = 1 below a critical doping and full ionization above it, constant E_dop (Charon's form,
  used only for the cross-check).

Fermi-Dirac statistics: n = N_c F_1/2(eta), F_1/2 normalised so F -> exp(eta) for eta -> -inf.
  DEVSIM expressions use the Bednarczyk & Bednarczyk (1978) approximation for F_1/2 (max error 0.38 %) and
  Nilsson's (1973) inverse (max density error 0.54 %), both checked against the exact integral in the unit tests.
  In the device equations FD enters through the degeneracy factor gamma = F_1/2(eta)/exp(eta): n = gamma N_c e^eta,
  the Scharfetter-Gummel currents use the effective potential psi + dEc - V_t ln(gamma_n) (generalised Einstein
  relation), and the contacts use the FD equilibrium.

Klaassen unified mobility (Klaassen, Solid-State Electron. 35, 953 and 961, 1992): lattice, ionized-impurity,
  carrier-carrier and clustering terms, with minority-carrier screening G(P) and electron-hole scattering F(P).
  Parameters in silicon.yaml (kl_*); G-function constants below. The ionized (not total) dopant densities are
  used when incomplete ionization is on, as in Charon.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from simws.units import K_B_EV

SQRT_PI = math.sqrt(math.pi)

# Klaassen (1992) G-function and screening constants
KL_G = dict(ag=0.89233, bg=0.41372, cg=0.005978, alpha_g=0.28227, alpha_prime_g=0.72169, beta_g=0.19778,
            gamma_g=1.80618)
KL_ME, KL_MH = 1.0, 1.258          # carrier masses in Klaassen's screening (m0)

SPECIES_DONOR = ("P", "As")
SPECIES_ACCEPTOR = ("B",)


# ---------------------------------------------------------------------------------------------- Fermi-Dirac
def fd12_exact(eta):
    """F_1/2(eta), normalised 2/sqrt(pi) int sqrt(x)/(1+exp(x-eta)) dx (-> exp(eta) when non-degenerate)."""
    from scipy.integrate import quad
    def one(e):
        if e < -30:
            return math.exp(e)
        top = max(e, 0.0) + 60.0
        return 2 / SQRT_PI * quad(lambda x: math.sqrt(x) / (1 + math.exp(min(x - e, 700.0))), 0, top,
                                  points=[max(e, 0.0)], limit=400)[0]
    return np.vectorize(one)(np.asarray(eta, float))


def fd12_bednarczyk(eta):
    eta = np.asarray(eta, float)
    v = eta ** 4 + 50 + 33.6 * eta * (1 - 0.68 * np.exp(-0.17 * (eta + 1) ** 2))
    return 1 / (np.exp(-eta) + 0.75 * SQRT_PI * v ** (-3 / 8))


def eta_from_u_exact(u):
    """Inverse of F_1/2: eta with F_1/2(eta) = u (u = n/N_c)."""
    from scipy.optimize import brentq
    def one(x):
        if x < 1e-12:
            return math.log(x)
        return brentq(lambda e: float(fd12_exact(e)) - x, math.log(x) - 2, max(math.log(x), 0) + 2 * x ** (2 / 3) + 5)
    return np.vectorize(one)(np.asarray(u, float))


def eta_from_u_nilsson(u):
    u = np.asarray(u, float) * (1 + 1e-12)
    nu = (0.75 * SQRT_PI * u) ** (2 / 3)
    return np.log(u) / (1 - u ** 2) + nu / (1 + (0.24 + 1.08 * nu) ** -2)


def fd12_expr(eta: str) -> str:
    """DEVSIM expression for F_1/2(eta) (Bednarczyk)."""
    e = f"({eta})"
    v = f"(pow({e},4) + 50 + 33.6*{e}*(1 - 0.68*exp(-0.17*pow({e} + 1,2))))"
    return f"(1/(exp(-{e}) + {0.75 * SQRT_PI!r}*pow({v}, -0.375)))"


U_CLAMP = 1000.0   # n/Nc above 1e3 (n > 4e21 cm^-3 at 77 K) is unphysical; only wild Newton iterates get there


def eta_expr(u: str) -> str:
    """DEVSIM expression for eta = F_1/2^-1(u) (Nilsson). u is clamped at U_CLAMP so a wild Newton iterate cannot
    overflow exp(-ln gamma); below the clamp the expression is exactly Nilsson's."""
    uu = f"(min({u}, {U_CLAMP!r})*1.000000000001)"
    nu = f"pow({0.75 * SQRT_PI!r}*{uu}, {2 / 3!r})"
    return f"(log({uu})/(1 - pow({uu},2)) + {nu}/(1 + pow(0.24 + 1.08*{nu}, -2)))"


def ln_gamma_expr(u: str) -> str:
    """ln(gamma) = ln(u) - eta(u): zero in the non-degenerate limit."""
    return f"(log(min({u}, {U_CLAMP!r})*1.000000000001) - {eta_expr(u)})"


# ---------------------------------------------------------------------------------------------- densities of states
def Nc(mat, T):
    """Green (1990) density of states x (T/300)^1.5. Named cryo_* so the 0.8 n_i(T) model is untouched (a set with
    Nc_300/Nv_300, e.g. the DFT profile, switches intrinsic_density to its DOS branch)."""
    return mat.get("cryo_Nc_300", "cm**-3") * (T / 300.0) ** 1.5


def Nv(mat, T):
    return mat.get("cryo_Nv_300", "cm**-3") * (T / 300.0) ** 1.5


# ---------------------------------------------------------------------------------------------- incomplete ionization
def ii_params(mat, species: str) -> Dict[str, float]:
    g = lambda k, u: mat.get(f"ii_{species}_{k}", u)
    return dict(E0=g("E0", "eV"), Nref=g("Nref", "cm**-3"), c=g("c", "dimensionless"), Nb=g("Nb", "cm**-3"),
                d=g("d", "dimensionless"), g=g("g", "dimensionless"))


def ionization_energy(p, N):
    return p["E0"] / (1 + (np.asarray(N, float) / p["Nref"]) ** p["c"])


def bound_fraction(p, N):
    return 1 / (1 + (np.asarray(N, float) / p["Nb"]) ** p["d"])


def ionized_fraction(p, N, carrier_prime, N_band, T, model="altermatt", N_crit=None, E_const=None):
    """N+/N given the 'Boltzmann-equivalent' carrier density n' = N_band exp(eta) (= n for Boltzmann)."""
    kT = K_B_EV * T
    N = np.asarray(N, float)
    if model == "charon":
        n1 = N_band * np.exp(-E_const / kT)
        g = 1 / p["g"]                                   # Charon's degeneracy factor (2 donors, 4 acceptors)
        f = 1 / (1 + g * np.asarray(carrier_prime, float) / n1)
        return np.where(N <= N_crit, f, 1.0)
    E = ionization_energy(p, N)
    n1 = N_band * np.exp(-E / kT)
    b = bound_fraction(p, N)
    c = np.asarray(carrier_prime, float)
    return 1 - b * c / (c + p["g"] * n1)


def ionized_expr(p, N: str, carrier_prime: str, N_band: float, T: float, model="altermatt", N_crit=None,
                 E_const=None) -> str:
    """DEVSIM node expression for the ionized density (N is a node expression of the total dopant density)."""
    kT = K_B_EV * T
    if model == "charon":
        n1 = N_band * math.exp(-E_const / kT)
        f = f"(1/(1 + {1 / p['g']!r}*({carrier_prime})/{n1!r}))"
        return f"ifelse(({N}) <= {N_crit!r}, ({N})*{f}, ({N}))"
    E = f"({p['E0']!r}/(1 + pow(({N})/{p['Nref']!r}, {p['c']!r})))"
    n1 = f"({N_band!r}*exp(-{E}/{kT!r}))"
    b = f"(1/(1 + pow(({N})/{p['Nb']!r}, {p['d']!r})))"
    return f"(({N})*(1 - {b}*({carrier_prime})/(({carrier_prime}) + {p['g']!r}*{n1})))"


def neutral_equilibrium(mat, ND, NA, T, donor="P", acceptor="B", fermi_dirac=True, model="altermatt",
                        n_i=None, N_crit=None, E_const=None) -> Tuple[float, float, float, float]:
    """Exact local charge neutrality p - n + N_D+ - N_A- = 0 for uniform doping (the closed-form reference).
    Exact Fermi-Dirac integral (not the device's Nilsson approximation): the difference to device_equilibrium is the
    statistics approximation error. Returns (n, p, N_D+, N_A-) in cm^-3."""
    from scipy.optimize import brentq
    from simws.materials.semiconductor import intrinsic_density
    kT = K_B_EV * T
    ni = n_i if n_i is not None else intrinsic_density(mat, T)
    Ncv, Nvv, Eg = band_params(mat, T, ni)                # same band parameters as the device (energies from E_v)
    pd = ii_params(mat, donor) if ND > 0 else None
    pa = ii_params(mat, acceptor) if NA > 0 else None
    F = (lambda e: float(fd12_exact(e))) if fermi_dirac else math.exp

    def parts(EF):
        eta_n, eta_p = (EF - Eg) / kT, -EF / kT
        n, p = Ncv * F(eta_n), Nvv * F(eta_p)
        nprime, pprime = Ncv * math.exp(eta_n), Nvv * math.exp(eta_p)
        Dp = ND * float(ionized_fraction(pd, ND, nprime, Ncv, T, model, N_crit, E_const)) if pd else 0.0
        Am = NA * float(ionized_fraction(pa, NA, pprime, Nvv, T, model, N_crit, E_const)) if pa else 0.0
        return n, p, Dp, Am

    def f(EF):
        n, p, Dp, Am = parts(EF)
        return (p + Dp - n - Am) / max(ND + NA, ni, 1.0)

    EF = brentq(f, -1.0, Eg + 1.0, xtol=1e-15, rtol=1e-15, maxiter=500)
    return parts(EF)


# ---------------------------------------------------------------------------------------------- Klaassen mobility
def _pmin(T, meff):
    """P at the minimum of Klaassen's G(P) (dG/dP = 0), found by bracketing in log P (robust at any T)."""
    import math as _m
    from scipy.optimize import brentq
    g = KL_G
    t = T / 300.0
    a1 = (t / meff) ** g["alpha_g"]
    a2 = (meff / t) ** g["alpha_prime_g"]

    def dG(lp):
        P = _m.exp(lp)
        return (g["ag"] * g["beta_g"] * a1 * (g["bg"] + P * a1) ** (-g["beta_g"] - 1)
                - g["cg"] * g["gamma_g"] * a2 ** (-g["gamma_g"]) * P ** (-g["gamma_g"] - 1))

    lo, hi = _m.log(1e-8), _m.log(1e4)
    if dG(lo) * dG(hi) > 0:
        return 0.0                                     # no interior minimum: no lower bound on P
    return _m.exp(brentq(dG, lo, hi, xtol=1e-14))


def klaassen_params(mat, carrier: str, species: str = "P") -> Dict[str, float]:
    pre = f"kl_n_{species}_" if carrier == "n" else "kl_p_"
    g = lambda k, u: mat.get(pre + k, u)
    out = dict(mumax=g("mumax", "cm**2/V/s"), mumin=g("mumin", "cm**2/V/s"), theta=g("theta", "dimensionless"),
               Nref=g("Nref", "cm**-3"), alpha=g("alpha", "dimensionless"))
    out.update(NrefD=mat.get("kl_NrefD", "cm**-3"), NrefA=mat.get("kl_NrefA", "cm**-3"),
               cD=mat.get("kl_cD", "dimensionless"), cA=mat.get("kl_cA", "dimensionless"))
    return out


def klaassen_mobility(mat, carrier, Na, Nd, n, p, T, species="P"):
    """Low-field Klaassen mobility (cm^2/Vs), numpy version (reference for the DEVSIM expression)."""
    k = klaassen_params(mat, carrier, species)
    g = KL_G
    t = T / 300.0
    Na, Nd, n, p = (np.asarray(v, float) for v in (Na, Nd, n, p))
    mu_L = k["mumax"] * t ** (-k["theta"])
    mu1 = k["mumax"] ** 2 / (k["mumax"] - k["mumin"]) * t ** (3 * k["alpha"] - 1.5)
    mu2 = k["mumax"] * k["mumin"] / (k["mumax"] - k["mumin"]) * t ** -0.5
    NaE = Na * (1 + Na ** 2 / (k["cA"] * Na ** 2 + k["NrefA"] ** 2))
    NdE = Nd * (1 + Nd ** 2 / (k["cD"] * Nd ** 2 + k["NrefD"] ** 2))
    meff, mr = (KL_ME, 1 / KL_MH) if carrier == "n" else (KL_MH, KL_MH)
    minority = p if carrier == "n" else n
    Nsc = NaE + NdE + minority
    Psc = 1 / (2.459 / (3.97e13 * Nsc ** (-2 / 3)) + 3.828 * (n + p) / (1.36e20 * meff)) * t ** 2
    Pe = np.maximum(Psc, _pmin(T, meff))
    G = (1 - g["ag"] * (g["bg"] + Pe * (t / meff) ** g["alpha_g"]) ** (-g["beta_g"])
         + g["cg"] * (Pe * (meff / t) ** g["alpha_prime_g"]) ** (-g["gamma_g"]))
    Fp = (0.7643 * Psc ** 0.6478 + 2.2999 + 6.5502 * mr) / (Psc ** 0.6478 + 2.3670 - 0.01552 * mr)
    if carrier == "n":
        Nsce = NdE + G * NaE + p / Fp
    else:
        Nsce = NaE + G * NdE + n / Fp
    mu_DAeh = mu1 * (Nsc / Nsce) * (k["Nref"] / Nsc) ** k["alpha"] + mu2 * (n + p) / Nsce
    return 1 / (1 / mu_L + 1 / mu_DAeh)


def klaassen_expr(mat, carrier, Na: str, Nd: str, n: str, p: str, T: float, species="P") -> str:
    """DEVSIM node expression of the Klaassen mobility (T fixed; densities are node expressions)."""
    k = klaassen_params(mat, carrier, species)
    g = KL_G
    t = T / 300.0
    mu_L = k["mumax"] * t ** (-k["theta"])
    mu1 = k["mumax"] ** 2 / (k["mumax"] - k["mumin"]) * t ** (3 * k["alpha"] - 1.5)
    mu2 = k["mumax"] * k["mumin"] / (k["mumax"] - k["mumin"]) * t ** -0.5
    NaE = f"(({Na})*(1 + pow({Na},2)/({k['cA']!r}*pow({Na},2) + {k['NrefA'] ** 2!r})))"
    NdE = f"(({Nd})*(1 + pow({Nd},2)/({k['cD']!r}*pow({Nd},2) + {k['NrefD'] ** 2!r})))"
    meff, mr = (KL_ME, 1 / KL_MH) if carrier == "n" else (KL_MH, KL_MH)
    minority = p if carrier == "n" else n
    Nsc = f"({NaE} + {NdE} + ({minority}) + 1e-30)"
    Psc = (f"(1/({2.459 / 3.97e13!r}*pow({Nsc}, {2 / 3!r}) + {3.828 / (1.36e20 * meff)!r}*(({n}) + ({p})))"
           f"*{t * t!r})")
    Pe = f"max({Psc}, {_pmin(T, meff)!r})"
    a1 = (t / meff) ** g["alpha_g"]
    a2 = (meff / t) ** g["alpha_prime_g"]
    G = (f"(1 - {g['ag']!r}*pow({g['bg']!r} + {Pe}*{a1!r}, {-g['beta_g']!r})"
         f" + {g['cg']!r}*pow({Pe}*{a2!r}, {-g['gamma_g']!r}))")
    Fp = (f"((0.7643*pow({Psc},0.6478) + {2.2999 + 6.5502 * mr!r})/(pow({Psc},0.6478) + {2.3670 - 0.01552 * mr!r}))")
    if carrier == "n":
        Nsce = f"({NdE} + {G}*{NaE} + ({p})/{Fp} + 1e-30)"
    else:
        Nsce = f"({NaE} + {G}*{NdE} + ({n})/{Fp} + 1e-30)"
    mu = (f"({mu1!r}*({Nsc}/{Nsce})*pow({k['Nref']!r}/{Nsc}, {k['alpha']!r})"
          f" + {mu2!r}*(({n}) + ({p}))/{Nsce})")
    return f"(1/({1 / mu_L!r} + 1/{mu}))"


# ---------------------------------------------------------------------------------------------- device statistics
# The device equations define FD statistics through Nilsson's inverse: n' = n / gamma_n = N_c exp(eta_nil(n/N_c)).
# Everything that must be exactly consistent with the device (contact equilibria, initial guesses, the closed-form
# neutrality check of the gate) uses these same relations.
def band_params(mat, T, n_i):
    """(N_c, N_v, E_g) with N_c N_v exp(-E_g/kT) = n_i^2: Green's densities of states scaled by a common factor
    so the Boltzmann limit is exactly the 0.8 model (n_i from simws.materials, Varshni gap)."""
    kT = K_B_EV * T
    Eg = mat.get("varshni_Eg0", "eV") - mat.get("varshni_alpha", "eV/K") * T * T / (T + mat.get("varshni_beta", "K"))
    Nc0, Nv0 = Nc(mat, T), Nv(mat, T)
    s = n_i / math.sqrt(Nc0 * Nv0 * math.exp(-Eg / kT))
    return Nc0 * s, Nv0 * s, Eg


def prime_from_density(n, Nb, fermi_dirac=True):
    n = np.asarray(n, float)
    return Nb * np.exp(eta_from_u_nilsson(n / Nb)) if fermi_dirac else n


def density_from_prime(nprime, Nb, fermi_dirac=True):
    """Inverse of prime_from_density (vectorised bisection in log u; exact to 1e-14 relative)."""
    nprime = np.asarray(nprime, float)
    if not fermi_dirac:
        return nprime
    target = np.log(nprime / Nb)
    # eta_nil(u) ~ ln u for small u and ~ (3 sqrt(pi) u / 4)^(2/3) for large u: bracket ln u accordingly
    lo = np.minimum(target, 0.0) - 2.0
    hi = np.where(target < 0, 2.0, 1.5 * np.log(np.maximum(target, 1.0) * 4 / (3 * SQRT_PI)) + 4.0)
    hi = np.maximum(hi, target + 2.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", invalid="ignore"):
            f = eta_from_u_nilsson(np.exp(mid)) - target
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return Nb * np.exp(0.5 * (lo + hi))


def device_equilibrium(mat, ND, NA, n_ie, T, Ncv, Nvv, donor="P", acceptor="B", incomplete=True,
                       fermi_dirac=True, model="altermatt", N_crit=None, E_const=None):
    """Local neutrality p - n + N_D+ - N_A- = 0 with the device's statistics, for arrays of (ND, NA, n_ie).
    Returns (n, p, n', p'); n' p' = n_ie^2 (band-edge shifts of BGN and Ge included through n_ie)."""
    from scipy.optimize import brentq
    ND, NA, n_ie = (np.atleast_1d(np.asarray(v, float)) for v in (ND, NA, n_ie))
    pd, pa = ii_params(mat, donor), ii_params(mat, acceptor)
    cache, out = {}, np.zeros((4, ND.size))
    for i, (d, a, ni) in enumerate(zip(ND, NA, n_ie)):
        key = (float(f"{d:.10g}"), float(f"{a:.10g}"), float(f"{ni:.10g}"))
        if key not in cache:
            def parts(x, d=d, a=a, ni=ni):
                nprime = math.exp(x)
                pprime = ni * ni / nprime
                n = float(density_from_prime(nprime, Ncv, fermi_dirac))
                p = float(density_from_prime(pprime, Nvv, fermi_dirac))
                Dp = d * float(ionized_fraction(pd, d, nprime, Ncv, T, model, N_crit, E_const)) if (incomplete and d > 0) else d
                Am = a * float(ionized_fraction(pa, a, pprime, Nvv, T, model, N_crit, E_const)) if (incomplete and a > 0) else a
                return n, p, nprime, pprime, Dp, Am

            def f(x):
                n, p, _, _, Dp, Am = parts(x)
                return (p + Dp - n - Am) / max(d + a, ni, 1.0)

            x0 = math.log(ni)
            x = brentq(f, x0 - 120, x0 + 120, xtol=1e-14, rtol=1e-15, maxiter=500)
            cache[key] = parts(x)[:4]
        out[:, i] = cache[key]
    return out[0], out[1], out[2], out[3]
