"""Schottky (metal-semiconductor) contacts for DEVSIM regions built by simws.tcad.physics.setup_region (0.9b).

Implemented from the original papers (Charon's Schottky contacts serve only as an independent cross-check):

thermionic emission     Crowell & Sze (1966) boundary condition: the electron flux into the metal is
                        v_n (n_s - n_0B), v_n = A_n* T^2 / (q N_c) the thermionic recombination velocity and
                        n_0B = N_c exp(-q phi_Bn / kT) the interface density in equilibrium (holes likewise with
                        phi_Bp = E_g - phi_Bn). With a flat quasi-Fermi level this gives
                        J = A* T^2 exp(-q phi_Bn/kT) (exp(qV/kT) - 1); with drift-diffusion behind it, the combined
                        thermionic-emission/diffusion theory follows automatically.
image-force lowering    Schottky effect (Sze, Physics of Semiconductor Devices, 3rd ed., 3.2.4):
                        d_phi = sqrt(q E_s / (4 pi eps_s)) at the interface field E_s; it lowers the barrier for
                        both emission directions, v -> v exp(d_phi / V_t), so equilibrium stays current-free.
tunnelling              thermionic-field and field emission (Padovani & Stratton 1966; Crowell & Rideout 1969): electrons
                        below the barrier top tunnel through the depletion barrier. With the depletion-approximation
                        (parabolic) barrier of band bending U_b, the WKB exponent at energy t U_b above the bulk band
                        edge is (2 U_b / E00) g(t), g(t) = sqrt(1-t)/2 - (t/2) ln((1 + sqrt(1-t))/sqrt(t)), so the
                        emission is enhanced by 1 + Gamma,
                            Gamma = (U_b/kT) int_0^1 exp((U_b/kT)(1 - t) - (2 U_b/E00) g(t)) dt,
                        evaluated per contact node from the local interface field (U_b = eps E_s^2 / (2 q N)) with
                        32-point Gauss-Legendre quadrature. The same factor multiplies both emission directions, so
                        equilibrium stays current-free. Padovani-Stratton's TFE/FE formulas are saddle-point
                        evaluations of this integral (simws.validation.contact_checks compares them).

The electrostatic boundary condition puts the conduction band edge phi_Bn above the metal Fermi level
(Potential = V + V_t ln(n_0B / n_i) - dEc, the 0.8 convention n = n_i exp((psi + dEc)/V_t)).

    from simws.tcad.contacts import schottky_contact
    used = setup_region(dev, reg, mat, net, tot, phys, {"anode": False, "cathode": False})
    schottky_contact(dev, reg, "anode", mat, phys, phi_Bn=0.70, A_n=112.0, A_p=32.0)
"""
from __future__ import annotations

import math
from typing import Optional

from simws.units import K_B_EV, Q_E

H_BAR = 1.054571817e-34      # J s
M0 = 9.1093837015e-31        # kg
EPS0_F_PER_CM = 8.8541878128e-14


def thermionic_velocity(A_star: float, T: float, N_band: float) -> float:
    """v_R = A* T^2 / (q N) in cm/s (A* in A cm^-2 K^-2, N in cm^-3)."""
    return A_star * T * T / (Q_E * N_band)


def te_current(phi_B: float, V, T: float, A_star: float):
    """Closed-form thermionic emission J = A* T^2 exp(-phi_B/kT) (exp(V/kT) - 1), A/cm^2."""
    import numpy as np
    kT = K_B_EV * T
    return A_star * T * T * math.exp(-phi_B / kT) * (np.exp(np.asarray(V, float) / kT) - 1.0)


def e00(N: float, m_rel: float, eps_r: float) -> float:
    """Padovani-Stratton characteristic energy E00 = (q hbar / 2) sqrt(N / (m* eps)) in eV (N in cm^-3)."""
    N_m3 = N * 1e6
    eps = eps_r * EPS0_F_PER_CM * 100.0          # F/m
    return (Q_E * H_BAR / 2.0) * math.sqrt(N_m3 / (m_rel * M0 * eps)) / Q_E


GL_N = 32


def g_wkb(t):
    """Normalised WKB exponent of the parabolic depletion barrier at energy fraction t of the band bending."""
    import numpy as np
    t = np.asarray(t, float)
    r = np.sqrt(np.clip(1 - t, 0, None))
    return 0.5 * r - 0.5 * t * np.log((1 + r) / np.sqrt(t))


def tunnel_gamma(Ub, kT, E00, n=GL_N):
    """Gamma (numpy, Gauss-Legendre) for band bending Ub, kT and E00 in eV."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * (x + 1), 0.5 * w
    Ub = np.atleast_1d(np.asarray(Ub, float))[:, None]
    a, b = Ub / kT, 2 * Ub / E00
    return np.sum(w * a * np.exp(a * (1 - t) - b * g_wkb(t)), axis=1)


def tunnel_gamma_reference(Ub, kT, E00):
    """Same integral by adaptive quadrature (scipy), for checks."""
    import numpy as np
    from scipy.integrate import quad
    a, b = Ub / kT, 2 * Ub / E00
    f = lambda t: a * np.exp(a * (1 - t) - b * float(g_wkb(t)))
    return quad(f, 0.0, 1.0, limit=400, epsabs=0, epsrel=1e-12)[0]


def _gamma_expr(Ub: str, kT: float, E00: str, n=GL_N) -> str:
    """DEVSIM node expression for Gamma (Ub and E00 are node expressions in eV)."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * (x + 1), 0.5 * w
    terms = [f"{float(wi)!r}*exp(({Ub})*{float((1 - ti) / kT)!r} - 2*({Ub})/({E00})*{float(g_wkb(ti))!r})"
             for ti, wi in zip(t, w)]
    return f"(({Ub})/{kT!r})*(" + " + ".join(terms) + ")"


def schottky_contact(device, region, contact, mat, phys, phi_Bn: float, A_n: float = 112.0, A_p: float = 32.0,
                     image_force: bool = False, tunneling: bool = False, m_tunnel: float = 0.26,
                     n_i: Optional[float] = None) -> dict:
    """Replace the (ohmic) contact equations of `contact` by a Schottky contact with electron barrier phi_Bn (eV).
    The bias parameter is the same '<contact>_bias' the ohmic contacts use. Returns the parameters used."""
    from devsim import contact_equation, get_parameter, set_parameter
    from devsim.python_packages.model_create import CreateContactNodeModel
    from simws.materials.semiconductor import intrinsic_density
    from simws.tcad import cryo as C
    T = phys.T
    kT = K_B_EV * T
    ni = n_i if n_i is not None else get_parameter(device=device, region=region, name="n_i")
    Nc, Nv, Eg = C.band_params(mat, T, ni if n_i is not None else intrinsic_density(mat, T))
    vn, vp = thermionic_velocity(A_n, T, Nc), thermionic_velocity(A_p, T, Nv)
    n0B, p0B = Nc * math.exp(-phi_Bn / kT), Nv * math.exp(-(Eg - phi_Bn) / kT)
    p = f"{contact}_sk_"
    for k, v in (("vn", vn), ("vp", vp), ("n0B", n0B), ("p0B", p0B)):
        set_parameter(device=device, region=region, name=p + k, value=v)
    bias = f"{contact}_bias"
    try:
        get_parameter(device=device, name=bias)
    except Exception:  # noqa: BLE001
        set_parameter(device=device, name=bias, value=0.0)

    # interface field E_s (V/cm): a node solution holding the mean |E| of the edges at each contact node, updated by
    # an outer iteration after every solve (relax_schottky), like the Klaassen mobility; the factors below depend on
    # it only weakly, and the converged state does not depend on the lag
    eps = get_parameter(device=device, region=region, name="Permittivity")
    if image_force or tunneling:
        from devsim import node_solution, set_node_values
        node_solution(device=device, region=region, name="Es_node")
        set_node_values(device=device, region=region, name="Es_node", init_from="node_index")
        set_node_values(device=device, region=region, name="Es_node", values=[0.0] * len(_node_list(device, region)))
        _SK.setdefault((device, region), set()).add(contact)
    factors = []
    # image-force lowering: exp(d_phi/V_t), d_phi = sqrt(q E_s / (4 pi eps)) [V]
    if image_force:
        factors.append(f"exp(pow({Q_E / (4 * math.pi * eps)!r}*max(abs(Es_node), 1.0), 0.5)/V_t)")
    # tunnelling through the depletion barrier: 1 + Gamma(U_b, kT, E00) with U_b = eps E_s^2 / (2 q |N|) [eV]
    e00_coef = None
    if tunneling:
        e00_coef = e00(1.0, m_tunnel, eps / EPS0_F_PER_CM)            # E00 = e00_coef * sqrt(N)
        Nd = "max(abs(NetDoping), 1e10)"
        Ub = f"min({eps / (2 * Q_E)!r}*pow(Es_node, 2)/{Nd}, {phi_Bn!r})"
        E00x = f"({e00_coef!r}*pow({Nd}, 0.5))"
        factors.append(f"(1 + {_gamma_expr(f'max({Ub}, 1e-9)', kT, E00x)})")
    lower = "*".join(factors) if factors else "1"
    area = "ContactSurfaceArea/NodeVolume"
    # electrons: flux v_n (n - n0B) into the metal = a loss at the contact node; the terminal current is that flux
    fn = f"(ElectronCharge*{p}vn*{lower}*(Electrons - {p}n0B)*{area})"
    fp = f"(ElectronCharge*{p}vp*{lower}*(Holes - {p}p0B)*{area})"
    models = {
        f"{p}pot": f"Potential - {bias} - V_t*log({p}n0B/n_i) + dEc",
        f"{p}nmodel": f"ElectronGeneration - {fn}", f"{p}ncur": f"-{fn}",
        f"{p}pmodel": f"HoleGeneration + {fp}", f"{p}pcur": f"{fp}",
    }
    for name, expr in models.items():
        CreateContactNodeModel(device, contact, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateContactNodeModel(device, contact, f"{name}:{v}", f"diff({expr}, {v})")
    # the potential row is Dirichlet; the carrier rows keep the bulk edge flux and add the surface flux
    contact_equation(device=device, contact=contact, name="PotentialEquation", node_model=f"{p}pot",
                     edge_charge_model="contactcharge_edge" if _has_edge(device, region, "contactcharge_edge") else "")
    contact_equation(device=device, contact=contact, name="ElectronContinuityEquation", edge_model="ElectronCurrent",
                     node_model=f"{p}nmodel", node_current_model=f"{p}ncur")
    contact_equation(device=device, contact=contact, name="HoleContinuityEquation", edge_model="HoleCurrent",
                     node_model=f"{p}pmodel", node_current_model=f"{p}pcur")
    return {"phi_Bn_eV": phi_Bn, "phi_Bp_eV": Eg - phi_Bn, "vn_cm_s": vn, "vp_cm_s": vp, "n0B_cm3": n0B,
            "p0B_cm3": p0B, "Nc_cm3": Nc, "Nv_cm3": Nv, "Eg_eV": Eg, "image_force": image_force,
            "tunneling": tunneling, "m_tunnel": m_tunnel if tunneling else None}


_SK: dict = {}


def _contact_field(device, region):
    """Mean |ElectricField| over the edges incident to each node (V/cm)."""
    import numpy as np
    from devsim import edge_from_node_model, get_edge_model_values, get_node_model_values
    edge_from_node_model(device=device, region=region, node_model="node_index")
    i0 = np.array(get_edge_model_values(device=device, region=region, name="node_index@n0")).astype(int)
    i1 = np.array(get_edge_model_values(device=device, region=region, name="node_index@n1")).astype(int)
    E = np.abs(np.array(get_edge_model_values(device=device, region=region, name="ElectricField")))
    n = len(get_node_model_values(device=device, region=region, name="node_index"))
    s, c = np.zeros(n), np.zeros(n)
    np.add.at(s, i0, E); np.add.at(s, i1, E); np.add.at(c, i0, 1); np.add.at(c, i1, 1)
    return s / np.maximum(c, 1)


def update_schottky_field(device, region, relax: float = 1.0) -> float:
    """Refresh E_s at the contact nodes (under-relaxed by `relax`); returns the largest relative change of the full
    (unrelaxed) update."""
    import numpy as np
    from devsim import get_node_model_values, set_node_values
    if (device, region) not in _SK:
        return 0.0
    new = _contact_field(device, region)
    old = np.array(get_node_model_values(device=device, region=region, name="Es_node"))
    change = float(np.max(np.abs(new - old) / np.maximum(np.abs(new), 1.0)))
    set_node_values(device=device, region=region, name="Es_node", values=(old + relax * (new - old)).tolist())
    return change


def relax_schottky(device, region, opts, tol=1e-8, max_iter=200) -> int:
    """Outer iteration of the field-dependent emission factors (image force, tunnelling) after a solve. Strong
    tunnelling (Gamma >> 1 depends exponentially on E_s) makes plain Picard oscillate: the update is under-relaxed,
    halving the relaxation whenever the change grows. The converged state does not depend on the relaxation."""
    from devsim import solve
    r, last = 1.0, None
    for k in range(max_iter):
        ch = update_schottky_field(device, region, relax=r)
        if ch < tol:
            return k
        if last is not None and ch > 0.9 * last:
            r = max(r * 0.5, 0.02)
        last = ch
        solve(type="dc", **opts)
    raise RuntimeError(f"Schottky interface field did not converge (convergence failure) in {max_iter} iterations")


def _has_edge(device, region, name) -> bool:
    from devsim import get_edge_model_list
    return name in get_edge_model_list(device=device, region=region)


def _node_list(device, region):
    from devsim import get_node_model_values
    return get_node_model_values(device=device, region=region, name="node_index")
