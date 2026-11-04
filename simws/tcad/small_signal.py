"""Small-signal AC analysis of the HBT (0.9b): f_T, f_max and the charge-control transit time.

AC      DEVSIM's small-signal solve (solve(type="ac")) linearises the drift-diffusion system about the DC point, with
        the contacts as circuit nodes driven by voltage sources (simws.tcad.hbt.build(circuit=True)). Y parameters
        in common emitter: y_ij = -I(V_i) / v_j with v_j the AC drive on source j (all other sources AC-shorted).
        h21 = y21 / y11. f_T from the single-pole form h21 = beta_0 / (1 + j f / f_beta), fitted with the
        small-signal beta_0 (h21 at f_T/1e4) and |h21(f)| at f = f_T/10: f_T = beta_0 f_beta (the plain f |h21| is
        low by the beta roll-off, 3 % at f = f_T/20 for beta 86). Mason's unilateral gain
        U = |y21 - y12|^2 / (4 (Re y11 Re y22 - Re y12 Re y21)), f_max = f sqrt(U(f)).
charge  quasi-static charge control (Kroemer; Gummel's transit-time picture): with the collector-emitter voltage
        fixed, every hole the base delivers is stored in the device, so the input capacitance is
        C_in = q d(int p dV)/dV_BE and tau = C_in / g_m = dQ_B / dI_C; then f_T = 1 / (2 pi tau). The AC f_T and this
        must agree while non-quasi-static effects are small (0.9b gate: 5 %).
Currents and capacitances are per cm of emitter length (2D device); ratios (h21, U) and frequencies are not.
"""
from __future__ import annotations

import math

import numpy as np

SOURCES = ("VE", "VB", "VC")


def _src_current(name, solution="dcop"):
    from devsim import get_circuit_node_value
    return get_circuit_node_value(node=f"{name}.I", solution=solution)


def bias(VBE: float, VCE: float, opts=None, cb=None):
    """Ramp the circuit sources to (VBE, VCE) with the emitter grounded."""
    from devsim import get_circuit_node_value
    from simws.tcad.solve import ramp_source
    o = dict(absolute_error=1e10, relative_error=1e-10, maximum_iterations=60)
    o.update(opts or {})
    vc0 = get_circuit_node_value(node="collector_bias", solution="dcop")
    vb0 = get_circuit_node_value(node="base_bias", solution="dcop")
    ramp_source("VC", VCE, start=vc0, step=0.1, callback=cb, **o)
    ramp_source("VB", VBE, start=vb0, step=0.05, callback=cb, **o)


def mobility_callback(case):
    """Klaassen mobility depends on the carriers (cryo=True): relax it at every bias step, as the Gummel adapter does.
    The AC solve itself linearises at frozen mobility (d mu / d n is not in DEVSIM's Jacobian); the charge-control
    f_T includes it through the finite differences, so the AC-vs-charge-control difference bounds that omission."""
    if not getattr(case, "cryo", False):
        return None
    from simws.tcad.physics import relax_mobility
    o = dict(absolute_error=1e10, relative_error=1e-10, maximum_iterations=40)
    return lambda _v=None: relax_mobility("d", "si", o)


def dc_currents():
    """Terminal currents into the device (A per cm of emitter length): emitter, base, collector."""
    return {c: -_src_current(k) for k, c in zip(SOURCES, ("emitter", "base", "collector"))}


def y_params(f: float):
    """Common-emitter Y parameters at frequency f (S per cm): {(i, j)} with 1 = base, 2 = collector."""
    from devsim import circuit_alter, solve
    Y = {}
    for j, drive in ((1, "VB"), (2, "VC")):
        circuit_alter(name=drive, param="acreal", value=1.0)
        solve(type="ac", frequency=f)
        for i, k in ((1, "VB"), (2, "VC")):
            Y[(i, j)] = -complex(_src_current(k, "ssac_real"), _src_current(k, "ssac_imag"))
        circuit_alter(name=drive, param="acreal", value=0.0)
    return Y


def figures(Y, f, beta0=None):
    h21 = Y[(2, 1)] / Y[(1, 1)]
    fT_pole = None
    if beta0 is not None and beta0 > abs(h21):
        fbeta = f / math.sqrt((beta0 / abs(h21)) ** 2 - 1)
        fT_pole = beta0 * fbeta
    den = 4 * (Y[(1, 1)].real * Y[(2, 2)].real - Y[(1, 2)].real * Y[(2, 1)].real)
    U = abs(Y[(2, 1)] - Y[(1, 2)]) ** 2 / den if den > 0 else float("inf")
    return {"f": f, "h21": abs(h21), "fT_est": f * abs(h21), "fT_single_pole": fT_pole, "U": U,
            "fmax_est": f * math.sqrt(U) if U > 0 else None}


def hole_charge():
    """q * integral of p over the device, C per cm."""
    from devsim import get_node_model_values
    from simws.units import Q_E
    p = np.array(get_node_model_values(device="d", region="si", name="Holes"))
    v = np.array(get_node_model_values(device="d", region="si", name="NodeVolume"))
    return Q_E * float(np.sum(p * v))


def charge_control(VBE, VCE, dV=1e-3, opts=None, cb=None):
    """g_m, C_in = dQ_B/dV_BE and tau = C_in / g_m by central differences at fixed V_CE (cb: per-step callback,
    e.g. the Klaassen mobility relaxation of cryogenic devices, so each point is self-consistent)."""
    vals = []
    for v in (VBE - dV, VBE + dV):
        bias(v, VCE, opts, cb=cb)
        if cb:
            cb(v)
        vals.append((dc_currents()["collector"], hole_charge()))
    bias(VBE, VCE, opts, cb=cb)
    if cb:
        cb(VBE)
    gm = (vals[1][0] - vals[0][0]) / (2 * dV)
    Cin = (vals[1][1] - vals[0][1]) / (2 * dV)
    return {"gm_S_per_cm": gm, "Cin_F_per_cm": Cin, "tau_s": Cin / gm, "fT_charge_Hz": gm / (2 * math.pi * Cin)}


def hbt_small_signal(case=None, VBE=0.8, VCE=1.0, decades=(0.05, 0.1, 0.2), opts=None) -> dict:
    """DC point, charge-control f_T, then AC f_T and f_max at frequencies a fraction of the charge-control f_T
    (inside the -20 dB/decade range, below the non-quasi-static roll-off)."""
    from simws.tcad.hbt import HBTCase, build
    case = case or HBTCase()
    build(case, circuit=True)
    cb = mobility_callback(case)
    bias(VBE, VCE, opts, cb=cb)
    if cb:
        cb(VBE)
    dc = dc_currents()
    cc = charge_control(VBE, VCE, opts=opts, cb=cb)
    f_lo = 1e-4 * cc["fT_charge_Hz"]
    beta0 = abs(figures(y_params(f_lo), f_lo)["h21"])
    ac = [figures(y_params(r * cc["fT_charge_Hz"]), r * cc["fT_charge_Hz"], beta0) for r in decades]
    fT_ac = ac[1]["fT_single_pole"]
    return {"VBE": VBE, "VCE": VCE, "dc_A_per_cm": dc, "beta": dc["collector"] / dc["base"], "beta0_ac": beta0,
            "charge": cc, "ac": ac, "fT_ac_Hz": fT_ac, "fmax_Hz": ac[1]["fmax_est"],
            "fT_rel_diff": fT_ac / cc["fT_charge_Hz"] - 1}


# ---------------------------------------------------------------------------------------------- noise
# Impedance-field method (Shockley, Copeland & James 1966; van Vliet 1975 for the source strengths): DEVSIM's noise
# solve returns, for an output circuit current, the Green's function G_eq(r) of each continuity equation (response of
# the output to a unit injection at r) and its gradient. Local sources, one-sided spectral densities:
#   diffusion            S_J = 4 q^2 D n (electrons, likewise holes)      -> int |grad G|^2 S_J dV, evaluated on the
#                        edges as the discrete Dirichlet form sum (EdgeCouple / EdgeLength) |dG|^2 S_J (consistent
#                        with the finite-volume operator; node gradients are too coarse in 2D)
#   SRH g-r              S = 2 q^2 (r + g), r + g = (n p + n_ie^2) / (tau_p (n + n1) + tau_n (p + p1))
#                        one event removes an electron and a hole       -> int |G_n - G_p|^2 S dV
# Check: a short pn diode (diffusion only) and one dominated by SRH both give full shot noise 2 q I to 1e-3.

def _grad2(name, region, dims):
    from devsim import get_node_model_values
    s = 0.0
    for part in ("real", "imag"):
        for ax in dims:
            g = np.array(get_node_model_values(device="d", region=region, name=f"{name}_{part}_grad{ax}"))
            s = s + g * g
    return s


def current_noise(output: str, f: float = 1.0, region: str = "si", srh: bool = True) -> dict:
    """One-sided current noise spectral density (A^2/Hz, per cm of depth in 2D) of the circuit current `output`
    (e.g. 'VC.I') at frequency f, split into diffusion and SRH generation-recombination parts."""
    from devsim import get_node_model_list, get_node_model_values, get_parameter, solve
    from simws.units import Q_E
    solve(type="noise", frequency=f, output_node=output)
    g = lambda n: np.array(get_node_model_values(device="d", region=region, name=n))
    have = set(get_node_model_list(device="d", region=region))
    Vt = get_parameter(device="d", region=region, name="V_t")
    vol, n, p = g("NodeVolume"), g("Electrons"), g("Holes")
    from devsim import edge_from_node_model, get_edge_model_values
    e = lambda n: np.array(get_edge_model_values(device="d", region=region, name=n))
    for v in ("node_index",):
        edge_from_node_model(device="d", region=region, node_model=v)
    i0, i1 = e("node_index@n0").astype(int), e("node_index@n1").astype(int)
    w = e("EdgeCouple") / e("EdgeLength")
    # DEVSIM leaves G = 0 on contact nodes (their rows are contact equations). Physically a carrier injected at an
    # ohmic contact leaves through it: G there is +-1 for the output's own contact and 0 for the others, i.e. the
    # nearest of {-1, 0, 1} to the interior neighbours' mean.
    at_c = g("AtContactNode") > 0
    nb_sum, nb_cnt = np.zeros(len(n), complex), np.zeros(len(n))

    def fix_contacts(G):
        nb_sum[:] = 0; nb_cnt[:] = 0
        inner0, inner1 = ~at_c[i0], ~at_c[i1]
        np.add.at(nb_sum, i0[inner1], G[i1[inner1]]); np.add.at(nb_cnt, i0[inner1], 1)
        np.add.at(nb_sum, i1[inner0], G[i0[inner0]]); np.add.at(nb_cnt, i1[inner0], 1)
        G = G.copy()
        m = at_c & (nb_cnt > 0)
        G[m] = np.round((nb_sum[m] / nb_cnt[m]).real)
        return G

    S_diff = 0.0
    for eq, dens, mu in (("ElectronContinuityEquation", n, g("mu_n_node")),
                         ("HoleContinuityEquation", p, g("mu_p_node"))):
        G = fix_contacts(g(f"{output}_{eq}_real") + 1j * g(f"{output}_{eq}_imag"))
        dG2 = np.abs(G[i1] - G[i0]) ** 2
        Dn = Vt * 0.5 * (mu[i0] * dens[i0] + mu[i1] * dens[i1])
        S_diff += float(np.sum(w * dG2 * 4 * Q_E * Q_E * Dn))
    S_gr = 0.0
    if srh and "tau_n" in have:
        nie = g("n_ie")
        rg = (n * p + nie ** 2) / (g("tau_p") * (n + nie) + g("tau_n") * (p + nie))
        Gn = g(f"{output}_ElectronContinuityEquation_real") + 1j * g(f"{output}_ElectronContinuityEquation_imag")
        Gp = g(f"{output}_HoleContinuityEquation_real") + 1j * g(f"{output}_HoleContinuityEquation_imag")
        S_gr = float(np.sum(np.abs(Gn - Gp) ** 2 * 2 * Q_E * Q_E * rg * vol))
    return {"f": f, "S_A2_per_Hz": S_diff + S_gr, "S_diffusion": S_diff, "S_gr": S_gr}


def hbt_noise(case=None, VBE=0.75, VCE=1.0, f=1.0, opts=None) -> dict:
    """Low-frequency collector and base current noise against full shot noise 2 q I (ideal transistor limit)."""
    from simws.tcad.hbt import HBTCase, build
    from simws.units import Q_E
    case = case or HBTCase()
    build(case, circuit=True)
    bias(VBE, VCE, opts)
    dc = dc_currents()
    out = {"VBE": VBE, "VCE": VCE, "dc_A_per_cm": dc}
    for term, src in (("collector", "VC.I"), ("base", "VB.I")):
        r = current_noise(src, f)
        r["ratio_to_2qI"] = r["S_A2_per_Hz"] / (2 * Q_E * abs(dc[term]))
        out[term] = r
    return out
