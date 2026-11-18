"""Circuits through the hub, and the first device-to-circuit link (contract amendment A9; needs the circuit
engine: tools/install_spice.sh).

1. An RC low-pass (AC and step response), a divider and a diode, each against its closed form.
2. The link: the DEVSIM pn diode is solved, an ideal (Shockley) diode is fitted to its I-V, and that diode is
   run in ngspice. The last column says how far an ideal diode is from the TCAD device (model-form error).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_circuit_demo.py [--store DIR] [--inprocess]
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

K_B, Q_E = 1.380649e-23, 1.602176634e-19


def main(argv) -> int:
    from simws import catalogue, engines
    from simws.chain.diode import circuit_from_fit, fit_shockley
    from simws.hub.jobs import Hub
    from simws.spec.library import diode_iv, divider, pn_diode_1d, rc_lowpass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="hub store folder (default: the store of this checkout)")
    ap.add_argument("--inprocess", action="store_true", help="run the adapters in this Python")
    args = ap.parse_args(argv[1:])
    if not engines.module("ngspice").available():
        print("ngspice (with OSDI) is not installed: bash tools/install_spice.sh")
        return 2
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)

    def run(p):
        r = hub.submit("circuit.ngspice", params={"problem": p})
        if r.status != "ok":
            raise SystemExit(f"{p.title}: status {r.status}: {(r.message or '')[-400:]}")
        return r

    print("\n1. circuits against closed forms (worst difference over all points)")
    r = run(divider())
    print(f"   divider        v(mid) = {r.q('circuit.v.mid').value:.12f} V        (exact 0.75)")
    R, C = 1e3, 1e-9
    r = run(rc_lowpass())
    f = np.asarray(r.q("circuit.v.out.mag").axes[0].values)
    w = 2 * math.pi * f * R * C
    e_mag = np.max(np.abs(r.q("circuit.v.out.mag").array() * np.sqrt(1 + w ** 2) - 1))
    e_ph = np.max(np.abs(r.q("circuit.v.out.phase").array() + np.arctan(w)))
    print(f"   RC, AC         |H| {e_mag:.1e} relative, phase {e_ph:.1e} rad   ({len(f)} frequencies)")
    r = run(rc_lowpass(analysis="tran"))
    t = np.asarray(r.q("circuit.v.out").axes[0].values)
    e_t = np.max(np.abs(r.q("circuit.v.out").array() - (1 - np.exp(-t / (R * C)))))
    print(f"   RC, step       {e_t:.1e} V                              ({len(t)} time points)")
    for model in ("builtin", "va"):
        p = diode_iv(model=model)
        r = run(p)
        V = np.asarray(r.q("circuit.i.vin").axes[0].values)
        ref = 1e-14 * (np.exp(Q_E * V / (K_B * 300.15)) - 1)
        e = np.max(np.abs(-r.q("circuit.i.vin").array() / ref - 1))
        print(f"   diode ({model:7s}) {e:.1e} relative" + ("  (Verilog-A, compiled by OpenVAF)" if model == "va" else ""))

    print("\n2. device to circuit: DEVSIM pn diode -> ideal diode -> ngspice")
    V = [round(0.30 + 0.05 * k, 2) for k in range(7)]
    d = hub.submit("device.devsim", params={"problem": pn_diode_1d(V=V)})
    if d.status != "ok":
        raise SystemExit(f"DEVSIM diode: status {d.status}: {(d.message or '')[-300:]}")
    J = d.q("dc.I.anode").array()
    fit = fit_shockley(V, J, 300.0)
    area = 1e-12
    c = run(circuit_from_fit(fit, area_m2=area, start=0.30, stop=0.60, step=0.05))
    I = -c.q("circuit.i.vin").array()
    print(f"   fit: Js = {fit['Js_A_m2']:.4e} A/m^2, N = {fit['N']:.4f}  (area {area:g} m^2, T = 300 K)")
    print(f"   {'V':>5s}  {'DEVSIM J A/m^2':>15s}  {'ngspice I/area':>15s}  {'difference':>10s}")
    for v, j, i in zip(V, J, I):
        print(f"   {v:5.2f}  {j:15.6e}  {i / area:15.6e}  {100 * (i / area / j - 1):+9.3f}%")
    print("   the difference is how far an ideal diode is from the drift-diffusion device (model-form error).")
    print("\ntrust: circuit results are 'unverified' (no second circuit engine yet; contract A9.7).\n")
    print(catalogue.table(catalogue.rows(hub, adapter="circuit.*")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
