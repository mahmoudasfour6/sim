"""First real run of the Charon translator (laptop only): the same pn diode on DEVSIM and on Charon through the
engine-neutral description, compared with the tolerance of the 0.9a cross-check (2 % where |J| > 1e-6 A/cm^2).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_charon_check.py [workdir]

The Charon execute/collect path was written without Charon available and has never run. Expect to debug it here.
This script changes nothing in the store: it calls the engine modules directly.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

GATE, J_FLOOR = 0.02, 1e-6          # simws.validation.charon_xcheck (fixed for 0.9a)


def problem(T: float = 300.0):
    from simws.spec.library import pn_diode_1d
    V = [round(0.1 * k, 3) for k in range(0, 8)]
    p = pn_diode_1d(NA_cm3=1e17, ND_cm3=1e17, junction_um=0.5, length_um=1.0, mu_n=1000.0, mu_p=400.0, tau_s=1e-7,
                    T=T, nodes=801, V=V)
    return p.with_changes({"models.statistics": "fermi_dirac", "models.incomplete_ionization": "constant_energy",
                           "models.ii_energy_eV": 0.045, "models.ii_critical_doping_cm3": 1e18})


def main(argv) -> int:
    from simws import engines
    work = Path(argv[1]) if len(argv) > 1 else Path(tempfile.mkdtemp(prefix="v1_charon_"))
    p = problem()
    for name in ("devsim", "charon"):
        r = engines.unsupported(name, p)
        if r:
            print(f"{name} refuses the problem: {r}")
            return 2
    ch = engines.module("charon")
    if not ch.available():
        print("Charon is not installed on this machine (config toolchain.charon_exe). Decks only:")
        for f in ch.prepare(p, work):
            print("  ", f)
        return 3
    a = engines.module("devsim").run(p)
    b = ch.run(p, work)
    Ja, Jb = np.array(a.currents["anode"]), np.array(b.currents["anode"])
    worst = 0.0
    for v, x, y in zip(a.sweep_V, Ja, Jb):
        rel = abs(x / y - 1) if abs(y) > J_FLOOR else float("nan")
        worst = max(worst, 0.0 if rel != rel else rel)
        print(f"V = {v:5.2f}  devsim {x: .6e}  charon {y: .6e}  rel {rel:.3e}")
    print(f"worst relative difference where |J| > {J_FLOOR:g} A/cm^2: {worst:.3e}  (gate {GATE})  "
          f"{'PASS' if worst <= GATE else 'FAIL'}")
    print("charon flags:", b.flags, "| workdir:", work)
    return 0 if worst <= GATE else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
