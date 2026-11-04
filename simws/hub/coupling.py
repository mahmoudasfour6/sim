"""Coupling registry (interface specification: coupling registry; design-review fix 7).

Every hand-over between two pieces of physics is listed here with its direction and where the coupling happens.
The rule: a TWO-WAY coupling (A changes B and B changes A, iterated to self-consistency) runs inside ONE adapter
process, so solver state and warm starts survive every iteration, or (0.9b) runs as preCICE participants: separate
long-lived processes on their own meshes, iterated by preCICE's implicit coupling with conservative mapping. Only ONE-WAY hand-overs go through the store, where each is a job
boundary with its own key, gatekeeper check and graph edge.

    from simws.hub import coupling
    coupling.check()            # [] when the registry obeys the rule
    coupling.for_adapter("devsim.mos_cv")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Coupling:
    producer: str               # adapter (or "law:<material>.<parameter>" for in-process material laws)
    consumer: str
    quantity: str
    direction: str              # one_way | two_way
    where: str                  # store | in_adapter | precice
    host: Optional[str] = None  # adapter that hosts a two-way loop
    tolerance: Optional[str] = None
    mechanism: str = ""


REGISTRY: List[Coupling] = [
    Coupling("materials.intrinsic", "devsim.hbt_gummel", "mat.n_i", "one_way", "store",
             mechanism="intrinsic density enters every carrier statistic"),
    Coupling("radiation.tid_mos", "devsim.mos_cv", "tid.N_ot", "one_way", "store",
             tolerance="dV_fb within 2e-4 V (tests/connections)", mechanism="trapped oxide charge shifts flat band"),
    Coupling("dft.si_bandedge", "law:silicon.n_i", "dft.Nc_300", "one_way", "store",
             mechanism="DFT densities of states feed n_i(T) through the computed 'dft' profile"),
    Coupling("devsim.strike_hbt", "radiation.ser_rate", "strike.Q_collector", "one_way", "store",
             mechanism="collected charge vs LET sets the critical LET of the cross section"),
    Coupling("simws.coupling.electrothermal", "simws.coupling.electrothermal", "T(x), J(x)", "two_way",
             "in_adapter", host="simws.coupling.electrothermal",
             tolerance="max |dT| < 1e-3 K between iterations", mechanism="Joule heating <-> temperature-dependent mobility"),
    Coupling("devsim.hbt_gummel", "devsim.hbt_gummel", "ionized dopants, mobility(n, p)", "two_way", "in_adapter",
             host="devsim.hbt_gummel", tolerance="Newton (freeze-out) and outer mobility iteration to 1e-8 relative",
             mechanism="0.9a cryogenic physics: carriers set ionization and carrier-carrier scattering"),
    Coupling("simws.coupling.precice_bridge", "simws.coupling.precice_bridge", "Heat (conservative), T (consistent)",
             "two_way", "precice", host="simws.coupling.precice_bridge",
             tolerance="2-norm |dT| < 1e-6 K per window; power transfer conserved to 1e-8",
             mechanism="DEVSIM Joule heating <-> scikit-fem lattice heat on its own gmsh mesh"),
]


def check(registry=None) -> List[str]:
    problems = []
    for c in registry or REGISTRY:
        if c.direction not in ("one_way", "two_way"):
            problems.append(f"{c.producer}->{c.consumer}: unknown direction {c.direction}")
        if c.direction == "two_way" and c.where not in ("in_adapter", "precice"):
            problems.append(f"{c.producer}<->{c.consumer}: two-way couplings must run inside one adapter or under "
                            "preCICE, not via the store")
        if c.direction == "two_way" and not c.host:
            problems.append(f"{c.producer}<->{c.consumer}: two-way coupling needs a host adapter")
        if c.where not in ("store", "in_adapter", "precice"):
            problems.append(f"{c.producer}->{c.consumer}: unknown location {c.where}")
    return problems


def for_adapter(name: str) -> List[Coupling]:
    return [c for c in REGISTRY if name in (c.producer, c.consumer, c.host)]
