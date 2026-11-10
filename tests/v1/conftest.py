"""Shared fixtures for the architecture-v1 tests (written from docs/ARCHITECTURE_V1.md and simws/spec only)."""
import os
from pathlib import Path

import pytest

# DEVSIM on this build machine needs explicit LAPACK/BLAS paths; on the laptop DEVSIM finds its own libraries, so
# the variable is only set when these exact files exist (and never overrides a value the shell already set).
_LIBS = ["/usr/lib/x86_64-linux-gnu/liblapack.so.3", "/usr/lib/x86_64-linux-gnu/libblas.so.3"]
if all(Path(p).exists() for p in _LIBS):
    os.environ.setdefault("DEVSIM_MATH_LIBS", ":".join(_LIBS))


@pytest.fixture
def hub(tmp_path):
    from simws.hub.jobs import Hub
    return Hub(tmp_path / "store", mode="inprocess")


def make_charon_problem(V=(0.0, 0.1, 0.2), T=300.0, NA=1e17, ND=1e17, mu_n=1000.0, mu_p=400.0, tau=1e-7,
                        E_ion=0.045, N_crit=1e18, nodes=801):
    """A pn diode that satisfies the Charon capability sheet of contract section 4.5: 1.0 um long, junction at
    0.5 um, Fermi-Dirac, constant mobility, constant equal lifetimes, constant-energy incomplete ionization,
    band-gap narrowing and Auger off (already off in pn_diode_1d), DC sweep of the anode (x_min) only."""
    from simws.spec.library import pn_diode_1d
    p = pn_diode_1d(NA_cm3=NA, ND_cm3=ND, junction_um=0.5, length_um=1.0, mu_n=mu_n, mu_p=mu_p, tau_s=tau, T=T,
                    nodes=nodes, V=list(V))
    return p.with_changes({"models.statistics": "fermi_dirac",
                           "models.incomplete_ionization": "constant_energy",
                           "models.ii_energy_eV": E_ion,
                           "models.ii_critical_doping_cm3": N_crit})


@pytest.fixture
def charon_problem():
    return make_charon_problem


def kirchhoff_ok(currents: dict, k: int, floor: float) -> bool:
    """Contract section 5 Kirchhoff rule: |sum over contacts| <= 1e-4 x max|I| + floor at sweep index k."""
    vals = [c[k] for c in currents.values()]
    return abs(sum(vals)) <= 1e-4 * max(abs(v) for v in vals) + floor


@pytest.fixture
def kirchhoff():
    return kirchhoff_ok


# ------------------------------------------------------------------------------------------------ fake adapters (A2.3)
# Small adapters with chosen engine names, registered for one test only, so that compare_results sees two stored
# results from DIFFERENT (or the SAME) engines without running a real engine. Their output is y = scale * (1 + t)
# on t = 0, 1, 2 s; it is test data, not physics.
FAKE_ADAPTERS = {"fake.alpha": "fake_alpha", "fake.alpha_twin": "fake_alpha", "fake.beta": "fake_beta"}


def _fake_module():
    import types

    from simws.hub.adapter import Adapter
    from simws.hub.schema import Manifest, OutputSpec

    mod = types.ModuleType("simws_v1_fake_adapters")

    def make(name, engine):
        class _Fake(Adapter):
            manifest = Manifest(name=name, version="1.0", env="v1_fake", engine=engine, engine_version="1",
                                inputs=["scale"], outputs=[OutputSpec(name="y", unit="V")])

            def run(self, job, ctx):
                return float(ctx.param("scale"))

            def collect(self, job, ctx, raw):
                xs = [0.0, 1.0, 2.0]
                return [ctx.quantity("y", [raw * (1.0 + x) for x in xs], "V",
                                     axes=[{"name": "t", "unit": "s", "values": xs}])]
        return _Fake

    for i, (name, engine) in enumerate(FAKE_ADAPTERS.items()):
        setattr(mod, f"Fake{i}", make(name, engine))
    return mod


@pytest.fixture
def fake_adapters(monkeypatch):
    """Registers fake.alpha (engine fake_alpha), fake.alpha_twin (engine fake_alpha) and fake.beta (engine
    fake_beta); returns submit(hub, adapter, scale) -> ResultEnvelope."""
    import sys

    from simws.hub import adapter as A
    A.names()                                            # load the built-in registry first
    mod = _fake_module()
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    for i, name in enumerate(FAKE_ADAPTERS):
        monkeypatch.setitem(A.REGISTRY, name, f"{mod.__name__}:Fake{i}")

    def submit(hub, name, scale):
        r = hub.submit(name, params={"scale": scale})
        assert r.status == "ok", r.message
        return r
    return submit
