# simws architecture v1: quick start

v1 lets you describe a device once and run it on any engine that supports it. Nothing old was removed: the legacy
adapters (`devsim.hbt_gummel` and the rest) still work and are the reference the new path is checked against.
The full contract, with every rule and the recorded amendments, is `docs/ARCHITECTURE_V1.md`.

## Describe a problem

```python
from simws.spec.library import pn_diode_1d, npn_hbt_2d

p = pn_diode_1d(V=[0.3, 0.4, 0.5])                 # a ready-made description
h = npn_hbt_2d(VBE=[0.5, 0.6, 0.7], ge_peak=0.15)   # flagship A device, non-cryogenic settings
cold = h.with_changes({"models.temperature_K": 250.0, "device.mesh.scale": 0.7})
```

A description is plain data (`p.model_dump(mode="json")`); a typo or a missing field fails at once. New devices are
written with `DeviceSpec`, `ModelSpec`, `DCSweep` from `simws.spec` (see `simws/spec/library.py` for two examples).
Units: micrometres, cm^-3, V, K.

## Ask which engines can run it

```python
from simws.engines.planner import plan
print(plan(h).summary())      # devsim runs it; charon refuses, with the reasons
```

Each engine has a capability sheet (`simws.engines.capability("devsim")`). A problem outside an engine's tested
temperature range still runs and is flagged `extrapolated:...`.

## Run it through the hub

```python
from simws.hub.jobs import Hub
hub = Hub()
res = hub.submit("device.devsim", params={"problem": p.model_dump(mode="json")})
res.q("dc.I.anode").value      # A/m^2 in 1D, A/m in 2D, positive into the device
```

Results are cached by content, like every hub job. `device.<engine>` adapters are screening fidelity for now (no
mesh error bars); paper-fidelity numbers still come from the legacy adapters.

## Sweep, store a measurement, compare, and read the trust label

```python
from simws.study import Study
from simws import measured, compare, catalogue
from simws.hub import trust

sr = Study(hub, p, engines=["devsim", "charon"]).sweep({"models.temperature_K": [300.0, 350.0]})
print(sr.table("dc.I.anode"))                       # refused engines appear with their reasons

m = measured.ingest(hub, "my_diode_300K", axis=("V_anode", V_meas, "V"),
                    columns={"dc.I.anode": (J_meas, "A/m**2")},
                    meta={"source": "citation of the paper or lab notebook", "digitized": True})
c = compare.compare_results(hub, sr.points[0]["result_id"], "dc.I.anode", m.id, metric="ratio", tolerance=1.3)
compare.record(hub, c)
print(trust.assess_stored(hub, sr.points[0]["result_id"]))
print(catalogue.table(catalogue.rows(hub, adapter="device.*", temperature=(250, 400))))
```

Trust levels: `sketch` (exploration), `unverified`, `verified` (code verification, or agreement with a DIFFERENT
engine: two runs of one engine verify nothing), `validated` (matches measured data ingested with
`label="validation"`, and no measurement comparison failed; matching `label="calibration"` data does not count),
`measured` (the data itself). Curves are compared with log interpolation whenever the reference is of one sign,
because currents are exponential in bias.
`in_range` is False when the result is outside what its engine or models were tested for. `measured.ingest` refuses
data without a source.

## What is not in v1

- Charon: the translator writes its input decks, but running Charon and reading its output has never been executed.
  First test: `tools/v1_charon_check.py` on the laptop.
- DEVSIM translator limits: one silicon region (with Ge grading), ohmic contacts, equilibrium and DC sweeps. AC,
  noise, strikes, Schottky contacts, MOS, SiC and mesh error bars are still only in the legacy modules.
- The 23 workflow scripts that bypass the hub are unchanged.
