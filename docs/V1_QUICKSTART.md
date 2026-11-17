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

## Ion implantation (second problem family, engine RustBCA)

An implant is described the same way as a device, with its own description class: a beam, a layered target, and
how many ions to follow. It needs the RustBCA engine once: `bash tools/install_rustbca.sh` (it compiles RustBCA
and adds one package to the `tcad` env; it needs a Rust compiler and tells you how to get one).

```python
from simws.spec.library import implant_si
from simws.engines.planner import plan
from simws.study import Study

p = implant_si("P", energy_keV=100, tilt_deg=7, dose_cm2=1e15)     # 31P into amorphous silicon
print(plan(p).summary())                                           # rustbca runs it; the device engines refuse
sr = Study(hub, p).sweep({"beam.energy_keV": [50.0, 100.0, 150.0]})
print(sr.table("implant.Rp"))
r = hub.result(sr.points[0]["result_id"])
r.q("implant.Rp"), r.q("implant.dRp"), r.q("implant.lateral_straggle")     # metres, each with its error bar
r.q("implant.profile")            # stopping probability per ion and per metre, on the axis "depth"
r.q("implant.concentration")      # atoms per m^3 for the dose given (as implanted, before any anneal)
```

`python tools/v1_implant_demo.py` runs boron, phosphorus and arsenic at 50 and 100 keV and prints the table.

What to keep in mind when reading an implant result:

- The error bars are counting statistics only (flag `budget:statistical_only`). They do not cover the collision
  physics.
- The target is amorphous: no channeling. A crystalline wafer shows a deeper tail than this gives.
- **Phosphorus** ranges are 13 to 18 % deeper than the published 1975 range tables (gate 25 %). **Boron ranges
  come out about 35 % too deep**: this failed its acceptance check and is recorded as a FAIL (contract A7.9);
  boron results carry the flag `implant:known_failure:`. Arsenic has not been checked against anything. Boron
  and arsenic results are flagged `extrapolated:` and `in_range` is False: use them for trends, not for numbers.
- Every implant result is `unverified`: there is no second implant engine yet to compare with.
- No damage or vacancy profiles yet, and no beams above 25 keV per nucleon.

## Single-event energy deposition (third problem family, engine G4SEE)

What does one particle leave in a small sensitive volume (SV) of a chip? The description is a particle source, a
silicon block with an SV inside and optional back-end-of-line layers on top, and a histogram. It needs the G4SEE
engine once: `bash tools/install_g4see.sh` (it makes a separate conda env `g4see` with Geant4 11.2.2 and builds
G4SEE v1.0 there; a few GB of Geant4 data are downloaded the first time).

```python
from simws.spec.library import see_si_slab

p = see_si_slab("alpha", energy_MeV=5.0, sv_thickness_um=10.0)          # 5 MeV alphas, SV 0 to 10 um deep
p = see_si_slab("proton", 10.0, sv_thickness_um=2.0, sv_depth_um=5.0,   # buried SV under two BEOL layers
                beol=[("passivation", "G4_SILICON_DIOXIDE", 1.0), ("metal", "G4_Cu", 0.5)])
r = hub.submit("see.g4see", params={"problem": p})
r.q("see.edep_mean")        # eV, with error bar (counting + histogram binning)
r.q("see.p_above")          # probability per particle of leaving at least the threshold, axis "threshold" (eV)
r.q("see.edep_spectrum")    # probability per eV, axis "edep"
```

`python tools/v1_see_demo.py` runs alphas and protons and compares the mean deposit with the NIST stopping
tables.

What to keep in mind:

- Electromagnetic physics only unless you set `hadronic="ftfp_bert_hp"` (flag `see:em_only`): without it, no
  nuclear reactions, so neutrons deposit nothing.
- Energy, not charge: dividing by the electron-hole pair energy of silicon is left to you.
- The checks so far (alphas and protons, 5 to 10 MeV, silicon) test the set-up against the NIST tables, which
  Geant4 shares; every result is `unverified` because there is no second, independent single-event code yet.

## What is not in v1

- DEVSIM translator limits: one silicon region (with Ge grading), ohmic contacts; equilibrium, DC sweeps and
  two-port small-signal analysis. Noise, strikes, Schottky contacts, MOS and SiC are still only in the legacy
  modules. Mesh error bars exist for DC results, not for AC.
- Charon runs one case: the 1D abrupt pn diode.
- Implantation and single events: see the limits above.
- The 23 workflow scripts that bypass the hub are unchanged.
