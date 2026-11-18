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
- No beams above 25 keV per nucleon, and no channeling model (A15: no openly published parameters were found).

### Measured stopping, a second engine, damage (contracts A12, A13, A14)

```python
p = implant_si("B", 50).with_changes({"models.electronic_stopping": "empirical"})   # A12: LS x 1.38 for boron
r1 = hub.submit("implant.rustbca", params={"problem": implant_si("P", 50)})
r2 = hub.submit("implant.g4implant", params={"problem": implant_si("P", 50)})       # Geant4, different code
from simws import compare
compare.record(hub, compare.compare_results(hub, r1.id, "implant.Rp", r2.id, tolerance=0.10))
d = hub.submit("implant.rustbca", params={"problem": implant_si("B", 50, n_ions=2000, displacement_eV=36.0)})
d.q("implant.vacancies_per_ion"), d.q("implant.vacancy_profile"), d.q("implant.vacancy_depth_mean")
```

- "empirical" stopping (A12) exists for boron in silicon only: Lindhard-Scharff times 1.381 +- 0.195 from the
  measured points of the IAEA stopping database. The measured sets disagree with each other (Birge ratio 10.9).
- `g4implant` (A13) is Geant4 with G4ScreenedNuclearRecoil; build it with `bash tools/install_g4implant.sh`. In
  Lindhard-Scharff mode it uses RustBCA's stopping formula, so a comparison tests the collision codes. Result:
  boron agrees within 6 to 10 %, phosphorus differs by 13 to 16 % (recorded FAIL). No implant result is
  `verified` today: boron in Lindhard-Scharff mode is a known failure (agreement on a wrong model does not count),
  and phosphorus failed the comparison. Geant4's "empirical" stopping missed the 1975 tables by 21 to 32 %.
- Damage (A14) needs `displacement_eV`; for Geant4 use at least 30 eV (below that its count barely follows Ed,
  A14.7). There is no default (silicon: 12.5 eV minimum, 36 eV direction average,
  Holmstrom et al. 2008; 15 eV is common in BCA work). Counts are full cascades without recombination: an upper
  bound. RustBCA runs damage through its command-line program in 20 batches (memory stays below a few hundred MB).
- `python tools/v1_implant_verify_demo.py` runs both engines on boron and phosphorus and the damage table.

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

## Circuits (fourth problem family, engine ngspice with OpenVAF)

A circuit is a netlist (resistors, capacitors, sources, diodes, transistors, Verilog-A models, foundry model
libraries), one analysis (op, dc, ac or tran) and the probes to report. It needs the circuit engine once:
`bash tools/install_spice.sh` (a separate conda env `spice` with ngspice 47 built with OSDI, and OpenVAF; the
ngspice on conda-forge cannot load OpenVAF models).

```python
from simws.spec.library import rc_lowpass, diode_iv, SHOCKLEY_VA
r = hub.submit("circuit.ngspice", params={"problem": rc_lowpass()})        # AC of an RC low-pass
r.q("circuit.v.out.mag"), r.q("circuit.v.out.phase")                       # axis "f" in Hz
r = hub.submit("circuit.ngspice", params={"problem": diode_iv(model="va")}) # a Verilog-A diode via OpenVAF
r.q("circuit.i.vin")                     # SPICE sign: the current entering the source at its first node
```

Device to circuit: `simws.chain.diode.fit_shockley` fits an ideal diode to a DEVSIM I-V and
`circuit_from_fit` makes the matching ngspice problem; `python tools/v1_circuit_demo.py` runs both and shows
how far the ideal diode is from the TCAD device.

What to keep in mind: no uncertainty is attached to circuit results (flag `budget:none`); results that use a
foundry library carry its file's SHA-256 in a flag, because the library is not part of the description; every
circuit result is `unverified` (no second circuit engine yet).

## Single event to circuit (contract A10)

`simws.chain.see` turns deposited energy into charge (3.66 eV per pair in silicon, Scholze et al. 1998), charge
into a current pulse (Messenger's double exponential, with time constants you supply), finds a node's critical
charge by bisection over ngspice runs, and reads from a G4SEE result the probability that one particle deposits
at least that charge.

```python
from simws.chain import see as C
qc = C.critical_charge(hub, circuit, "n", threshold_V=0.5, tau_rise_s=1e-12, tau_fall_s=10e-12,
                       q_low_C=5e-15, q_high_C=3e-13)               # circuit: a transient CircuitProblem
up = C.upset_probability(hub.result(see_result_id), qc)             # P, its bracket P_low..P_high, sigma
```

`python tools/v1_upset_demo.py` runs 5 MeV alphas against three nodes. All collected charge goes to one node and
the pulse shape is assumed, so treat P as an estimate of the mechanism, not of a real chip.

## Energy balance: a different-physics check (contract A11)

`device.devsim_eb` solves the same 1D Problem with an electron energy balance (Baccarani-Wordeman 1985): hot
electrons, their falling mobility and velocity saturation. It runs on DEVSIM too, so comparing it with
`device.devsim` tells you how much a drift-diffusion result depends on that physics; it never makes either
result "verified". `python tools/v1_energy_balance_demo.py` compares the two on a bar and on the pn diode.

## What is not in v1

- DEVSIM translator limits: one silicon region (with Ge grading), ohmic contacts; equilibrium, DC sweeps and
  two-port small-signal analysis. Noise, strikes, Schottky contacts, MOS and SiC are still only in the legacy
  modules. Mesh error bars exist for DC results, not for AC.
- Charon runs one case: the 1D abrupt pn diode.
- Implantation, single events and circuits: see the limits above.
- The 23 workflow scripts that bypass the hub are unchanged.
