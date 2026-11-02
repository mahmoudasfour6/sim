# simws: running everything locally

Everything below runs on this laptop (WSL2, conda envs under `~/miniforge3/envs`). Nothing needs a cluster,
container or cloud CI.

## Daily commands

```bash
cd ~/sim_workspace
W=~/miniforge3/envs/workflow/bin
$W/snakemake -n                                   # what would run
$W/snakemake --cores 12 --resources mem_mb=20000  # core results (cached; only changed stages rerun)
$W/snakemake --cores 12 --resources mem_mb=20000 studies   # Sobol sweep + convergence studies
$W/python tools/summarize_results.py              # one-screen summary of results/
$W/pytest -q            # fast tier (seconds)
$W/pytest -m slow -q    # every solver against golden values (~30-60 min)
```

`config/local_tuning.yaml` (written by `tools/bench_local.py`) sets the QE rank count and sweep worker count the
workflow uses. Re-run the benchmark after changing the WSL memory/CPU limits (`%UserProfile%\.wslconfig`) or
the power plan: on battery the Core Ultra 5 125U throttles and the best worker count drops.

## What each piece is for

| Module | Use | Checked by |
|---|---|---|
| `simws.tcad.models` | velocity saturation (Caughey-Thomas), impact ionization (Van Overstraeten-de Man) in DEVSIM | `validate_models` rule: closed-form mobility, ionization-integral breakdown voltage |
| `simws.tcad.solve` | `ramp_bias` / `ramp_source`: adaptive bias stepping that backs off on Newton failure | unit + slow tests |
| `simws.radiation.geant4` | Geant4 ion track -> (depth, radius) deposition, LET profile | `geant4_track` rule vs G4EmCalculator |
| `simws.radiation.srim` | parse SRIM SR-Module tables (run SRIM natively on Windows), LET along the path (CSDA) | unit tests |
| `simws.radiation.track` | carrier generation from a track for 2D DEVSIM (charge conserved on the mesh) | unit tests |
| `simws.tcad.mixed_mode` | device + SPICE-like circuit in one Newton solve; `rc_check`, `seu_diode` | RC analytic check, SEU slow test |
| `simws.sweep` | grid / LHS / Sobol designs, parallel local runs, SQLite store with resume, Dakota deck/driver | unit tests (Ishigami-type check) |
| `simws.verify` | refinement studies: observed order, Richardson, GCI, noise-level detection | `convergence` rule |

## simws 0.3: device physics, axisymmetric strikes, calibration

```python
from simws.tcad.physics import Physics, setup_region          # full drift-diffusion for any DEVSIM region
setup_region(dev, "si", mat, net_doping="...", total_doping="...",
             phys=Physics(T=350.0, field_mobility=True, impact=False),
             contacts={"anode": False, "cathode": False})       # name -> is_circuit
```
Models: Masetti doping-dependent mobility, Scharfetter SRH lifetime, Dziewior-Schmid Auger, Slotboom band-gap
narrowing (Klaassen 1992) applied consistently to the currents, Caughey-Thomas velocity saturation with
v_sat(T), Van Overstraeten-de Man impact ionization with its temperature factor, n_i(T). All parameters and
references are in `simws/materials/data/silicon.yaml`; `load("silicon").provisional()` lists the ones still to
calibrate. Checks: `results/models/physics_checks.json` (equilibrium current 1e-16 of the flux scale, mobility
to 1e-13, lifetimes to 3e-6, n_i(T) within 7 % of the Misiakos-Tsamakis measurement up to 400 K).

Axisymmetric (r-z) strike, absolute units (A, fC), realistic doping:
```python
from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric
r = seu_axisymmetric(AxiSEUCase(let=10.0, ND=1e20, NA=1e17, V_drain=1.0))
r["collected_charge_drain_fC"], r["peak_drain_current_A"]
```

Calibration to your measurements (CSV with columns V,I and/or V,C):
```python
from simws.calibrate import Calibration, DiodeSpec, Parameter, diode_model, load_measurement
data = load_measurement("measured/D3.csv")
spec = DiodeSpec(area_cm2=1e-4, p_plus=1e19, base_thickness_um=300)
cal = Calibration([Parameter("N_base", 1e15, 1e17, log=True), Parameter("srh_tau_max_p", 1e-8, 1e-3, log=True)],
                  lambda p: diode_model(spec, p, data["iv"]["V"], data["cv"]["V"]), data, {"iv": 0.05, "cv": 0.02})
fit = cal.fit(); print(fit.table())          # best values, 95 % intervals, correlations, chi2/dof
```
Run `results/calibration/selftest.json` first: it proves the fit recovers known parameters from noisy synthetic
data. Any material parameter in silicon.yaml can be a fit parameter; DiodeSpec fields describe the device.

Off-laptop backup: create an empty private repository, then `bash tools/setup_offsite_backup.sh <URL>`; the
nightly job pushes to it afterwards. MKL PARDISO is opt-in (`SIMWS_DEVSIM_SOLVER=mkl`, needs `mkl` in tcad):
in tests here it failed on the electrothermal device and did not fix the mixed-mode convergence, so UMFPACK
stays the default.

## simws 0.4: event-by-event strikes, upset rates, figures, releases, study log

```bash
W=~/miniforge3/envs/workflow/bin
$W/snakemake paper --cores 12 --resources mem_mb=20000     # everything + figures -> results/figures
```
- **Geant4 events -> DEVSIM** (`geant4_events`, `event_seu` rules): each simulated ion's own (depth, radius) dose
  is remapped conservatively onto the r-z mesh and drives its own strike; a Gaussian-track run at the same mean
  LET is the reference. Ions and counts: `event_ions`, `event_count`, `event_seu_count` in config/config.yaml.
- **Upset rate** (`ser` rule, `simws.radiation.ser`): critical LET from the axisymmetric LET sweep and your
  critical charge (`seu_qcrit_fC`), step cross section over the drain area, isotropic integration over a LET
  spectrum. No environment is built in: run CREME96 (or OMERE/SPENVIS) for your orbit and shielding, save the
  LET spectrum, and set `ser_spectrum`, `ser_let_unit`, `ser_flux_unit` (check the file header for units).
  The engine is checked against a closed form to 0.1 % on every run.
- **Figures** (`paper` rule, `tools/make_figures.py`): PDF + 300 dpi PNG + a JSON sidecar with the plotted
  numbers and the commit that produced them; `methods_parameters.md/.tex` lists every material parameter with
  value, unit, status and reference.
- **Release for a paper**: commit, then
  `$W/python tools/release.py --title "..." --version 1.0 --author "Asfour, Mahmoud|Sabanci University|<ORCID>"`
  builds `release/<name>.tar.gz` (source at that commit, env lockfiles, results, SHA-256 manifest,
  CITATION.cff, .zenodo.json). Add `--upload --sandbox` (with `ZENODO_TOKEN`) to try it on sandbox.zenodo.org,
  `--upload` for the real draft; the DOI is minted only when you press Publish.
- **Study log**: `python tools/study.py new "question"` starts notes/<date>_<slug>.md (commit recorded);
  `python tools/study.py link <note> results/.../summary.json` appends the result with its provenance.

## Single-event upset example

```python
from simws.tcad.mixed_mode import seu_diode, SEUCase
from simws.radiation import srim
table = srim.load("O_in_Si.txt")                          # SRIM output copied from Windows
r = seu_diode(SEUCase(let=lambda z: table.let_vs_depth(100.0, z), track_length_um=5.0))
print(r["v_node_initial_V"], r["v_node_min_V"], r["collected_charge_substrate_fC"])
```

Limits to keep in mind (also in the docstrings):
- The ion track is planar 2D (a track repeated along the out-of-plane depth W). Good for comparing designs and
  code-to-code checks; absolute SEU cross sections need 3D or axisymmetric geometry.
- With a circuit node attached (mixed mode), keep the substrate at <= 1e16 cm^-3 and the drain at <= 1e19 cm^-3,
  or Newton cycles on minority carriers. Fixed-bias strikes (`simws.tcad.axisym`) run at 1e20 / 1e17.
- Impact ionization is for high-voltage devices; it is off in the SEU case (no effect at logic-level bias).

## Nightly regression

Run `tools/register_nightly.ps1` once in Windows PowerShell. Every night at 02:30 it runs `tools/nightly.sh`
inside WSL (fast tier, slow tier, workflow and studies) and writes `results/nightly/<date>.log`. The newest log
is also copied to `%USERPROFILE%\sim_backup\nightly_latest.log`. The laptop must be plugged in. The task
is allowed to wake it from sleep, but not to start it when it is shut down.

## Sweeps on your own stage

```python
from simws.sweep import Space, Store, run_design
def stage(p):                       # top-level function; returns scalar outputs
    ...
    return {"V_min": ..., "Q_coll": ...}
space = Space({"let": (1, 40), "R_load": (5e3, 5e4)})
run_design(stage, space.lhs(200), Store("results/sweeps/seu.sqlite"), workers=6)
```
Interrupt it at any time; rerunning skips finished points. `store.failures()` lists the points that failed and
their tracebacks.
