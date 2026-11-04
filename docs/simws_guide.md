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

## simws 0.5: MOS and total dose, process variation, 4H-SiC

- **MOS capacitor** (`simws.tcad.mos`): gate / SiO2 / Si in 1D, quasi-static C-V, fixed interface charge `Q_f`,
  oxide sheets at any depth (`oxide=[(nm_from_Si, cm^-2)]`), uniform interface traps `D_it` (neutral at
  midgap). Solved with equilibrium carriers (no DC current flows, so this is exact for quasi-static C-V).
  Checked against closed forms: flat band to 2e-9 V, Debye capacitance 2e-7, charge shifts 1e-9, D_it 1e-3.
- **Total ionizing dose** (`simws.radiation.tid`): g0 = 8.1e12 pairs/(cm3 rad), field-dependent charge yield
  (Dozier 1987), saturating oxide-trap buildup, hydrogen-model interface traps; `dose_response` re-solves
  the device per dose and reports dV_fb, dV_midgap (oxide charge only) and dV at inversion. f_ot, N_sat and the
  hydrogen-model set are technology dependent: calibrate them to your oxide before quoting absolute shifts.
- **SRH trap level**: `Physics(trap_level_eV=E_t - E_i)` for deep-level traps (implementation exact to 1e-15).
- **Process variation** (`simws.surrogate.PCE`): Latin-hypercube design in the inputs' own distributions
  (normal or uniform), polynomial-chaos fit with leave-one-out Q2, Sobol indices from the coefficients, and
  10^6-sample distributions (3-sigma quantiles). `mos_variability` rule: 100 MOS runs -> V_fb, V_inv, C_min.
- **4H-SiC** (`simws.tcad.sic`): band gap and DOS masses (Ayalew 2004), nitrogen hexagonal/cubic and
  aluminium incomplete ionization, Caughey-Thomas mobility with anisotropy along the c-axis, Hatakeyama 2004
  impact ionization along/across c. Leakage in SiC is ~1e-30 A: probe breakdown with injected carriers.

## simws 0.6: SiGe

- **Material** (`sige.yaml`, `Physics(ge="<x expression>", ge_mode="strained"|"relaxed")`): any Ge profile, as a
  constant or an expression of x/y. Strained on Si(001): gap reduction 0.75x eV, all in the valence band
  (conduction offset taken as 0, provisional), Nc ratio 2/3 and (NcNv) ratio exp(-sqrt(5x)). Relaxed: the
  Ioffe gap fit and the Rieger-Vogl valence offset. Permittivity interpolated to Ge. Mobility factors are 1.0
  (provisional): calibrate them with measured Gummel data. The drift-diffusion core now carries general band
  edges (dEc, dEv), so heterojunctions are exact at equilibrium (n p = n_ie^2 to 4e-14, no current).
- **SiGe HBT** (`simws.tcad.hbt`): 2D npn with emitter, intrinsic/extrinsic base, collector and subcollector;
  `gummel`, `early_voltage`; `ge_peak=0` gives the identical Si BJT. Checked: the SiGe/Si collector-current
  ratio equals the generalized Moll-Ross (Kroemer) integral on the device's own solution to 0.05 %, and so
  does the ideality (graded base: about 62 mV/dec, above 59.5). Graded Ge raises I_C about 6x at 0.6 V with the
  same base current, and the Early voltage about 4x (21 -> 88 V for the default device).
- **Buried strained-SiGe pMOS channel** (`MOSCase(cap_nm, well_nm, ge_x)`, `hole_sheets`): holes fill the SiGe
  well first; past about -1.6 V the Si cap surface takes over (two-step C-V). Checked: the hole onset moves by
  0.13 V for x = 0.25 against an effective valence offset of 0.17 eV, with the well holding 70x the cap charge.
- **HBT total dose** (`simws.radiation.hbt_rad.excess_base_current`, `dose_response`): interface traps over
  the emitter-base periphery as surface recombination (s = sigma v_th N_it) plus the oxide charge there. The
  excess base current equals the surface recombination integral to 3e-4, is linear in N_it, has ideality
  about 1.5 (neutral-base and depletion parts); I_C does not change. The dose mapping uses the MOS
  gate-oxide TID parameters (no ELDRS, no annealing): calibrate before quoting gain after dose.
- **HBT ion strike** (`strike`): planar 2D track through the emitter, collector at 2 V. Charge is conserved
  to 1e-5 and linear in LET at low LET (83 % collected in 2 ns). From about 3 MeV cm2/mg the track
  shorts emitter to collector and the collected charge exceeds the deposited charge (ion-shunt effect). The
  planar track exaggerates this. No substrate junction is modelled.
- Rules: `sige_checks` (in `all`) and `sige_devices` (in `studies`); figures `fig_sige_gummel`,
  `fig_sige_buried_channel`, `fig_hbt_tid`, `fig_hbt_strike`.

## simws 0.7: material parameters from first principles

- **Quantum ESPRESSO driver** (`simws.dft.qe`): writes pw.x inputs (scf, bands, relax, spin-orbit, hybrids),
  runs them (MPI as the qe_si_scf rule), reads the XML. Equation of state with a SciPy-free Birch-Murnaghan fit.
- **Band edges** (`simws.dft.bandedge`): conduction minimum and m_l, m_t; heavy/light/split-off hole masses
  with spin-orbit along [100], [110], [111] and their DOS averages; Delta_so; N_c(T) and N_v(T) summed directly
  over DFT band energies, so non-parabolicity is kept (checked against parabolic bands to 1 %); HSE06 gap as
  a scissor on PBE (EXX q-mesh = k-mesh; a coarser q-mesh over-corrects by 0.4 eV).
- **Elastic constants and deformation potentials** (`simws.dft.strain`): C11, C12, C44 (Kleinman relaxed) at
  the DFT lattice constant, checked against the equation-of-state bulk modulus; Xi_u and b (spin-orbit).
- **Ensemble and uncertainty** (`simws.studies.dft_params`, rule `dft_params` in `studies`): LDA, PBE and
  PBEsol; value = mean, sigma = functional spread combined with convergence variants. Written as the computed
  material profile `results/dft/silicon.dft.yaml`: `load("silicon", profile="dft", computed_dir="results/dft")`;
  after review, `python tools/promote_computed.py results/dft/silicon.dft.yaml --yes` puts it in the git-tracked
  database. Device code uses it through `intrinsic_density` (DFT N_c(T), N_v(T) with the measured gap).
  The base literature set is never changed.
- **Checks** (`results/dft/checks.json`): every value against its measurement with tolerances fixed in
  advance (`simws.validation.dft_checks`). Reference run: m_l +3.9 %, m_t +0.5 %, hole masses -1 to -9 %,
  Delta_so 48 vs 44 meV, C_ij 7-9 % soft, Xi_u 8.9 eV, b -2.32 eV, HSE gap 1.165 eV (1.170 measured; its sigma
  covers only the scissor spread: add a 6x6x6 grid for convergence), n_i(T) 8 % below the measured fit with the
  right temperature slope (the T^1.5 model drifts by 11 % between 250 and 400 K).
- **Pseudopotentials**: the validated norm-conserving set ships in `simws/dft/pseudo` (QE distribution, GPL-2).
  Use your own through `dft.pseudo_dir` / `dft.pseudos` in config.yaml, then re-check.
- **CHGNet benchmark** (`simws.dft.mlip`): SiGe alloy lattice constants (random 64-atom cells) and elastic
  constants against DFT and experiment. Result here: too soft (Si C_ij 20-40 % low, Ge unusable) and the wrong
  sign of the SiGe bowing, so nothing from CHGNet enters the database. Re-run after fine-tuning.
- **SiGe critical thickness**: Matthews-Blakeslee and People-Bean with the measured a(x) (Dismukes) and elastic
  constants. The 0.6 HBT base (0-15 % graded, 50 nm) is just past equilibrium (metastable, as production bases
  are), and the buried pMOS well (25 %, 10 nm) sits at the equilibrium limit (t/h_MB = 0.99).
- Not covered yet: Ge band edges (PBE closes the Ge gap; needs a hybrid or GW start), SiGe alloy band offsets
  from supercells, defect levels.

## simws 0.8: the hub (core contract)

Every calculation now has an identity, a home and a history. `notebooks/hub_tour.ipynb` walks through it.

- **Submit through the hub** (`simws.hub.jobs.Hub`): `hub.submit("devsim.hbt_gummel", params={"VBE": (V, "V"),
  "T": (350, "K")})`. The job key (SHA-256 of the canonical inputs: SI base units except eV and MeV cm2/mg,
  adapter and engine version, env lock hash, member, seed) is the job's identity, so 300 K and 0.3 kK are the
  same job and a repeated submit is served from the store. Defaults that change the answer (pseudopotential files
  and their hashes, cut-offs) are pinned into the key at submit time.
- **Adapters** (`simws/adapters`): `materials.intrinsic`, `devsim.hbt_gummel`, `devsim.mos_cv`,
  `devsim.strike_hbt`, `devsim.mms_poisson`, `radiation.tid_mos`, `radiation.ser_rate`, `dft.si_bandedge`. Each
  runs as its own process in its conda env (`python -m simws.run`), converts to the canonical conventions at its
  boundary (passive current sign, tensile stress, engineering shear, dose in Gy(material), ...: `simws.conventions`)
  and runs physics guards (Kirchhoff, mass action, ideality, monotonicity, charge conservation).
- **Gatekeeper** (`simws.hub.gatekeeper`): declared outputs present, units of the right dimension, finite values,
  canonical conventions, fields on disk. Results outside the validity box are kept and flagged `extrapolated`;
  screening results without an uncertainty are flagged `unverified`; paper fidelity refuses them. Every failure
  lands in one category: not_converged, timeout, out_of_memory, invalid_input, outside_validity, engine_error,
  check_failed.
- **Store and knowledge graph** (`store/`, git-ignored): immutable result objects (`objects/<id>/envelope.json`)
  plus `graph.sqlite` with Job, Result and Quantity nodes, produced / consumed / depends_on / supersedes edges and
  an event ledger. `hub.graph.ancestors(...)`, `hub.graph.is_stale(...)`, `hub.graph.to_networkx()`.
- **Uncertainty** (`simws.hub.uncertainty`): member IDs (`functional=PBE|sample=3`) keep ensembles correlated across
  quantities; `budget()` combines numerical (GCI/2), parametric (sample spread) and model-form (range/sqrt 12)
  parts and names the dominant one. **Verification** records (`simws.hub.verification`): solution (refinement
  study, GCI) and code (manufactured solutions, closed forms).
- **Code verification**: the production Poisson-Boltzmann discretisation, driven by a manufactured solution through
  the same `extra_charge` hook the oxide charge uses, converges at order 2.0001 (`simws.tcad.mms`).
- **Upgrade gate** (rule `hub_gate`, slow tests `test_simws_hub_gate_*`): the hub reproduces the 0.7 results it
  wraps to round-off: TID midgap shift at 1 Mrad, SiGe collector enhancement at 0.6 V, HBT strike charge
  (largest relative difference 4e-15) and the PBE electron masses (identical).
- **Material laws** (`simws.materials.laws`): parameters as functions of T with validity boxes (Si gap 0-1000 K,
  mobility 200-500 K, n_i 250-400 K, k 300-600 K); 4H-SiC laws report `validity_unknown` until their fit ranges
  are checked. New sets: `al2o3`, `hfo2` (Robertson 2004, provisional until checked against the primary source)
  and `interfaces.yaml` (Si/SiO2 D_it and Q_f, 4H-SiC/SiO2 D_it, Ni/4H-SiC barrier) as literature
  distributions: `laws.sample_interface(...)`.
- **PseudoDojo**: `tools/fetch_pseudodojo.sh` installs the PseudoDojo v0.4 ONCVPSP set and `dft.pseudo_set:
  pseudodojo` makes it the default (cut-off 48 Ry, the table's high hint). If only the fully relativistic PBE file
  could be fetched it also serves the scalar runs (flag `pseudodojo_fr_as_scalar`); if nothing could be fetched the
  bundled set stays and DFT results carry `pseudos:bundled`. The 0.7 golden test stays pinned to the bundled set;
  `test_simws_pseudodojo_vs_bundled` compares the two (masses 3 %, conduction minimum 1 %, gap 50 meV). Reference run (PBE, PseudoDojo
  fully relativistic file used for the scalar runs): m_l 0.957 vs 0.961, m_t 0.193 vs 0.195, conduction minimum
  0.841 vs 0.842 of Gamma-X, DFT gap 0.554 vs 0.583 eV.
- **Citations** (`simws.hub.citations`): every result carries its adapter's keys; `bibliography(results)` prints
  the list or BibTeX. Entries are marked unchecked until confirmed against the publisher record.
- **Overnight queue**: `python -m simws.queue add <adapter> --params '{...}'`, `python -m simws.queue run --until
  06:30`; interrupted entries are re-queued and committed results are never recomputed. The nightly task runs
  the queue, then `python -m simws.hub.maintain` (prunes large `until_validated` fields once every consumer has
  passed, writes `store/reports/disk_<date>.json`, warns above 80 % disk).
- **Flagship study A** (`simws/studies/specs/flagship_a.yaml`, rule `flagship_a` in `studies`): SiGe HBT vs Si BJT,
  77-400 K, screening fidelity. First run (current gain at 1 A/m, i.e. 1 uA per um of emitter):
  SiGe 106 -> 68 and Si 15 -> 21 from 250 to 400 K (the SiGe gain falls and the Si gain rises with temperature,
  as H2 states); collector-current ratio at 0.6 V 9.2, 5.7, 4.0, 3.1 at 250, 300, 350, 400 K. That ratio follows
  B exp(dE/kT) with dE = 63 meV (rms log residual 0.006), while the graded-only form x/(1 - e^-x) does not fit
  (0.24): H1 as written is rejected. Likely reason, to test in 0.9: the emitter-base depletion pushes the neutral
  base edge into Ge-containing material, so an abrupt offset term adds to the grading (the 0.6 ideality check
  points the same way). Below 250 K everything is flagged extrapolated; the SiGe device does not reach
  equilibrium at 77 K (not_converged) and the Si BJT gain there is 0.2, which needs the cryogenic models before it
  means anything.
- Not covered yet: strain and dose in the material laws (0.12), the field store with conservative transfer, paper
  fidelity for devices (mesh-refinement GCI per result), freeze-out and low-temperature mobility (cryogenic package).

## simws 0.9a: hub fixes and the cryogenic core

`notebooks/cryo_tour.ipynb` walks through it. Nothing below changes a 0.8 result: every new model is off unless
asked for, and the 0.8 unit, connection and slow tests still pass against their golden values.

- **Hub fixes.**
  - Job keys round floats to 12 significant digits, so 300.0000000000001 K is 300 K.
  - The environment lock hashes only physics packages (`envs/<env>.explicit.txt` plus `envs/<env>.pip.txt`), so
    a new Jupyter does not invalidate the store.
  - `Manifest.physics_version` enters the key, while the code version only goes into provenance.
  - Cheap adapters run whole ensembles as one job (`batch_members`).
  - The store refuses network and Windows file systems (`/mnt/c`, 9p, nfs). Override with
    `SIMWS_STORE_ALLOW_UNSAFE=1`.
  - `hub.sketch(func, params, name)` stores quick explorations flagged `sketch`. Paper fidelity refuses them.
  - The coupling registry (`simws.hub.coupling`) enforces that two-way couplings live inside one adapter.
- **Canary relinking.** After an upgrade changes job keys, run `python -m simws.hub.relink`. It reruns two
  canaries per adapter (one for DFT, which must agree to 1e-6) and links every other result to its new key
  (`equivalent_to` edge, `relinked` event) when the canaries reproduce to 1e-9. When a canary differs, nothing of
  that adapter is linked, and those results are recomputed when next requested. The installer runs it.
- **Cryogenic core** (`Physics(incomplete_ionization=True, fermi_dirac=True, mobility="klaassen")`;
  `HBTCase(cryo=True)`; `devsim.hbt_gummel` param `cryo`):
  - **Freeze-out.** Altermatt 2006 (P, As, B): the ionization energy falls with doping and the bound fraction
    vanishes at the Mott transition. `ii_model="charon"` gives Charon's constant-energy form for cross-checks.
  - **Fermi–Dirac statistics** through the degeneracy factor γ, using Nilsson's inverse (density error at most
    0.49 %, recorded as a model error). The factor enters:
    - the space charge;
    - the Scharfetter–Gummel currents, which use the generalized Einstein relation, ψ + ΔEc + V_t ln γn;
    - the contacts;
    - SRH and Auger recombination, whose equilibrium is n·p = γn·γp·n_ie².
  - **Band-gap narrowing with FD.** Slotboom's narrowing is apparent (fitted with Boltzmann statistics), so with
    FD the physical value −kT ln γ(300 K) of the majority carriers is used. Without this, degeneracy is counted
    twice and a cold Si BJT shows an unphysical gain of about 230.
  - **Klaassen 1992 mobility**: lattice, impurity, carrier–carrier scattering and clustering. It depends on the
    carriers, so after every bias point it is iterated to self-consistency (`relax_mobility`). The iteration is
    damped and restartable: a failed Newton solve restores the state and halves the update, and the converged
    state does not depend on this.
  - **Band parameters.** Green's Nc and Nv are scaled together so that the Boltzmann limit is exactly the 0.8
    n_i(T).
  - **Extended precision** (DEVSIM 80-bit) is on automatically for cryo devices below 200 K. Minority densities
    there fall below 1e-300 cm⁻³, and in double precision the SiGe HBT Newton iteration stalls at 77 K. Results
    are identical where both converge.
  - **Current-bounded sweeps.** The Gummel adapter's `IC_max_A_per_m` ends the sweep once IC (or the
    ideal-slope prediction for the next point) passes the limit, and flags `sweep_stopped`. At 77 K, 0.05 V is a
    factor of about 1800 in current.
- **Checks** (`simws.validation.cryo_checks`; `tests/unit/test_cryo.py`; rule `cryo_gate`):

  | Check | Result | Gate |
  |---|---|---|
  | Freeze-out in a uniform bar vs the local-neutrality solution | 5e-13 | 1e-6 |
  | Small-bias current vs q n μ V/L with Klaassen | 1.5e-12 | 1e-4 |
  | Zero-bias current on a bar graded from 1e15 to 1e20 cm⁻³, degenerate end, 77 and 300 K, relative to the drift scale (Einstein relation / detailed balance) | 3e-13 | 1e-10 |
  | Asymmetry of the conductance matrix of a three-contact bar (Onsager reciprocity) | passes at 300 K (77 K runs in `cryo_gate`) | 1e-6 |

  The HBT adapter also refuses a result whose zero-bias terminal current is above the noise floor.
- **Charon cross-check** (`bash tools/run_charon_xcheck.sh`, laptop; gate passed 4 Nov 2026).
  - **Setup.** A 1e17/1e17 cm⁻³ pn diode with Charon's incomplete-ionization form, Fermi–Dirac statistics, fixed
    mobilities and SRH. Our band parameters are written into the Charon decks. Both codes use the same 1.25 nm mesh
    along x, written by `simws.validation.charon_mesh` in the same layout as Charon's own pndiode.exo.
  - **Results:**

    | Case | Worst gated current difference (gate 2 %) |
    |---|---|
    | 300 K | 0.07 % |
    | 150 K | 0.11 % |
    | 77 K | 0.04 % |
    | 300 K, SRH off (diagnostic) | 0.05 % |

    The frozen-out majority densities agree to 3e-4 (93 % ionized at 300 K, 57–67 % at 150 K, 10–13 % at 77 K). The
    built-in voltages agree to within 30 µV.
  - **What it took** (Charon 2.2):
    - `Import State File <mesh> at index 1` on a bare mesh aborts silently with MPI_ABORT 14.
    - The nonlinear-Poisson stage aborts with these material blocks.
    - A Band Gap block needs `temperature dependence is on`.
    - Drift-diffusion needs a backtracking line search.
    - Cold cases are reached by temperature continuation at 0 V: 300 → 200 → 150 → 110 → 95 → 85 → 77 K.
    - `python tools/charon_bisect.py` isolates a silently aborting deck.
  - **Mesh finding.** On Charon's 10 nm mesh the SRH peak (width about kT/qE) is not resolved, so the
    recombination-dominated currents differ by up to ±14 % about the converged value, with opposite signs in the two
    codes: node lumping undershoots and quadrature overshoots. The 10 nm control case reproduces that exactly.
- **Paper fidelity for devices.** `devsim.hbt_gummel` at `fidelity="paper"` runs three meshes (spacings ×1,
  1/√2 and ½). Each quantity gets a numerical uncertainty (`components.numerical`) and `verification["mesh"]` per
  bias point. The gatekeeper requires both.
  - The GCI of the finest mesh is used only when the observed order lies between 0.5 and 4 and the meshes are
    near the asymptotic range.
  - Otherwise the uncertainty is a conservative bound, Fs |f1 − f2| / (r^0.5 − 1) / k, and the result is flagged
    `mesh_not_asymptotic`.
  - `simws.verify.analyse` now also refuses to report a GCI when the differences do not shrink with refinement.
    Before, it returned a negative one.
  - **Finding:** the HBT mesh used since 0.6 is not converged.
    - At 300 K, IC drops 0.9 % and IB 4–5 % per √2 refinement, with no clean order.
    - So paper-fidelity σ is about 3 % on IC and 8–12 % on IB and β.
    - Ratios such as SiGe/Si cancel most of it (Moll–Ross 0.05 %).
    - The likely source is current crowding at the emitter-contact corner. Mesh grading there (gmsh) is a
      0.9b item.
- **Flagship A v2** (`specs/flagship_a.yaml`; the 0.8 run is `specs/flagship_a_v08.yaml`):
  - Uses the cryogenic core from 77 to 400 K, submits jobs in parallel (`--workers`; the rule uses 6), and stops
    sweeps at 1 mA/µm.
  - New quantities `hbt.y_nb_edge`, `hbt.ge_nb_edge` and `hbt.dEg_ge_nb_edge`: the emitter edge of the neutral
    base, its Ge fraction and its Ge-induced gap reduction.
  - The H1 follow-up fits ratio = exp(ΔEg_edge/kT) × A x/(1 − e⁻ˣ), with ΔEg_edge read from the device at each
    temperature, not fitted.
  - Result of the first full run: every temperature from 77 to 400 K converges for both devices, with no
    partial sweeps.
  - **Current gain at 1 A/m** (1 µA per µm of emitter) falls with cooling for the Si BJT and rises for the SiGe
    HBT, as H2 states:

    | T (K) | 77 | 150 | 200 | 250 | 300 | 350 | 400 |
    |---|---|---|---|---|---|---|---|
    | Si BJT β | 2.1 | 5.0 | 7.7 | 11 | 15 | 19 | 24 |
    | SiGe HBT β | 298 | 106 | 88 | 81 | 79 | 79 | 80 |
    | IC ratio (SiGe/Si) | 348 | 39 | 17 | 9.4 | 5.8 | 4.1 | 3.1 |

    The IC ratio is taken at 0.95, 0.75 and 0.65 V at 77, 150 and 200 K, and at 0.6 V from 250 K up.
  - **Within 250–400 K** the ratio again follows B exp(dE/kT) with dE = 63 meV (rms 0.006), the same as 0.8.
  - **Over the whole 77–400 K window**, neither single form holds: graded only gives rms 0.99, exp only gives
    rms 0.25.
  - **H1b fits** with rms 0.018: exp(ΔEg_edge/kT) × 1.10 x/(1 − e⁻ˣ). Here ΔEg_edge is 14–28 meV (Ge fraction
    0.037–0.049 at the edge) and the remaining grade is 49 meV.
  - The value of ΔEg_edge matches the material model: 0.75 eV × 0.047 minus the DOS term 0.49 kT gives 22 meV
    at 300 K.
  - The remaining grade is below the nominal 67 meV between the neutral-base edges. The likely cause is the
    base–collector depletion cutting the grade short; checking it is a 0.9b item.
  - The SiGe gain at 250–400 K (about 80) differs from 0.8 (106 → 68) because of Klaassen vs Masetti mobility
    and the FD-consistent narrowing in the emitter.
  - Below 250 K, results keep `extrapolated` and `cryo_unvalidated`: the core is checked against closed forms
    and Charon, not against measured devices.

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
