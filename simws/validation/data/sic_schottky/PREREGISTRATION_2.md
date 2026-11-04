# SiC Schottky held-out gate, protocol 2 (pre-registration)

Fixed 2026-11-05, approved by the user (option 1), before the second dataset was opened. Committed before any data
of it is extracted; the git history is the timestamp.

## Why a protocol 2
Protocol 1 (Vivona et al. 2020, WC/4H-SiC) failed as declared: the held-out window reached 1e-1 A/cm^2, where an
extrinsic series resistance (2.9 ohm cm^2, set-up dominated) sets the current. The thermionic region agreed within
a factor 1.18, but that figure was computed after seeing the data and is not a pass. Protocol 1 stays recorded as
a failure.

## Dataset (candidate) and eligibility
- Candidate: M. Vivona et al., "Ni Schottky barrier on heavily doped phosphorous implanted 4H-SiC",
  doi:10.1088/1361-6463/ac13f3, arXiv:2102.08927 (same group as protocol 1: the measurement set-up may be similar;
  recorded as a limit on independence).
- Eligible only if a figure shows forward I-V (or J-V) curves at >= 3 temperatures on a logarithmic current axis.
  Eligibility is checked from the figure captions and text only, before any curve is extracted. If it is not
  eligible, the next candidate is chosen by the user, and this file is amended before that dataset is opened.

## Fixed rules
1. Split (code: split_rule_2): fit at the lowest temperature and at the temperature closest to the middle of the
   measured range (excluding the lowest); every other temperature is held out. The rule is applied by the code to
   the temperatures found in the data; nobody chooses temperatures.
2. Model and free parameters unchanged from protocol 1: thermionic emission with image-force lowering, WKB
   tunnelling (m* = 0.42, eps_r = 9.7), Gaussian barrier inhomogeneity (Werner-Guettler) and a series resistance;
   free: phi_B0, sigma_s, R_s, fitted on the fit temperatures in 1e-7 to 1e-1 A/cm^2. A* = 146 A cm^-2 K^-2 and
   Nc(300 K) = 1.7e19 cm^-3 (provisional).
3. Doping: the near-surface donor concentration stated in the paper (for an implanted profile, its value within
   the first 50 nm); the sentence it comes from is quoted in meta.json. Area: the paper's, if current is plotted.
4. Held-out window (code: window_2): from 1e-7 A/cm^2 up to J_top(T) = min(1e-1, (kT/q) / R_s), R_s from the fit.
5. Acceptance: every held-out point in its window within a factor 1.3 of the measurement (unchanged).
6. Each held-out temperature needs >= 10 points in its window; otherwise the gate is inconclusive (not a pass).
7. Digitization as for protocol 1 (vector extraction if the figure is vector, else colour separation), with the
   overlay check and the scripts stored next to the data.

## Amendment 1 (2026-11-05, before any curve of the second dataset was extracted)
- Eligibility: met. Fig. 3 caption and text: forward J-V at 25-115 C in 15 C steps (7 temperatures). The rule gives
  fit = 25 and 70 C, held out = 40, 55, 85, 100, 115 C.
- Doping (rule 3): the paper states two near-surface values: the SIMS peak 1e20 cm^-3 (chemical phosphorus) and the
  active donor density 1.97e19 cm^-3 (TFE fit of the 25 C curve, a fit-set temperature, agreeing with SCM of the
  active profile). The active density is the one that sets the barrier field, so N_D = 1.97e19 cm^-3. The paper's
  own per-temperature N_D (Fig. 3 inset) is not used.

## Result (2026-11-05)
FAIL, conclusive (52-65 points per held-out temperature). Fit at 25 and 70 C: phi_B0 2.12 eV, sigma_s 0,
R_s 0.73 ohm cm^2, rms ln residual 0.77 (the model does not fit even the fit temperatures). Held-out worst ratios:
40 C 1.85, 55 C 1.21, 85 C 3.9, 100 C 6.6, 115 C 11.2.

Diagnostics after the result (post hoc, not part of the gate):
- The data's ideality falls from 1.80 (25 C) to 1.37 (115 C), as thermionic-field emission predicts
  (n = (E00/kT) coth(E00/kT) = 1.77 at 25 C with kT/qE00 = 0.61): the tunnelling regime is the right one.
- Switching off the image-force factor: phi_B0 1.74 eV (the paper's TFE fit: 1.77 eV), rms 0.44, held-out worst
  5.2. At 2e19 cm^-3 the interface field is ~3.6 MV/cm and the image-force lowering ~0.23 eV; applying it as a
  factor exp(d_phi/kT) on top of a WKB integral over an unrounded triangular barrier double-counts at high field.
- The paper's own TFE fits need a barrier that falls with T (1.77 -> 1.66 eV, about -1.2 meV/K). The model's
  Gaussian inhomogeneity can only make the effective barrier rise with T.
Model work this points to: image-force rounding inside the WKB integral (not as a separate factor); Fermi-Dirac
statistics for N_D > N_c; a physical origin for the falling barrier (band-gap shift, implantation defects).
A further test needs a protocol 3 and a third dataset nobody has opened.
