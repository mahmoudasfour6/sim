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
