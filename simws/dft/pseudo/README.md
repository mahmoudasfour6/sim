Silicon norm-conserving pseudopotentials used by simws.studies.dft_params (shipped so every machine reproduces
the same numbers). They come from the Quantum ESPRESSO distribution (pseudo/ directory of the QE source and the
Debian/Ubuntu quantum-espresso-data package), Copyright the Quantum ESPRESSO group, GPL-2.

| file                 | type                                  | functional | md5                              |
|----------------------|---------------------------------------|------------|----------------------------------|
| Si.pbe-rrkj.UPF      | norm-conserving RRKJ, non-relativistic | PBE        | d7f89eb60b7b396f5f4f9eaa825fcfdc |
| Si.rel-pbe-rrkj.UPF  | norm-conserving RRKJ, fully relativistic (spin-orbit) | PBE | ca11b4d55ae68281b492616f935c6016 |
| Si.pz-vbc.UPF        | norm-conserving von Barth-Car         | LDA (PZ)   | a974d1b8727157e37210f3f86afb6210 |

To use other pseudopotentials set dft.pseudo_dir and dft.pseudos (Si_pbe, Si_pbe_fr, Si_lda) in
config/config.yaml; re-run the checks (results/dft/checks.json) before trusting the new set.
