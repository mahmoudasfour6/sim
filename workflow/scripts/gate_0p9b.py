"""simws 0.9b gate (Build Plan): contacts, small-signal and noise, shared geometry and conservative coupling.

    python workflow/scripts/gate_0p9b.py --outdir results/gate_0p9b [--parts contacts,small_signal,geometry]

contacts      thermionic emission against its closed form to 1e-4 (and the 1/mu scaling of the deviation), detailed
              balance at zero bias, image-force lowering (2 %), WKB quadrature (1e-4), tunnelling bookkeeping (1e-3);
              Padovani-Stratton reported as model-form spread (simws.validation.contact_checks)
small_signal  f_T from AC h21 within 5 % of 1/(2 pi tau) from charge partitioning (SiGe and Si HBT), depletion
              capacitance, shot noise 2qI, Nyquist 4kTG (simws.validation.small_signal_checks)
geometry      one gmsh geometry for DEVSIM, scikit-fem and Geant4; conservative transfers to 1e-8 with the L2
              round-trip error recorded; two-way electro-thermal coupling through preCICE conserving power to 1e-8
              (simws.validation.geometry_checks)
The Schottky cross-check against Charon (SK300, SK200) runs on the laptop with tools/run_charon_xcheck.sh.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--parts", default="contacts,small_signal,geometry")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
parts = a.parts.split(",")
res = {}

if "contacts" in parts:
    from simws.validation import contact_checks as C
    r = C.run_all()
    res["contacts"] = r
    print(f"contacts: TE {r['te_closed_form']['worst_rel_diff']:.2e} (1/mu ratio "
          f"{r['te_closed_form']['deviation_ratio_mu'][0]:.1f}), balance {r['detailed_balance']['ratio']:.1e}, image "
          f"{r['image_force']['worst_rel_diff']:.1e}, quadrature {r['tunnel_quadrature']['worst_rel_diff']:.1e}, "
          f"tunnel device {r['tunnel_device']['worst_rel_diff']:.1e} -> {'PASS' if r['pass'] else 'FAIL'}")

if "small_signal" in parts:
    from simws.validation import small_signal_checks as S
    r = S.run_all(with_hbt=True)
    res["small_signal"] = r
    for row in r["hbt_ft"]["rows"]:
        print(f"small_signal: Ge {row['ge_peak']:.2f} VBE {row['VBE']:.2f} fT {row['fT_ac_GHz']:.2f} GHz vs "
              f"{row['fT_charge_GHz']:.2f} GHz ({100 * row['rel_diff']:+.2f} %), fmax {row['fmax_GHz']:.1f} GHz")
    print(f"small_signal: C_dep {r['depletion_capacitance']['worst_rel_diff']:.1e}, shot "
          f"{r['shot_noise']['worst_rel_diff']:.1e}, Nyquist {r['nyquist_2d']['rel_diff']:.1e} -> "
          f"{'PASS' if r['pass'] else 'FAIL'}")

if "geometry" in parts:
    from simws.validation import geometry_checks as G
    r = G.run_all(out / "geometry_work")
    res["geometry"] = r
    t = r["transfer"]
    print(f"geometry: volumes {r['volumes']['structured']['worst']:.1e} (unstructured "
          f"{r['volumes']['unstructured']['worst']:.1e}), contacts {r['contacts']['worst_rel']:.1e}, same device "
          f"{r['same_device']['worst_rel']:.1e}")
    print(f"geometry: transfer conservation {t['worst_conservation']:.1e}; round-trip L2 "
          + ", ".join(f"{k} {v['roundtrip_l2_rel']:.2e}" for k, v in t.items()
                      if isinstance(v, dict) and v.get("roundtrip_l2_rel") is not None))
    for k, rows in r["precice"].items():
        if isinstance(rows, list):
            for row in rows:
                print(f"geometry: preCICE {k:18s} {row['bias_V']:.1f} V  T_max {row['T_max_K']:.4f} K "
                      f"(dT vs in-process {row['dT_max_K']:+.2e} K)  power transfer {row['transfer_conservation']:.1e}"
                      f"  energy {row['energy_balance']:.1e}  its {row['coupling_iterations']}")
    print(f"geometry -> {'PASS' if r['pass'] else 'FAIL'}")

res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
(out / "gate_0p9b.json").write_text(json.dumps(res, indent=1, default=float))
print("0.9b GATE", "PASS" if res["pass"] else "FAIL")
raise SystemExit(0 if res["pass"] else 1)
