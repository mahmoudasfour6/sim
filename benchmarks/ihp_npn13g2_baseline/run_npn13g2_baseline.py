#!/usr/bin/env python3
"""
IHP SG13G2 npn13G2 foundry-model baseline (300 K reference)
===========================================================
Runs the production IHP VBIC model (sg13g2_hbt_mod.lib, corner hbt_typ) in ngspice and
extracts the reference characteristics that any in-house TCAD SiGe HBT must reproduce
before cryogenic / radiation claims are made:

  1. Forward Gummel  I_C, I_B vs V_BE  (V_CB = 0 V)
  2. DC current gain beta vs I_C
  3. Output characteristics I_C vs V_CE (forced I_B)
  4. Cutoff frequency f_T vs I_C (from |h21| at a fixed frequency in the -20 dB/dec region)
  5. Same at the process corners hbt_bcs / hbt_wcs (spread band)

Every number produced here comes directly from ngspice + the foundry model. Nothing is
hand-tuned. Valid temperature range of the model is the foundry range (~ -40..125 C),
so this script does NOT extrapolate to 77 K.

Run with the `eda` environment:
    /home/mahmoudasfour/miniforge3/envs/eda/bin/python run_npn13g2_baseline.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PDK_MODELS = "/home/mahmoudasfour/sim_workspace/ihp_sg13g2_pdk/ihp-sg13g2/libs.tech/ngspice/models"
NGSPICE = os.environ.get("NGSPICE", "/home/mahmoudasfour/miniforge3/envs/eda/bin/ngspice")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

VCE_BIAS = 1.5          # V, IHP datasheet condition for f_T
F_H21 = 20e9            # Hz, frequency used for f_T = f * |h21|
DEVICE = "npn13G2"
NX = 1                  # emitter multiplicity (0.07 x 0.9 um^2 drawn per finger)


def run_ngspice(netlist: str, tag: str) -> str:
    path = os.path.join(OUT, f"{tag}.cir")
    with open(path, "w") as f:
        f.write(netlist)
    res = subprocess.run([NGSPICE, "-b", path], capture_output=True, text=True, cwd=OUT)
    with open(os.path.join(OUT, f"{tag}.log"), "w") as f:
        f.write(res.stdout + "\n--- STDERR ---\n" + res.stderr)
    if res.returncode != 0:
        sys.exit(f"ngspice failed for {tag}; see {tag}.log")
    return path


def header(corner: str, temp_c: float) -> str:
    return f"""* IHP SG13G2 {DEVICE} baseline ({corner}, {temp_c} C)
.lib {PDK_MODELS}/cornerHBT.lib {corner}
.param temp={temp_c}
.option temp={temp_c} tnom=27 gmin=1e-15 abstol=1e-16 reltol=1e-5
"""


def gummel(corner="hbt_typ", temp_c=27.0):
    """Forward Gummel at exact V_CB = 0 V (V_C = 0, V_B = 0, sweep V_E = -V_BE)."""
    tag = f"gummel_{corner}_{int(temp_c)}C"
    net = header(corner, temp_c) + f"""
VC c 0 DC 0
VB b 0 DC 0
VE e 0 DC -0.6
X1 c b e 0 {DEVICE} Nx={NX}
.control
dc VE -0.35 -1.02 -0.005
let vbe = -v(e)
let ic = -i(VC)
let ib = -i(VB)
wrdata {tag}.dat vbe ic ib
.endc
.end
"""
    run_ngspice(net, tag)
    d = np.loadtxt(os.path.join(OUT, f"{tag}.dat"))
    vbe, ic, ib = d[:, 1], d[:, 3], d[:, 5]
    return vbe, ic, ib


def output_curves(corner="hbt_typ", temp_c=27.0, ib_list=(1e-6, 2e-6, 4e-6, 6e-6, 8e-6)):
    curves = {}
    for ibv in ib_list:
        tag = f"output_{corner}_ib{ibv*1e6:.0f}uA"
        net = header(corner, temp_c) + f"""
VC c 0 DC 1.0
IB 0 b DC {ibv}
X1 c b 0 0 {DEVICE} Nx={NX}
.control
dc VC 0 1.6 0.01
let ic = -i(VC)
wrdata {tag}.dat ic
.endc
.end
"""
        run_ngspice(net, tag)
        d = np.loadtxt(os.path.join(OUT, f"{tag}.dat"))
        curves[ibv] = (d[:, 0], d[:, 1])
    return curves


def ft_sweep(corner="hbt_typ", temp_c=27.0, vbe_pts=np.arange(0.72, 0.955, 0.01)):
    """Small-signal short-circuit current gain f_T(I_C) = F_H21 * |h21(F_H21)| with AC current injection."""
    tag = f"ft_{corner}_{int(temp_c)}C"
    blocks = []
    for i, v in enumerate(vbe_pts):
        blocks.append(
            f"alter VB dc={v:.4f}\n"
            f"op\n"
            f"let icdc_{i} = -i(VC)\n"
            f"print icdc_{i}\n"
            f"ac lin 1 {F_H21} {F_H21}\n"
            f"let h21_{i} = mag(i(VC))\n"
            f"print h21_{i}\n"
        )
    alters = "\n".join(blocks)
    net = header(corner, temp_c) + f"""
VC c 0 DC {VCE_BIAS}
VB b_dc 0 DC 0.8
Lbias b_dc b 1m
IB 0 b DC 0 AC 1.0
X1 c b 0 0 {DEVICE} Nx={NX}
.control
{alters}
.endc
.end
"""
    run_ngspice(net, tag)
    txt = open(os.path.join(OUT, f"{tag}.log")).read()
    ic_map, h21_map = {}, {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("icdc_"):
                idx = int(k.split("_")[1].split("[")[0])
                try:
                    ic_map[idx] = float(v.split()[0])
                except ValueError:
                    pass
            elif k.startswith("h21_"):
                idx = int(k.split("_")[1].split("[")[0])
                try:
                    h21_map[idx] = float(v.split()[0])
                except ValueError:
                    pass
    valid = sorted(set(ic_map.keys()) & set(h21_map.keys()))
    ic = [ic_map[i] for i in valid]
    ft = [h21_map[i] * F_H21 for i in valid]
    return np.array(ic), np.array(ft)


def main():
    results = {"device": DEVICE, "Nx": NX, "VCE_ft_V": VCE_BIAS, "f_h21_Hz": F_H21, "corners": {}}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    colors = {"hbt_typ": "k", "hbt_bcs": "tab:green", "hbt_wcs": "tab:red"}
    for corner in ("hbt_typ", "hbt_bcs", "hbt_wcs"):
        vbe, ic, ib = gummel(corner)
        beta = ic / np.maximum(ib, 1e-30)
        ic_ft, ft = ft_sweep(corner)
        mask = (ic > 1e-6) & (ic < 5e-3)
        i_pk = int(np.argmax(ft))
        idx_1ua = int(np.argmin(np.abs(ic - 1e-6)))
        idx_peak_beta = int(np.argmax(np.where(mask, beta, 0)))
        results["corners"][corner] = {
            "beta_peak": float(beta[idx_peak_beta]),
            "IC_at_beta_peak_A": float(ic[idx_peak_beta]),
            "VBE_at_IC_1uA_V": float(vbe[idx_1ua]),
            "fT_peak_GHz": float(ft[i_pk] / 1e9),
            "IC_at_fT_peak_A": float(ic_ft[i_pk]),
        }
        c = colors[corner]
        lw = 2.2 if corner == "hbt_typ" else 1.2
        ax[0, 0].semilogy(vbe, ic, color=c, lw=lw, label=f"I_C {corner}")
        ax[0, 0].semilogy(vbe, ib, color=c, lw=lw, ls="--", label=f"I_B {corner}")
        ax[0, 1].semilogx(ic[mask], beta[mask], color=c, lw=lw, label=corner)
        ax[1, 1].semilogx(ic_ft * 1e3, ft / 1e9, "o-", color=c, lw=lw, ms=3, label=corner)

    for ibv, (vce, icv) in output_curves().items():
        ax[1, 0].plot(vce, icv * 1e3, lw=1.8, label=f"I_B = {ibv*1e6:.0f} uA")

    ax[0, 0].set(xlabel="V_BE [V]", ylabel="Current [A]", title="Forward Gummel (V_CB = 0), 27 C", ylim=(1e-13, 1e-1))
    ax[0, 1].set(xlabel="I_C [A]", ylabel="beta = I_C / I_B", title="DC current gain")
    ax[1, 0].set(xlabel="V_CE [V]", ylabel="I_C [mA]", title="Output characteristics (typ, self-heating on)")
    ax[1, 1].set(xlabel="I_C [mA]", ylabel="f_T [GHz]", title=f"Cutoff frequency (V_CE = {VCE_BIAS} V)")
    for a in ax.flat:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(f"IHP SG13G2 {DEVICE} (Nx={NX}) foundry VBIC model - 300 K reference baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(HERE, "npn13g2_baseline_300K.png")
    fig.savefig(png, dpi=140)

    with open(os.path.join(HERE, "npn13g2_baseline_300K.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("PNG:", png)


if __name__ == "__main__":
    main()
