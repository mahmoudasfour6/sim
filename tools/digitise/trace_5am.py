"""Contract A23.1: trace the 5AM validation figures into simws/data/validation/ (vector figures read from the PDFs'
drawing operators; tools/digitise/vector.py). Run from the repo root with the source PDFs' folder:

    python tools/digitise/trace_5am.py --pdfs <folder> [--overlays <folder>]

The PDFs are not in the repo (copyright); their SHA-256 are recorded in each data file."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.digitise import vector as V  # noqa: E402

OUT = os.path.join("simws", "data", "validation")
VERTEX_NOTE = ("vector: the plotted line's vertices read from the PDF's drawing operators, axes calibrated on the "
               "tick marks in the same stream; vertex precision 0.01 pt")


def _r(a, n=6):
    return [float(f"{v:.{n}g}") for v in np.asarray(a, float)]


def pellish(pdfs, overlays):
    pdf = os.path.join(pdfs, "pellish_thesis.pdf")
    # Fig. 3, page 6 of the thesis (index 14): Gummel and beta
    clip3 = (140, 80, 470, 330)
    pp = V.paths(pdf, 14, clip3)
    x = V.Axis(181.56, 0.5, 434.43, 1.2)
    yl = V.Axis(295.68, -12.0, 92.88, 0.0, log=True)
    yb = V.Axis(295.68, 0.0, 92.88, 200.0)
    ic, ib, be = (V.dedupe(V.longest(pp, c)) for c in ((0, 0, 0), (1, 0, 0), (0, 0, 1)))
    for s in (ic, ib, be):
        assert len(s) == 71, len(s)
    fig3 = {"figure": "Fig. 3", "page_index": 14, "device": "IBM 5AM npn SiGe HBT, AE 0.5 x 2.5 um^2",
            "conditions": "forward Gummel, 300 K (room temperature); VCB not stated (taken as 0 V)",
            "method": VERTEX_NOTE,
            "axes": {"VBE_V": x.as_dict(), "current_A": yl.as_dict(), "beta": yb.as_dict()},
            "VBE_V": _r(x(ic[:, 0]), 5), "IC_A": _r(yl(ic[:, 1])), "IB_A": _r(yl(ib[:, 1])),
            "beta": _r(yb(be[:, 1]), 5),
            "beta_VBE_V": _r(x(be[:, 0]), 5),
            "note": "the caption: the beta spike near VBE = 0.4 V (outside this axis) is the analyser's resolution"}
    # Fig. 4, the same page: forced-IB output curves
    clip4 = (140, 380, 520, 720)
    pp4 = V.paths(pdf, 14, clip4)
    x4 = V.Axis(181.56, 0.0, 452.93, 4.0)
    y4 = V.Axis(638.96, 0.0, 429.65, 2.5)
    curves = {}
    for ib_uA, col in ((6.0, (0, 0, 0)), (7.5, (1, 0, 0)), (9.0, (0, 0.55, 0)), (10.5, (0, 0, 1)),
                       (12.0, (1, 0, 1))):
        L = V.dedupe(V.longest(pp4, col))
        curves[f"{ib_uA}"] = {"IB_uA": ib_uA, "VCE_V": _r(x4(L[:, 0]), 5), "IC_mA": _r(y4(L[:, 1]), 5)}
    fig4 = {"figure": "Fig. 4", "page_index": 14, "device": "IBM 5AM npn SiGe HBT, AE 0.5 x 2.5 um^2",
            "conditions": "forced IB (6, 7.5, 9, 10.5, 12 uA), VCE swept 0 to 4 V, 300 K",
            "method": VERTEX_NOTE, "axes": {"VCE_V": x4.as_dict(), "IC_mA": y4.as_dict()}, "curves": curves}
    data = {"source": {"cite": "J. A. Pellish, Bulk silicon-germanium heterojunction bipolar transistor process "
                               "feature implications for single-event effects analysis and charge collection "
                               "mechanisms, PhD dissertation, Vanderbilt University, Dec. 2008",
                       "url": "https://irbe.library.vanderbilt.edu/server/api/core/bitstreams/"
                              "9e1a1bd1-f650-4fd3-bba8-066f7aad00f8/content",
                       "pdf_sha256": V.sha256(pdf)},
            "fig3": fig3, "fig4": fig4}
    if overlays:
        V.overlay(pdf, 14, clip3, [ic, ib, be], os.path.join(overlays, "pellish_fig3.png"))
        V.overlay(pdf, 14, clip4, [V.longest(pp4, c) for c in ((0, 0, 0), (1, 0, 0), (0, 0.55, 0), (0, 0, 1),
                                                               (1, 0, 1))],
                  os.path.join(overlays, "pellish_fig4.png"))
    return data


def _series(xmap, ymap, pts, frame, xname, yname, **extra):
    q = V.inside(V.dedupe(pts), frame)
    d = dict(extra)
    d[xname] = _r(xmap(q[:, 0]), 5)
    d[yname] = _r(ymap(q[:, 1]))
    return d, q


def luo(pdfs, overlays):
    pdf = os.path.join(pdfs, "luo2.pdf")
    out = {"source": {"cite": "L. Luo, Physics, compact modeling and TCAD of SiGe HBT for wide temperature range "
                              "operation, PhD dissertation, Auburn University, 2011",
                      "url": "https://etd.auburn.edu/handle/10415/2884", "pdf_sha256": V.sha256(pdf),
                      "device": "IBM 5AM npn SiGe HBT, single 0.5 x 2.5 um^2 (Luo Figs. 6.53, 7.3; section 1.2)"}}
    ov = []
    # Fig. 1.2 (page 2, index 20): SIMS; the depth axis has 8 equal tick intervals and no numbers
    clip = (150, 440, 520, 720)
    pp = V.paths(pdf, 20, clip)
    tick = V.Axis(181.24, 0.0, 433.91, 8.0)
    u = V.Axis(181.24, 0.0, 433.91, 1.0)
    conc = V.Axis(613.21, 16.0, 459.93, 21.0, log=True)
    ge_ax = V.Axis(613.21, 0.0, 450.74, 0.12)
    G = sorted(V.by_color(pp, (0, 0, 1), 20), key=lambda q: q.points[0, 0])
    ge = V.dedupe(np.vstack([G[0].points, G[1].points[1:]]))
    B = V.dedupe(sorted(V.by_color(pp, (1, 0, 1), 100), key=lambda q: q.n_items)[-1].points)
    P = V.dedupe(max(V.by_color(pp, (0, 0, 0), 100), key=lambda q: q.n_items).points)
    A = sorted(V.by_color(pp, (1, 0, 0), 40), key=lambda q: q.points[0, 0])
    As_e, As_b = V.dedupe(A[0].points), V.dedupe(A[1].points)
    dashed = [q for q in V.paths(pdf, 20, clip) if V.same_color(q.color, (1, 0, 0)) and q.n_items == 1
              and np.ptp(q.points[:, 0]) < 0.01 and np.ptp(q.points[:, 1]) > 100]
    u_poly = float(u(dashed[0].points[0, 0]))
    f12 = {"figure": "Fig. 1.2", "page_index": 20, "method": VERTEX_NOTE,
           "conditions": "SIMS: As, B, P (cm^-3, log axis) and Ge mole fraction (linear, right axis) vs depth",
           "depth_axis": "no numbers on the page: u runs 0 to 1 across the frame (8 equal tick intervals); "
                         "u_tick = 8 u",
           "axes": {"u": u.as_dict(), "conc_cm3": conc.as_dict(), "Ge": ge_ax.as_dict()},
           "u_poly_mono": u_poly,
           "u_poly_mono_note": "the dashed red line at the As poly / monocrystalline interface",
           "As_emitter": {"u": _r(u(As_e[:, 0]), 5), "conc_cm3": _r(conc(As_e[:, 1]))},
           "As_emitter_note": "the trace stops at u = %.4f (still at 2.3e20): the plotted As ends there" %
                              float(u(As_e[-1, 0])),
           "B": {"u": _r(u(B[:, 0]), 5), "conc_cm3": _r(conc(B[:, 1]))},
           "B_note": "the main trace; the pieces beyond it are at the SIMS floor (about 1e16 to 1e17) and not used",
           "P": {"u": _r(u(P[:, 0]), 5), "conc_cm3": _r(conc(P[:, 1]))},
           "P_floor_cm3": 3e16,
           "As_buried": {"u": _r(u(As_b[:, 0]), 5), "conc_cm3": _r(conc(As_b[:, 1]))},
           "Ge": {"u": _r(u(ge[:, 0]), 5), "fraction": _r(ge_ax(ge[:, 1]), 5)}}
    gu, gv = u(ge[:, 0]), ge_ax(ge[:, 1])
    film = gu[gv > 0.01]
    f12["read_while_tracing"] = {"Ge_peak": round(float(gv.max()), 4),
                                 "Ge_above_1pct_u": [round(float(film.min()), 5), round(float(film.max()), 5)],
                                 "Ge_above_1pct_ticks": round(float(8 * (film.max() - film.min())), 4),
                                 "B_peak_cm3": float(f"{conc(B[:, 1]).max():.3g}")}
    out["fig1_2"] = f12
    ov.append((20, clip, [ge, B, P, As_e, As_b], "luo_fig1_2.png"))
    # Fig. 5.7(a) (page 74, index 92): IC vs VBE, 43 to 393 K; red = 192 K and above, blue = 162 K and below
    clip = (120, 300, 330, 512)
    pp = V.paths(pdf, 92, clip, overlap=True)
    xm, ym = V.Axis(154.47, 0.2, 314.53, 1.2), V.Axis(507.02, -10.0, 327.33, 2.0, log=True)
    frame = (154.47, 317.0, 314.53, 507.02)
    curves = []
    for col, temps in (((1, 0, 0), (393.0, 300.0, 262.0, 223.0, 192.0)),
                       ((0, 0, 1), (162.0, 136.0, 110.0, 93.0, 76.0, 60.0, 43.0))):
        runs = [V.joined(pp, col, 5)]
        # join each low piece to its upper piece: pieces end to start within 0.6 pt
        cand = sorted(V.by_color(pp, col, 5), key=lambda q: q.points[0, 0])
        lines, usedi = [], set()
        for i, a in enumerate(cand):
            if i in usedi:
                continue
            usedi.add(i)
            line = a.points
            for j, b in enumerate(cand):
                if j not in usedi and np.hypot(*(b.points[0] - line[-1])) <= 0.6:
                    line = np.vstack([line, b.points[1:]])
                    usedi.add(j)
            lines.append(line)
        assert len(lines) == len(temps), (col, len(lines))
        # hotter = lower VBE at a fixed current: order by VBE where IC first exceeds 1e-2 mA
        ref = ym.inverse(1e-2)
        def vbe_at(line):
            k = np.where(line[:, 1] <= ref)[0]
            return line[k[0], 0] if len(k) else 1e9
        lines.sort(key=vbe_at)
        for T, line in zip(temps, lines):
            d, q = _series(xm, ym, line, frame, "VBE_V", "IC_mA", T_K=T)
            curves.append(d)
            ov.append(None) if False else None
        ov.append((92, clip, lines, f"luo_fig5_7a_{col[2]:.0f}.png"))
    out["fig5_7a"] = {"figure": "Fig. 5.7(a)", "page_index": 92, "method": VERTEX_NOTE,
                      "conditions": "IC vs VBE, single 0.5 x 2.5 um^2; VCB not stated (Gummel: 0 V taken)",
                      "temperatures_from": "Fig. 5.7(b)'s markers (red 393, 300, 262, 223, 192 K; blue 162, "
                                           "136, 110, 93, 76, 60, 43 K); curves ordered by VBE at 1e-2 mA",
                      "axes": {"VBE_V": xm.as_dict(), "IC_mA": ym.as_dict()}, "curves": curves}
    # Fig. 4.7 (page 58, index 76): beta vs VBE at VCB = 0, 43 to 300 K
    clip = (176, 410, 460, 620)
    pp = V.paths(pdf, 76, clip, overlap=True)
    xm, ym = V.Axis(176.85, 0.6, 459.66, 1.2), V.Axis(619.92, 20.0, 419.27, 200.0)
    frame = (176.85, 410.0, 459.66, 640.0)
    curves, lines = [], []
    for T, col in ((300.0, (0, 0, 1)), (223.0, (1, 0, 0)), (162.0, (0, 1, 0)), (93.0, (1, 0, 1)), (43.0, (0, 0, 0))):
        line = max(V.by_color(pp, col, 50), key=lambda q: q.n_items).points
        d, q = _series(xm, ym, line, frame, "VBE_V", "beta", T_K=T)
        curves.append(d)
        lines.append(q)
    out["fig4_7"] = {"figure": "Fig. 4.7", "page_index": 76, "method": VERTEX_NOTE,
                     "conditions": "beta_0 vs VBE at VCB = 0 V (legend), 0.5 x 2.5 um^2",
                     "axes": {"VBE_V": xm.as_dict(), "beta": ym.as_dict()}, "curves": curves}
    ov.append((76, (160, 405, 470, 645), lines, "luo_fig4_7.png"))
    # Fig. 4.1 (page 54, index 72, upper): intrinsic base sheet resistance vs T, measured (blue markers)
    clip = (179, 100, 460, 325)
    pp = V.paths(pdf, 72, clip)
    xm, ym = V.Axis(179.67, 0.0, 459.39, 400.0), V.Axis(324.23, 4.0, 103.57, 7.0, log=True)
    mk = V.marker_centres(pp, (0, 0, 1))
    boxes = [q.points for q in pp if V.same_color(q.fill, (1, 1, 1)) and np.ptp(q.points[:, 0]) < 200]
    for b in boxes:                       # the legend's own marker sits in a white box: not a measurement
        lo, hi = b.min(0), b.max(0)
        mk = mk[~((mk[:, 0] >= lo[0]) & (mk[:, 0] <= hi[0]) & (mk[:, 1] >= lo[1]) & (mk[:, 1] <= hi[1]))]
    out["fig4_1"] = {"figure": "Fig. 4.1", "page_index": 72, "method": VERTEX_NOTE + "; the measured points are "
                     "the blue markers' centres", "conditions": "intrinsic base sheet resistance RsBi (ohm/sq) vs T",
                     "axes": {"T_K": xm.as_dict(), "RsBi_ohm_sq": ym.as_dict()},
                     "T_K": _r(xm(mk[:, 0]), 5), "RsBi_ohm_sq": _r(ym(mk[:, 1]))}
    ov.append((72, (150, 90, 470, 330), [mk], "luo_fig4_1.png"))
    # Fig. 4.2 (page 54, index 72, lower): M - 1 vs VCB at a small forced VBE, 43 to 300 K
    clip = (181, 420, 464, 629)
    pp = V.paths(pdf, 72, clip, overlap=True)
    xm, ym = V.Axis(181.22, 1.0, 463.49, 6.0), V.Axis(628.86, -4.0, 421.9, 0.0, log=True)
    frame = (181.22, 421.9, 463.49, 628.86)
    curves, lines = [], []
    for T, col in ((300.0, (0, 0, 1)), (223.0, (1, 0, 0)), (162.0, (0, 1, 0)), (93.0, (1, 0, 1)), (43.0, (0, 0, 0))):
        line = max(V.by_color(pp, col, 20), key=lambda q: q.n_items).points
        d, q = _series(xm, ym, line, frame, "VCB_V", "M_minus_1", T_K=T)
        curves.append(d)
        lines.append(q)
    out["fig4_2"] = {"figure": "Fig. 4.2", "page_index": 72, "method": VERTEX_NOTE,
                     "conditions": "M - 1 vs VCB at a small forced VBE (not stated; the same current at every T), "
                                   "0.5 x 2.5 um^2; Luo: the spikes near 4 V at 93 and 162 K are measurement errors",
                     "axes": {"VCB_V": xm.as_dict(), "M_minus_1": ym.as_dict()}, "curves": curves}
    ov.append((72, (160, 410, 480, 645), lines, "luo_fig4_2.png"))
    # Fig. 3.7(a) (page 49, index 67): fT vs JE at 300 K (green)
    clip = (100, 90, 330, 330)
    pp = V.paths(pdf, 67, clip, overlap=True)
    xm = V.Axis(115.2, -2.0, 269.25, 1.0, log=True)
    ym = V.Axis(281.95, 0.0, 160.5, 50.0)
    line = [q for q in V.by_color(pp, (0, 0.5, 0), 20) if q.points[:, 0].max() < 300 and q.points[:, 1].max() < 300]
    line = max(line, key=lambda q: q.n_items).points
    d, q = _series(xm, ym, line, (115.2, 150.0, 269.25, 282.0), "JE_mA_um2", "fT_GHz")
    out["fig3_7a"] = dict(figure="Fig. 3.7(a)", page_index=67, method=VERTEX_NOTE + "; the fT axis has no tick "
                          "marks in the stream: calibrated on its label centres (0 and 50 GHz)",
                          conditions="fT vs JE at 300 K, 0.5 x 2.5 um^2; VCB not stated",
                          axes={"JE_mA_um2": xm.as_dict(), "fT_GHz": ym.as_dict()}, T_K=300.0, **d)
    ov.append((67, (95, 140, 335, 315), [q], "luo_fig3_7a.png"))
    if overlays:
        for page, c, series, name in ov:
            V.overlay(pdf, page, c, series, os.path.join(overlays, name))
    return out


def sutton(pdfs, overlays):
    pdf = os.path.join(pdfs, "sutton_thesis.pdf")
    pp = V.paths(pdf, 147)
    xm, ym = V.Axis(203.61, 0.4, 477.52, 1.2), V.Axis(400.09, -10.0, 237.79, -2.0, log=True)
    L = V.chain(pp, (0, 0, 1))[:4]
    hot = sorted([l for l in L if l[:, 0].min() < 300], key=lambda l: l[:, 0].min())
    cold = sorted([l for l in L if l[:, 0].min() >= 300], key=lambda l: l[:, 0].min())
    frame = (203.61, 230.0, 477.52, 400.09)
    def pick(pair):
        # the collector current is the larger at the same VBE (smaller y), compared where both exist
        a, b = (V.inside(V.dedupe(p[np.argsort(p[:, 0])]), frame) for p in pair)
        x = 0.5 * (max(a[0, 0], b[0, 0]) + min(a[-1, 0], b[-1, 0]))
        ya, yb = np.interp(x, a[:, 0], a[:, 1]), np.interp(x, b[:, 0], b[:, 1])
        return (a, b) if ya < yb else (b, a)
    out = {"source": {"cite": "A. K. Sutton, Hardness assurance testing and radiation hardening by design "
                              "techniques for silicon-germanium heterojunction bipolar transistors and digital logic "
                              "circuits, PhD dissertation, Georgia Institute of Technology, Aug. 2009 (data of "
                              "Prakash et al., IEEE TNS 53(6):3175, 2006)",
                      "url": "https://repository.gatech.edu/entities/publication/96d2abe7-f88e-4768-8b19-7b54d6e3ba0d",
                      "pdf_sha256": V.sha256(pdf)}}
    curves = []
    for T, pair in ((300.0, hot), (77.0, cold)):
        jc, jb = pick(pair)
        curves.append({"T_K": T, "VBE_V_C": _r(xm(jc[:, 0]), 5), "JC_A_um2": _r(ym(jc[:, 1])),
                       "VBE_V_B": _r(xm(jb[:, 0]), 5), "JB_A_um2": _r(ym(jb[:, 1]))})
    out["fig72"] = {"figure": "Figure 72", "page_index": 147, "method": VERTEX_NOTE + "; the curves are drawn "
                    "as one path per segment and chained end to start", "device": "IBM 5AM, AE 0.50 x 1.0 um^2",
                    "conditions": "forward Gummel, VCB = 0 V, pre-irradiation (the solid blue curves), 300 K and "
                                  "77 K", "axes": {"VBE_V": xm.as_dict(), "J_A_um2": ym.as_dict()}, "curves": curves}
    if overlays:
        V.overlay(pdf, 147, (160, 200, 500, 430), L, os.path.join(overlays, "sutton_fig72.png"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--overlays", default=None)
    a = ap.parse_args()
    if a.overlays:
        os.makedirs(a.overlays, exist_ok=True)
    for name, fn in (("pellish2008.json", pellish), ("luo2011.json", luo), ("sutton2009.json", sutton)):
        d = fn(a.pdfs, a.overlays)
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(d, f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
