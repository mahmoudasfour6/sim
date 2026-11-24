"""Contract A25.1: trace Sutton's total-dose figures for the IBM 5AM into simws/data/validation/sutton2009.json
(vector figures read from the thesis' drawing operators; tools/digitise/vector.py). Run from the repo root:

    python tools/digitise/trace_tid.py --pdfs <folder with sutton_thesis.pdf> [--overlays <folder>]

Adds the keys fig22 to fig26, fig33, fig39, fig73 and fig72's post-irradiation curves; the pre-irradiation fig72
curves (A23, trace_5am.py) are kept as they are. Only the IBM 5AM series are traced."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.digitise import vector as V  # noqa: E402

OUT = os.path.join("simws", "data", "validation", "sutton2009.json")
VERTEX = ("vector: marker centres and error-bar caps read from the PDF's drawing operators, axes calibrated on the "
          "major tick marks in the same stream; vertex precision 0.01 pt")
LINES = ("vector: the plotted line's vertices read from the PDF's drawing operators (one path per segment, chained "
         "end to start), axes calibrated on the major tick marks in the same stream; vertex precision 0.01 pt")
BLUE, RED, GREEN, CYAN = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)


def _r(a, n=4):
    return [float(f"{v:.{n}g}") for v in np.asarray(a, float)]


CIRCLE, SQUARE, ANY = "circle", "square", "any"


def _shape_ok(p, shape):
    if shape == CIRCLE:
        return p.n_items == 4 and len(p.points) >= 5
    if shape == SQUARE:
        return p.n_items == 1 and len(p.points) == 4
    return p.n_items >= 2 or len(p.points) == 4


def markers(pp, color, filled, frame, exclude=(), max_size=8.0, shape=ANY):
    """Centres of the markers of one colour, filled or open (stroked only), inside `frame`, outside the
    `exclude` boxes (legends), sorted by x; duplicates (a fill and its outline) merged."""
    c = []
    for p in pp:
        if filled and not V.same_color(p.fill, color):
            continue
        if not filled and not (V.same_color(p.color, color) and p.fill is None):
            continue
        if not _shape_ok(p, shape):
            continue
        lo, hi = p.points.min(0), p.points.max(0)
        if np.any(hi - lo > max_size) or np.any(hi - lo < 1.0):
            continue
        m = 0.5 * (lo + hi)
        if not (frame[0] - 3 <= m[0] <= frame[2] + 3 and frame[1] - 3 <= m[1] <= frame[3] + 3):
            continue
        if any(b[0] <= m[0] <= b[2] and b[1] <= m[1] <= b[3] for b in exclude):
            continue
        c.append(m)
    out = []
    for m in sorted(c, key=lambda v: (v[0], v[1])):
        if not out or np.hypot(*(m - out[-1])) > 0.3:
            out.append(m)
    return np.array(out)


def error_bars(pp, color, centres, all_centres, frame=None):
    """For each marker centre: (low_y, high_y) in page points from the error-bar caps of that colour, or None.
    Each vertical bar starts a fixed gap outside its marker (the figure's style): the gap is the most frequent, over all
    bars, of the smallest marker-to-end distance above 2.3 pt (inside a marker is never a bar's start), and a bar
    belongs to the marker whose distance to the bar's nearer end is closest to that gap."""
    segs = [p.points for p in pp if V.same_color(p.color, color) and p.n_items == 1 and len(p.points) == 2
            and abs(p.points[0][0] - p.points[1][0]) < 0.05 and abs(p.points[0][1] - p.points[1][1]) > 0.05
            and (frame is None or (frame[1] - 1 <= p.points[:, 1].min() and p.points[:, 1].max() <= frame[3] + 1))]
    allc = np.asarray(all_centres)
    cand = []
    for s in segs:
        same = [j for j, m in enumerate(allc) if abs(m[0] - s[0][0]) < 0.3]
        ds = [(min(abs(s[0][1] - allc[j][1]), abs(s[1][1] - allc[j][1])), j) for j in same]
        ds = [(d, j) for d, j in ds if d > 2.3]
        if ds:
            cand.append((s, ds))
    if not cand:
        return [None] * len(centres)
    r = np.round([min(ds)[0] for _, ds in cand], 1)
    vals, counts = np.unique(r, return_counts=True)
    gap = float(vals[np.argmax(counts)])                       # the most frequent (the smallest on a tie)
    res = {i: [] for i in range(len(centres))}
    for s, ds in cand:
        d, j = min(ds, key=lambda t: abs(t[0] - gap))
        if abs(d - gap) > 1.0:
            continue
        inner = s[0] if abs(s[0][1] - allc[j][1]) < abs(s[1][1] - allc[j][1]) else s[1]
        outer = s[1] if inner is s[0] else s[0]
        for i, m in enumerate(centres):
            if np.hypot(*(m - allc[j])) < 0.3:
                res[i].append(outer[1])
    return [(max(ys), min(ys)) if len(ys) == 2 else None for ys in (res[i] for i in range(len(centres)))]


def dose_series(pp, xmap, ymap, color, frame, exclude=(), bars=True, shape=ANY):
    """The filled and the open markers of one colour, as two dose series (an open marker drawn at the same place as
    a filled one is the filled one's outline and is dropped), each with its error bars (min / max) when drawn."""
    fp = markers(pp, color, True, frame, exclude, shape=shape)
    op = markers(pp, color, False, frame, exclude, shape=shape)
    if len(fp) and len(op):
        op = np.array([m for m in op if not any(np.hypot(*(m - q)) < 0.3 for q in fp)]).reshape(-1, 2)
    allc = np.vstack([q for q in (fp, op) if len(q)]) if (len(fp) or len(op)) else np.zeros((0, 2))
    out = []
    for pts in (fp, op):
        d = {"dose_rad_SiO2": _r(xmap(pts[:, 0])) if len(pts) else [], "value": _r(ymap(pts[:, 1])) if len(pts) else []}
        if bars and len(pts):
            eb = error_bars(pp, color, pts, allc, frame)
            d["min"] = [None if b is None else float(f"{ymap(b[0]):.4g}") for b in eb]
            d["max"] = [None if b is None else float(f"{ymap(b[1]):.4g}") for b in eb]
        out.append((d, pts))
    return out


def lines(pp, color, frame, min_len=10):
    out = []
    for L in V.chain(pp, color):
        q = V.inside(V.dedupe(L), frame)
        if len(q) >= min_len:
            out.append(q[np.argsort(q[:, 0], kind="stable")])
    return out


def trace(pdfs, overlays):
    pdf = os.path.join(pdfs, "sutton_thesis.pdf")
    with open(OUT) as f:
        data = json.load(f)
    assert data["source"]["pdf_sha256"] == V.sha256(pdf), "not the PDF A23 traced"
    D = lambda x0, x1: V.Axis(x0, 5.0, x1, 7.0, log=True)          # noqa: E731  (1e5 to 1e7 rad(SiO2))
    ov = []

    # ---- Figs. 24 and 26: dIB/IB0 at VBE = 0.6 V (filled blue circles) and at JC = 1 uA/um^2 (open) -------------
    for key, page, frame, title in (("fig24", 86, (205.04, 193.77, 476.83, 374.96), "Figure 24"),
                                     ("fig26", 89, (205.04, 126.84, 476.83, 308.03), "Figure 26")):
        pp = V.paths(pdf, page)
        xm, ym = D(240.6, 476.83), V.Axis(frame[3], -2.0, frame[1], 2.0, log=True)
        legend = [(210, frame[1], 290, frame[1] + 80), (370, frame[3] - 40, 400, frame[3])]
        (fill, fp), (opn, op) = dose_series(pp, xm, ym, BLUE, frame, legend, shape=CIRCLE)
        mode = "forward" if key == "fig24" else "inverse"
        data[key] = {"figure": title, "page_index": page, "method": VERTEX, "device": "IBM 5AM, AE 0.50 x 1.0 um^2",
                     "conditions": f"63 MeV protons, all terminals grounded, 300 K; {mode} mode, VCB = 0 V; the 5AM "
                                   "series only; error bars are the min and max of 2 to 4 transistors",
                     "mode": mode, "axes": {"dose_rad_SiO2": xm.as_dict(), "dIB_over_IB0": ym.as_dict()},
                     "series": {"VBE_0.6V": fill, "JC_1uA_um2": opn}}
        ov.append((page, frame, [fp, op], f"sutton_{key}.png"))

    # ---- Figs. 23 and 25: dJB (the same points times JB0), for the trace check only -----------------------------
    for key, page, frame, title in (("fig23", 85, (209.87, 428.75, 476.98, 606.82), "Figure 23"),
                                     ("fig25", 88, (209.87, 216.68, 476.98, 394.76), "Figure 25")):
        pp = V.paths(pdf, page)
        xm, ym = D(244.82, 476.98), V.Axis(frame[3], -10.0, frame[1], -6.0, log=True)
        legend = [(210, frame[1], 290, frame[1] + 80), (370, frame[3] - 40, 400, frame[3]),
                  (210, frame[3] - 40, 290, frame[3])]
        (fill, fp), (opn, op) = dose_series(pp, xm, ym, BLUE, frame, legend, bars=False, shape=CIRCLE)
        data[key] = {"figure": title, "page_index": page, "method": VERTEX, "device": "IBM 5AM, AE 0.50 x 1.0 um^2",
                     "conditions": "dJB of the same points as " + ("Figure 24" if key == "fig23" else "Figure 26")
                                   + "; the axis is labelled uA/um^2 but its ticks run 1e-10 to 1e-6, so the "
                                     "values are as printed (the trace check uses ratios only)",
                     "axes": {"dose_rad_SiO2": xm.as_dict(), "dJB_as_printed": ym.as_dict()},
                     "series": {"VBE_0.6V": fill, "JC_1uA_um2": opn}}
        ov.append((page, frame, [fp, op], f"sutton_{key}.png"))

    # ---- Fig. 33: 5AM 0.5 x 2.5, three sources; filled = forward, open = inverse --------------------------------
    page, frame = 99, (206.71, 82.39, 476.92, 262.53)
    pp = V.paths(pdf, page)
    xm, ym = D(242.06, 476.92), V.Axis(262.53, -2.0, 82.39, 2.0, log=True)
    legend = [(350, 205, 380, 250)]
    s33, pts33 = {}, []
    for name, col in (("Co60", GREEN), ("Xray_10keV", CYAN), ("proton_63MeV", RED)):
        (f_, fp), (o_, op) = dose_series(pp, xm, ym, col, frame, legend)
        s33[name] = {"forward": f_, "inverse": o_}
        pts33 += [fp, op]
    data["fig33"] = {"figure": "Figure 33", "page_index": page, "method": VERTEX, "device": "IBM 5AM, AE 0.50 x 2.5 um^2",
                     "conditions": "dIB/IB0 at VBE = 0.6 V, VCB = 0 V, 300 K; 1.2 MeV Co-60 (30 rad/s), 10 keV X-ray "
                                   "(540 rad/s; dose in rad(SiO2) as plotted), 63 MeV protons (1 krad/s); error bars "
                                   "min / max of 2 to 4 transistors",
                     "axes": {"dose_rad_SiO2": xm.as_dict(), "dIB_over_IB0": ym.as_dict()}, "series": s33}
    ov.append((page, frame, pts33, "sutton_fig33.png"))

    # ---- Fig. 73: 5AM 0.5 x 1.0, forward, JC = 10 nA/um^2; 300 K open red circles, 77 K filled blue ----------------
    page, frame = 148, (205.04, 79.31, 476.83, 260.5)
    pp = V.paths(pdf, page)
    xm, ym = D(240.6, 476.83), V.Axis(260.5, -2.0, 79.31, 2.0, log=True)
    legend = [(210, 120, 240, 160)]
    _, (hot, hp) = dose_series(pp, xm, ym, RED, frame, legend, shape=CIRCLE)
    (cold, cp), _ = dose_series(pp, xm, ym, BLUE, frame, legend, shape=CIRCLE)
    data["fig73"] = {"figure": "Figure 73", "page_index": page, "method": VERTEX, "device": "IBM 5AM, AE 0.50 x 1.0 um^2",
                     "conditions": "63 MeV protons, irradiated and measured at the temperature; forward mode, "
                                   "dIB/IB0 at JC = 10 nA/um^2, VCB = 0 V; the 5AM series only (circles)",
                     "axes": {"dose_rad_SiO2": xm.as_dict(), "dIB_over_IB0": ym.as_dict()},
                     "series": {"300.0": hot, "77.0": cold}}
    ov.append((page, frame, [hp, cp], "sutton_fig73.png"))

    # ---- Fig. 22: inverse-mode Gummel, 5AM (blue solid): the two longest blue polylines ------------------------
    page, frame = 84, (207.33, 311.26, 477.54, 491.4)
    pp = V.paths(pdf, page)
    xm, ym = V.Axis(207.33, 0.0, 477.54, 1.2), V.Axis(491.4, -12.0, 311.26, 0.0, log=True)
    L = sorted(lines(pp, BLUE, (frame[0], frame[1], frame[2], frame[3] - 0.5)), key=len, reverse=True)
    L = [l for l in L if (l[:, 1].min() < 400)][:2]          # the legend sample is short and high up
    data["fig22"] = {"figure": "Figure 22", "page_index": page, "method": LINES, "device": "IBM 5AM, AE 0.50 x 1.0 um^2",
                     "conditions": "inverse-mode Gummel, pre-irradiation, VCB = 0 V, 300 K; currents per drawn AE; "
                                   "which curve is IC and which IB is not labelled: both are kept, named by their "
                                   "order at VBE = 0.6 V (upper, lower), and the A25 analysis assigns them",
                     "axes": {"VBE_V": xm.as_dict(), "J_A_um2": ym.as_dict()}}
    at = [float(np.interp(xm.inverse(0.6), l[:, 0], l[:, 1])) for l in L]
    order = np.argsort(at)                                    # smaller page y = larger current
    for name, i in zip(("upper", "lower"), order):
        data["fig22"][name] = {"VBE_V": _r(xm(L[i][:, 0]), 5), "J_A_um2": _r(ym(L[i][:, 1]))}
    ov.append((page, frame, L, "sutton_fig22.png"))

    # ---- Fig. 39: ideality of dIB vs VBE, 5AM 0.5 x 2.5, 3 Mrad; left panel forward, right inverse ---------------
    page = 106
    pp = V.paths(pdf, page)
    ym = V.Axis(382.66, 0.0, 176.8, 4.0)
    s39, pts39 = {}, []
    for mode, frame, x0, x1 in (("forward", (167.96, 176.8, 322.36, 382.66), 206.56, 283.76),
                                ("inverse", (322.36, 176.8, 476.75, 382.66), 360.95, 438.15)):
        xm = V.Axis(x0, 0.60, x1, 0.70)
        legend = [(380, 185, 400, 240)]
        for name, col in (("Co60", GREEN), ("Xray_10keV", CYAN), ("proton_63MeV", RED)):
            fr = (frame[0] + 0.5, frame[1], frame[2] - 0.5, frame[3])
            if mode == "forward":
                pts = markers(pp, col, True, fr, legend)
            else:
                pts = markers(pp, col, False, fr, legend)
                fill = markers(pp, col, True, fr, legend)
                pts = np.array([m for m in pts if not any(np.hypot(*(m - q)) < 0.3 for q in fill)]) if len(fill) else pts
            s39.setdefault(name, {})[mode] = {"VBE_V": _r(xm(pts[:, 0])), "n": _r(ym(pts[:, 1]))}
            pts39.append(pts)
    data["fig39"] = {"figure": "Figure 39", "page_index": page, "method": VERTEX, "device": "IBM 5AM, AE 0.50 x 2.5 um^2",
                     "conditions": "ideality of dIB vs VBE after 3 Mrad(SiO2), VCB = 0 V, 300 K; left panel forward "
                                   "(filled), right panel inverse (open)",
                     "axes": {"n": ym.as_dict()}, "series": s39}
    ov.append((page, (160, 170, 480, 390), pts39, "sutton_fig39.png"))

    # ---- Fig. 72: the post-irradiation curves (1, 3, 6 Mrad), 300 K and 77 K ------------------------------------
    page, frame = 147, (203.61, 217.48, 477.52, 400.09)
    pp = V.paths(pdf, page)
    xm, ym = V.Axis(203.61, 0.4, 477.52, 1.2), V.Axis(400.09, -10.0, 237.79, -2.0, log=True)
    pre = {c["T_K"]: c for c in data["fig72"]["curves"]}
    L = lambda c, k: (np.asarray(c["VBE_V_" + k[1]]), np.log10(np.asarray(c[k + "_A_um2"])))   # noqa: E731

    def near(v, lj, T, k):
        x, y = L(pre[T], k)
        m = (v >= x.min()) & (v <= x.max())
        out = np.full(v.shape, np.inf)
        out[m] = np.abs(lj[m] - np.interp(v[m], x, y))
        return out
    post, best = [], {}
    for dose, col in ((1.0e6, GREEN), (3.0e6, CYAN), (6.0e6, RED)):
        segs = [q.points for q in pp if V.same_color(q.color, col) and q.n_items == 1 and len(q.points) == 2]
        P = V.inside(np.vstack(segs), (frame[0], frame[1], frame[2], frame[3] - 0.3)) if segs else np.zeros((0, 2))
        v, lj = xm(P[:, 0]), np.log10(ym(P[:, 1]))
        on_jc = (near(v, lj, 300.0, "JC") < 0.05) | (near(v, lj, 77.0, "JC") < 0.05)
        legend = (v < 0.55) & (lj > -7.0)          # the legend's line samples, top left (1e-5 to 1e-3 A/um^2)
        on_jc |= legend
        d300, d77 = near(v, lj, 300.0, "JB"), near(v, lj, 77.0, "JB")
        T = np.where(v < 0.9, 300.0, np.where(d77 < d300, 77.0, 300.0))
        for TT in (300.0, 77.0):
            m = (~on_jc) & (T == TT)
            if m.sum() < 3:
                continue
            o = np.argsort(v[m], kind="stable")
            vv, jj = v[m][o], lj[m][o]
            keep = np.concatenate(([True], np.diff(vv) > 1e-4))
            best[(dose, TT)] = {"dose_rad_SiO2": dose, "T_K": TT, "kind": "JB", "n": int(keep.sum()),
                                "VBE_V": _r(vv[keep], 5), "J_A_um2": _r(10 ** jj[keep])}
    data["fig72"]["post_irradiation"] = sorted(best.values(), key=lambda c: (c["T_K"], c["dose_rad_SiO2"], c["kind"]))
    data["fig72"]["post_note"] = ("A25.1(e): 1.0 (green dotted), 3.0 (cyan dashed) and 6.0 (red dash-dot) Mrad(SiO2) "
                                  "curves, irradiated and measured at 300 K and 77 K: every segment of the colour; points on "
                                  "the pre-irradiation JC (|dlog10| < 0.05) are JC and dropped (JC does not change), as are the "
                                  "legend's line samples (VBE < 0.55 V, J > 1e-7 A/um^2), the "
                                  "rest is JB, at 300 K below 0.9 V and above it whichever pre-irradiation JB is nearer; "
                                  "the points sorted by VBE. " + VERTEX)
    ov.append((page, frame, [np.column_stack([xm.inverse(c["VBE_V"]), ym.inverse(c["J_A_um2"])])
                             for c in best.values()], "sutton_fig72_post.png"))

    # ---- checks: Figs. 23 / 25 = Figs. 24 / 26 x JB0, and which Fig. 22 curve is IB ---------------------------
    def jb0(dj, ratio):
        r = np.asarray(dj["value"]) / np.asarray(ratio["value"])
        return float(np.exp(np.mean(np.log(r)))), float(np.max(np.abs(np.log(r / np.exp(np.mean(np.log(r)))))))
    chk = {}
    for a, b in (("fig23", "fig24"), ("fig25", "fig26")):
        for s in ("VBE_0.6V", "JC_1uA_um2"):
            chk[f"{a}/{b} {s}"] = jb0(data[a]["series"][s], data[b]["series"][s])
    pre300 = {c["T_K"]: c for c in data["fig72"]["curves"]}[300.0]
    jb_fwd = float(10 ** np.interp(0.6, pre300["VBE_V_B"], np.log10(pre300["JB_A_um2"])))
    k_fwd = chk["fig23/fig24 VBE_0.6V"][0] / jb_fwd
    at = {n: float(10 ** np.interp(0.6, data["fig22"][n]["VBE_V"], np.log10(data["fig22"][n]["J_A_um2"])))
          for n in ("upper", "lower")}
    k_inv = {n: chk["fig25/fig26 VBE_0.6V"][0] / at[n] for n in at}
    ib = min(k_inv, key=lambda n: abs(np.log(k_inv[n] / k_fwd)))
    data["fig22"]["assignment"] = {
        "IB": ib, "IC": "upper" if ib == "lower" else "lower",
        "why": ("Figs. 23 / 24 give the forward JB0 at 0.6 V as %.3g (as printed), %.2f x Fig. 72's pre-irradiation "
                "JB there (%.3g A/um^2); Figs. 25 / 26 give the inverse JB0 as %.3g, which is %.2f x the %s curve "
                "of Fig. 22 at 0.6 V and %.2f x the other: the curve with the same factor as the forward case is IB"
                % (chk["fig23/fig24 VBE_0.6V"][0], k_fwd, jb_fwd, chk["fig25/fig26 VBE_0.6V"][0], k_inv[ib], ib,
                   k_inv["upper" if ib == "lower" else "lower"]))}
    data["trace_checks_A25"] = {k: {"JB0_as_printed": float(f"{v[0]:.4g}"), "max_abs_dln": float(f"{v[1]:.3g}")}
                                for k, v in chk.items()}

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    if overlays:
        for page, frame, series, name in ov:
            clip = (frame[0] - 40, frame[1] - 15, frame[2] + 15, frame[3] + 30)
            V.overlay(pdf, page, clip, [s for s in series if len(s)], os.path.join(overlays, name))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--overlays", default=None)
    a = ap.parse_args()
    if a.overlays:
        os.makedirs(a.overlays, exist_ok=True)
    trace(a.pdfs, a.overlays)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
