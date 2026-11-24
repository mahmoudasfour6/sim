"""Slope-continuity tracer for scanned line plots (digitising Ghante et al. 1995)."""
import numpy as np
from PIL import Image, ImageDraw

def load(path):
    return np.array(Image.open(path).convert("L")) < 128

def runs_in_col(b, x, y0, y1):
    col = b[y0:y1, x].astype(int)
    d = np.diff(np.concatenate(([0], col, [0])))
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return [(s + y0, e - 1 + y0) for s, e in zip(st, en)]

def trace(b, seed, x_end, frame, step=1, maxjump=18, hist=12, gap=12, anchors=None, thick=6):
    """Follow a curve from seed=(x,y) to x_end; frame=(x0,x1,y0,y1) interior bounds. anchors: dict x->y that
    override the prediction near those columns (to steer through crossings)."""
    x0, x1, y0, y1 = frame
    pts = [(seed[0], float(seed[1]))]
    x = seed[0]; misses = 0
    while (x - x_end) * step < 0:
        x += step
        if not (x0 <= x <= x1):
            break
        P = np.array(pts[-hist:], float)
        if len(P) >= 3:
            k = np.polyfit(P[:, 0], P[:, 1], 1)[0]
        else:
            k = 0.0
        ypred = pts[-1][1] + k * step * (1 + misses)
        if anchors:
            near = [ax for ax in anchors if abs(ax - x) <= 3]
            if near:
                ypred = anchors[near[0]]
        rs = runs_in_col(b, x, y0, y1)
        best = None
        for s, e in rs:
            yc = min(max(ypred, s), e)               # closest point of the run to the prediction
            dist = abs(yc - ypred)
            if e - s <= thick:                        # a plain line: its centre
                yc = 0.5 * (s + e)
                dist = abs(yc - ypred)
            if best is None or dist < best[1]:
                best = (yc, dist)
        if best is not None and best[1] <= maxjump * (1 + 0.5 * misses):
            pts.append((x, float(best[0]))); misses = 0
        else:
            misses += 1
            if misses > gap:
                break
    return pts

def overlay(path, curves, out, box):
    im = Image.open(path).convert("RGB").crop(box)
    d = ImageDraw.Draw(im)
    cols = [(230, 0, 0), (0, 160, 0), (0, 0, 230), (200, 0, 200), (0, 170, 170), (230, 130, 0), (120, 60, 0)]
    for i, (name, pts) in enumerate(curves):
        c = cols[i % len(cols)]
        P = [(x - box[0], y - box[1]) for x, y in pts]
        if len(P) > 1:
            d.line(P, fill=c, width=2)
        d.text((P[0][0] + 3, P[0][1] - 12), name, fill=c)
    im.save(out)
