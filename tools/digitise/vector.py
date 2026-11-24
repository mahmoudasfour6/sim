"""Read plotted data straight from a PDF's drawing operators (vector figures), contract A23.1.

A vector figure stores each plotted line as a path whose vertices are the data points (in page points). We take
those vertices, and calibrate the axes on the tick marks drawn in the same stream, so no pixel is guessed:

    from tools.digitise import vector as V
    pp = V.paths("thesis.pdf", 14, clip=(150, 80, 470, 330))          # page index from 0; clip in page points
    xmap = V.Axis(181.56, 0.5, 434.43, 1.2)                           # two tick positions and their values
    ymap = V.Axis(295.68, -12, 92.88, 0, log=True)                    # log axes take log10 values
    line = V.longest(pp, color=(0, 0, 0))                             # a series: its vertices
    vbe, ic = xmap(line[:, 0]), ymap(line[:, 1])
    V.overlay("thesis.pdf", 14, (150, 80, 470, 330), [line], "check.png")   # the trace drawn over the page

The trace error is the line width (reported by `paths`), converted with the axis map.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Path:
    color: Optional[Tuple[float, ...]]
    fill: Optional[Tuple[float, ...]]
    width: float
    points: np.ndarray        # (n, 2) vertices in page points, in drawing order
    n_items: int
    closed: bool


def paths(pdf: str, page: int, clip: Optional[Sequence[float]] = None, overlap: bool = False) -> List[Path]:
    """Every drawn path on the page (inside `clip` = (x0, y0, x1, y1) if given), as its vertex list.
    overlap=True keeps paths that only overlap the clip (curves drawn past a plot's frame and clipped there; their
    vertices outside the frame are not data shown on the page and are dropped by `inside`)."""
    import pymupdf
    doc = pymupdf.open(pdf)
    out = []
    for d in doc[page].get_drawings():
        r = d["rect"]
        if clip is not None:
            inside_clip = (r.x0 >= clip[0] - 1 and r.y0 >= clip[1] - 1 and r.x1 <= clip[2] + 1
                           and r.y1 <= clip[3] + 1)
            overlaps = r.x1 >= clip[0] and r.x0 <= clip[2] and r.y1 >= clip[1] and r.y0 <= clip[3]
            if not (inside_clip or (overlap and overlaps)):
                continue
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if not pts or (abs(pts[-1][0] - a.x) > 1e-6 or abs(pts[-1][1] - a.y) > 1e-6):
                    pts.append((a.x, a.y))
                pts.append((b.x, b.y))
            elif it[0] == "c":
                a, b = it[1], it[4]
                if not pts or (abs(pts[-1][0] - a.x) > 1e-6 or abs(pts[-1][1] - a.y) > 1e-6):
                    pts.append((a.x, a.y))
                pts.append((b.x, b.y))
            elif it[0] == "re":
                q = it[1]
                pts += [(q.x0, q.y0), (q.x1, q.y0), (q.x1, q.y1), (q.x0, q.y1)]
        if not pts:
            continue
        col = tuple(round(c, 3) for c in d["color"]) if d.get("color") else None
        fil = tuple(round(c, 3) for c in d["fill"]) if d.get("fill") else None
        out.append(Path(col, fil, float(d.get("width") or 0.0), np.array(pts, float), len(d["items"]),
                        bool(d.get("closePath"))))
    return out


def inside(points: np.ndarray, frame: Sequence[float], tol: float = 0.05) -> np.ndarray:
    """The vertices inside a plot frame (x0, y0, x1, y1), in order: what the page shows of a clipped curve."""
    p = np.asarray(points, float)
    m = ((p[:, 0] >= frame[0] - tol) & (p[:, 0] <= frame[2] + tol) & (p[:, 1] >= frame[1] - tol)
         & (p[:, 1] <= frame[3] + tol))
    return p[m]


def same_color(a, b, tol=0.02) -> bool:
    return a is not None and b is not None and len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def by_color(pp: List[Path], color, min_items: int = 1) -> List[Path]:
    return [p for p in pp if same_color(p.color, color) and p.n_items >= min_items]


def longest(pp: List[Path], color) -> np.ndarray:
    """The vertices of the path of that colour with the most segments (a plotted curve)."""
    cand = by_color(pp, color)
    if not cand:
        raise ValueError(f"no path of colour {color}")
    return max(cand, key=lambda p: p.n_items).points


def joined(pp: List[Path], color, min_items: int = 2, gap: float = 0.6) -> np.ndarray:
    """Paths of one colour joined end to start (a curve drawn in pieces), in drawing order; pieces whose start is
    more than `gap` points from the previous end start a new run, and the longest run is returned."""
    cand = by_color(pp, color, min_items)
    runs, cur = [], None
    for p in cand:
        if cur is not None and np.hypot(*(p.points[0] - cur[-1])) <= gap:
            cur = np.vstack([cur, p.points[1:]])
        else:
            if cur is not None:
                runs.append(cur)
            cur = p.points.copy()
    if cur is not None:
        runs.append(cur)
    if not runs:
        raise ValueError(f"no path of colour {color}")
    return max(runs, key=len)


def dedupe(points: np.ndarray, tol: float = 0.2) -> np.ndarray:
    """Drop vertices closer than `tol` page points to the one kept before them (a path's repeated start)."""
    pts = np.asarray(points, float)
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > tol:
            keep.append(i)
    return pts[keep]


def chain(pp: List[Path], color, tol: float = 0.05, max_items: int = 1) -> List[np.ndarray]:
    """Curves drawn as separate short segments (one path per segment), linked end to start into polylines (each
    returned with its vertices in order), longest first."""
    segs = [p.points for p in by_color(pp, color) if p.n_items <= max_items and len(p.points) >= 2]
    used = [False] * len(segs)
    starts = {}
    for i, s in enumerate(segs):
        starts.setdefault((round(s[0][0] / tol), round(s[0][1] / tol)), []).append(i)
    ends = {}
    for i, s in enumerate(segs):
        ends.setdefault((round(s[-1][0] / tol), round(s[-1][1] / tol)), []).append(i)
    out = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        line = [tuple(v) for v in segs[i]]
        for forward in (True, False):
            while True:
                key = (round(line[-1][0] / tol), round(line[-1][1] / tol)) if forward else \
                      (round(line[0][0] / tol), round(line[0][1] / tol))
                nxt = None
                for k in ([key] + [(key[0] + a, key[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
                    for j in (starts if forward else ends).get(k, []):
                        if not used[j]:
                            nxt = j
                            break
                    if nxt is not None:
                        break
                if nxt is None:
                    break
                used[nxt] = True
                if forward:
                    line += [tuple(v) for v in segs[nxt][1:]]
                else:
                    line = [tuple(v) for v in segs[nxt][:-1]] + line
        out.append(np.array(line))
    return sorted(out, key=len, reverse=True)


def marker_centres(pp: List[Path], color, max_size: float = 8.0) -> np.ndarray:
    """Centres of the small closed markers (circles, squares, diamonds) of a colour, sorted by x."""
    c = []
    for p in pp:
        if not (same_color(p.color, color) or same_color(p.fill, color)):
            continue
        lo, hi = p.points.min(0), p.points.max(0)
        if np.all(hi - lo <= max_size) and np.all(hi - lo > 0.5):
            c.append(0.5 * (lo + hi))
    c = np.array(c) if c else np.zeros((0, 2))
    return c[np.argsort(c[:, 0])] if len(c) else c


@dataclass
class Axis:
    """Page coordinate -> data value, from two tick marks. log=True: values are log10 of the data (the map
    returns the data itself)."""
    p0: float
    v0: float
    p1: float
    v1: float
    log: bool = False

    def __call__(self, p):
        v = self.v0 + (np.asarray(p, float) - self.p0) * (self.v1 - self.v0) / (self.p1 - self.p0)
        return 10.0 ** v if self.log else v

    def inverse(self, v):
        v = np.log10(np.asarray(v, float)) if self.log else np.asarray(v, float)
        return self.p0 + (v - self.v0) * (self.p1 - self.p0) / (self.v1 - self.v0)

    def per_point(self) -> float:
        """Data units (decades on a log axis) per page point."""
        return abs((self.v1 - self.v0) / (self.p1 - self.p0))

    def as_dict(self):
        return {"page_pt": [self.p0, self.p1], "value": [self.v0, self.v1],
                "scale": "log10" if self.log else "linear"}


def overlay(pdf: str, page: int, clip: Sequence[float], series: Sequence[np.ndarray], out_png: str,
            dpi: int = 300, colors=("#00c000", "#ff8000", "#00a0ff", "#c000c0", "#808000", "#00c0c0")):
    """The page region rendered at dpi with each series' points drawn on top (small crosses), for the check."""
    import pymupdf
    from PIL import Image, ImageDraw
    doc = pymupdf.open(pdf)
    rect = pymupdf.Rect(*clip)
    pix = doc[page].get_pixmap(dpi=dpi, clip=rect)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(img)
    k = dpi / 72.0
    for i, s in enumerate(series):
        col = colors[i % len(colors)]
        for x, y in np.asarray(s):
            px, py = (x - clip[0]) * k, (y - clip[1]) * k
            dr.line([(px - 3, py), (px + 3, py)], fill=col, width=1)
            dr.line([(px, py - 3), (px, py + 3)], fill=col, width=1)
    img.save(out_png)
    return out_png
