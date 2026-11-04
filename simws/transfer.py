"""Conservative field transfer between 2D meshes (0.9b).

A field moves between engines as cell integrals: every source cell's integral is split among the target cells it
overlaps in proportion to the overlap areas (polygon intersection, shapely). The split fractions of each source cell
sum to one by construction, so the total integral is conserved to round-off whatever the meshes; any part of the
source outside the target is reported (`lost_fraction`), never silently dropped.

Fields come in three forms:
  cells   P0 values on polygons (Geant4 dose bins, FEM P0 fields)
  nodes   P1 values on a triangle mesh (FEM nodal fields): turned into exact cell averages first
  volumes DEVSIM node values with control volumes: integral = sum v_i * NodeVolume_i. Each node's integral is spread
          over its incident triangles by the triangle's share of that node's volume (DEVSIM's own split of the
          control volume is not exported, so the share is taken as the lumped split area/3; the transfer stays exactly
          conservative because the shares of each node are normalised to one)

Every transfer returns a TransferRecord: source and target integrals, the conservation error, the area lost
outside the target, and, if asked, the L2 round-trip error (target -> source) as the recorded transfer error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransferRecord:
    integral_source: float
    integral_target: float
    conservation_rel_error: float
    lost_fraction: float                      # share of the source area outside the target mesh (its integral is
                                              # carried by the overlapping part of the same cell)
    unmoved_integral: float = 0.0             # integral of source cells wholly outside the target (not moved)
    roundtrip_l2_rel: Optional[float] = None  # ||back - source||_L2 / ||source||_L2 on the source mesh
    n_source: int = 0
    n_target: int = 0
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def tri_polys(points, tris) -> np.ndarray:
    """(n, 3, 2) vertex array of a triangle mesh."""
    return np.asarray(points, float)[:, :2][np.asarray(tris)]


def rect_polys(xedges, yedges) -> np.ndarray:
    """(nx*ny, 4, 2) polygons of a rectilinear grid (e.g. a Geant4 dose histogram), ordered x fastest."""
    xe, ye = np.asarray(xedges, float), np.asarray(yedges, float)
    out = []
    for j in range(len(ye) - 1):
        for i in range(len(xe) - 1):
            out.append([(xe[i], ye[j]), (xe[i + 1], ye[j]), (xe[i + 1], ye[j + 1]), (xe[i], ye[j + 1])])
    return np.array(out)


def _areas(polys) -> np.ndarray:
    p = np.asarray(polys, float)
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y, axis=-1))


class Overlap:
    """Sparse overlap areas A[s, t] between two polygon sets, computed once and reused for every field."""

    def __init__(self, src_polys, tgt_polys):
        import shapely
        from scipy.sparse import csr_matrix
        S = shapely.polygons(np.asarray(src_polys, float))
        T = shapely.polygons(np.asarray(tgt_polys, float))
        tree = shapely.STRtree(T)
        si, ti = tree.query(S, predicate="intersects")
        a = shapely.area(shapely.intersection(S[si], T[ti]))
        keep = a > 0
        self.n_src, self.n_tgt = len(S), len(T)
        self.A = csr_matrix((a[keep], (si[keep], ti[keep])), shape=(self.n_src, self.n_tgt))
        self.area_src = _areas(src_polys)
        self.area_tgt = _areas(tgt_polys)
        covered = np.asarray(self.A.sum(axis=1)).ravel()
        self.lost = np.clip(1 - covered / self.area_src, 0, 1)
        self.lost_fraction = float(np.sum(self.lost * self.area_src) / np.sum(self.area_src))
        self.outside = covered <= 0
        # each source cell's split fractions over the target, normalised to one where the cell is covered at all
        with np.errstate(invalid="ignore", divide="ignore"):
            inv = np.where(covered > 0, 1.0 / covered, 0.0)
        from scipy.sparse import diags
        self.F = diags(inv) @ self.A                      # rows sum to 1 (or 0 for a cell wholly outside)

    def cell_integrals(self, src_integrals) -> np.ndarray:
        """Per-target-cell integrals from per-source-cell integrals (exactly conservative)."""
        return self.F.T @ np.asarray(src_integrals, float)


def p1_cell_average(points, tris, nodal) -> np.ndarray:
    """Exact cell means of a P1 field on triangles."""
    return np.asarray(nodal, float)[np.asarray(tris)].mean(axis=1)


def cell_to_lumped_nodes(points, tris, cell_values):
    """P0 -> nodal values with lumped mass (area/3 per vertex): sum v_i m_i equals the P0 integral exactly."""
    tris = np.asarray(tris)
    a = _areas(tri_polys(points, tris))
    n = len(points)
    m = np.bincount(tris.ravel(), weights=np.repeat(a / 3, 3), minlength=n)
    I = np.bincount(tris.ravel(), weights=np.repeat(a * np.asarray(cell_values, float) / 3, 3), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(m > 0, I / m, 0.0)
    return v, m


def _l2(values, areas):
    return float(np.sqrt(np.sum(np.asarray(values) ** 2 * areas)))


def remap_cells(src_polys, src_values, tgt_polys, roundtrip=True, overlap: Overlap = None):
    """P0 -> P0 conservative remap. Returns (target cell values, TransferRecord)."""
    ov = overlap or Overlap(src_polys, tgt_polys)
    v = np.asarray(src_values, float)
    Is = v * ov.area_src
    It = ov.cell_integrals(Is)
    vt = It / ov.area_tgt
    rec = _record(Is.sum(), It.sum(), ov, Is)
    if roundtrip:
        back = Overlap(tgt_polys, src_polys)
        vb = back.cell_integrals(It) / ov.area_src
        rec.roundtrip_l2_rel = _l2(vb - v, ov.area_src) / max(_l2(v, ov.area_src), 1e-300)
    return vt, rec


def remap_nodes(src_points, src_tris, src_nodal, tgt_points, tgt_tris, roundtrip=True):
    """P1 (triangles) -> target cells and lumped target nodes. Returns (tgt_cell, tgt_nodal, TransferRecord); the
    record's target integral is the lumped nodal integral, equal to the cell integral to round-off."""
    sp, tp = tri_polys(src_points, src_tris), tri_polys(tgt_points, tgt_tris)
    vc = p1_cell_average(src_points, src_tris, src_nodal)
    vt, rec = remap_cells(sp, vc, tp, roundtrip=roundtrip)
    vn, m = cell_to_lumped_nodes(tgt_points, tgt_tris, vt)
    rec.integral_target = float(np.sum(vn * m))
    moved = rec.integral_source - rec.unmoved_integral
    rec.conservation_rel_error = abs(rec.integral_target - moved) / max(abs(moved), 1e-300)
    return vt, vn, rec


def remap_control_volumes(points, tris, node_values, node_volumes, tgt_polys, roundtrip=True):
    """DEVSIM node field (integral = sum v_i V_i) -> target cells. The node's integral is split over its incident
    triangles by area/3 shares, normalised per node. Returns (tgt cell values, TransferRecord)."""
    tris = np.asarray(tris)
    a = _areas(tri_polys(points, tris))
    w = np.repeat(a / 3, 3).reshape(-1, 3)
    wsum = np.bincount(tris.ravel(), weights=w.ravel(), minlength=len(points))
    Inode = np.asarray(node_values, float) * np.asarray(node_volumes, float)
    share = w / wsum[tris]
    Icell = np.sum(share * Inode[tris], axis=1)
    src_polys = tri_polys(points, tris)
    ov = Overlap(src_polys, tgt_polys)
    It = ov.cell_integrals(Icell)
    rec = _record(Inode.sum(), It.sum(), ov, Icell)
    if roundtrip:
        back = Overlap(tgt_polys, src_polys)
        vb = back.cell_integrals(It) / ov.area_src
        vc = Icell / ov.area_src
        rec.roundtrip_l2_rel = _l2(vb - vc, ov.area_src) / max(_l2(vc, ov.area_src), 1e-300)
    return It / ov.area_tgt, rec


def _record(Is, It, ov: Overlap, cell_I) -> TransferRecord:
    unmoved = float(np.sum(np.asarray(cell_I)[ov.outside]))
    moved = float(Is) - unmoved
    r = TransferRecord(integral_source=float(Is), integral_target=float(It),
                       conservation_rel_error=abs(float(It) - moved) / max(abs(moved), 1e-300),
                       lost_fraction=ov.lost_fraction, unmoved_integral=unmoved, n_source=ov.n_src, n_target=ov.n_tgt)
    if ov.lost_fraction > 1e-12:
        r.notes.append(f"{ov.lost_fraction:.3e} of the source area lies outside the target")
    if unmoved:
        r.notes.append(f"source cells wholly outside the target hold {unmoved:.6e} of the integral (not moved)")
    return r


def check_conservation(rec: TransferRecord, tol=1e-8) -> bool:
    return bool(rec.lost_fraction == 0 and rec.conservation_rel_error < tol)


def cells_to_control_volumes(src_polys, src_values, points, tris, node_volumes, roundtrip=False):
    """P0 source (e.g. a Geant4 energy-deposition grid, per-volume density) -> DEVSIM node densities. Source integrals
    are remapped to the device triangles, each triangle's integral goes to its three nodes in thirds, and the node
    density is that integral over the node's control volume, so sum g_i NodeVolume_i equals the moved integral exactly.
    Returns (node densities, TransferRecord)."""
    tris = np.asarray(tris)
    tp = tri_polys(points, tris)
    ov = Overlap(src_polys, tp)
    Is = np.asarray(src_values, float) * ov.area_src
    It = ov.cell_integrals(Is)
    In = np.bincount(tris.ravel(), weights=np.repeat(It / 3, 3), minlength=len(points))
    V = np.asarray(node_volumes, float)
    g = np.where(V > 0, In / np.where(V > 0, V, 1), 0.0)
    rec = _record(Is.sum(), float(np.sum(g * V)), ov, Is)
    if np.any((V <= 0) & (In != 0)):
        rec.notes.append("nodes with zero control volume received an integral (dropped)")
    return g, rec
