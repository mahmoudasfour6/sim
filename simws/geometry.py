"""One geometry source for every engine (0.9b): a 2D device cross-section described once, meshed with gmsh, loaded
into DEVSIM and scikit-fem from the same .msh file, and summarised for the Geant4 track tool.

    from simws.geometry import Geometry
    g = Geometry("diode")
    g.rect("si", "Silicon", 0, 1.0, 0, 0.5)                     # um
    g.contact("anode", 0, 0, 0, 0.5); g.contact("cathode", 1.0, 1.0, 0, 0.5)
    g.refine(0.5, 0.25, size=0.002, radius=0.05)                # local mesh size (um) near a point
    path = g.mesh("diode.msh", size=0.02)
    g.to_devsim(path, device="d")                               # regions + contacts, finalized, device created
    mesh, subdomains, boundaries = g.to_skfem(path)             # skfem MeshTri (um -> cm), cell/facet index sets
    g.geant4_slab(x_um=0.5)                                      # (G4 material, thickness um) along y at x

Regions are axis-aligned rectangles fragmented together (shared edges conform); contacts are straight boundary
segments. Physical groups carry the names, so both loaders see the same regions and contacts. Coordinates are um in
the description and cm in the .msh file and the engines (DEVSIM and the FEM models use cm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

UM = 1e-4
G4_MATERIAL = {"Silicon": "G4_Si", "Si": "G4_Si", "SiO2": "G4_SILICON_DIOXIDE", "Oxide": "G4_SILICON_DIOXIDE",
               "Al2O3": "G4_ALUMINUM_OXIDE", "Aluminum": "G4_Al", "Copper": "G4_Cu", "Tungsten": "G4_W"}


@dataclass
class Geometry:
    name: str
    rects: List[Tuple[str, str, float, float, float, float]] = field(default_factory=list)
    contacts: List[Tuple[str, float, float, float, float]] = field(default_factory=list)
    fields: List[Tuple[float, float, float, float]] = field(default_factory=list)
    sizes: Dict[int, Tuple[float, float]] = field(default_factory=dict)       # rect index -> (hx, hy)

    def rect(self, name, material, x0, x1, y0, y1, size=None):
        """size: spacing (um) for a structured mesh in this rectangle, a number or (hx, hy); default: the global size,
        capped at a quarter of the rectangle's thinner side."""
        if any(n == name and m != material for n, m, *_ in self.rects):
            raise ValueError(f"region {name!r} already has another material")
        self.rects.append((name, material, float(x0), float(x1), float(y0), float(y1)))
        if size is not None:
            self.sizes[len(self.rects) - 1] = (float(size), float(size)) if np.isscalar(size) else (float(size[0]), float(size[1]))
        return self

    def contact(self, name, x0, x1, y0, y1):
        if not (x0 == x1 or y0 == y1):
            raise ValueError("a contact is a straight boundary segment (x0 == x1 or y0 == y1)")
        self.contacts.append((name, float(x0), float(x1), float(y0), float(y1)))
        return self

    def refine(self, x, y, size, radius):
        """Mesh size `size` (um) within `radius` (um) of (x, y), growing linearly to the global size over 4 radius."""
        self.fields.append((float(x), float(y), float(size), float(radius)))
        return self

    @property
    def regions(self) -> List[str]:
        """Region names in order of first appearance (a region may be several rectangles)."""
        return list(dict.fromkeys(r[0] for r in self.rects))

    def material_of(self, region):
        return next(m for n, m, *_ in self.rects if n == region)

    # ------------------------------------------------------------------------------------------- gmsh
    def mesh(self, path, size=0.05, order=1, msh_version=2.2, structured=False) -> Path:
        """structured=True: transfinite right-triangle meshes, uniform along x and along y inside each rectangle (rect()
        size, else `size` capped at a quarter of the rectangle's thinner side; the finest x spacing propagates along
        columns and the finest y spacing along rows, so meshes stay conformal). No triangle is obtuse, so DEVSIM's
        control volumes tile every region exactly (an unstructured mesh with size gradients has obtuse triangles, and
        the volumes then overlap by up to ~2 %). refine() is ignored here: grade a structured mesh by splitting a
        region into rectangles with their own size. Every fragment must be a 4-sided brick."""
        import gmsh
        path = Path(path)
        gmsh.initialize(["-noenv"])
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(self.name)
            occ = gmsh.model.occ
            tags = [occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0) for _, _, x0, x1, y0, y1 in self.rects]
            if len(tags) > 1:
                out, out_map = occ.fragment([(2, t) for t in tags], [])
            else:                                         # nothing to fragment (OCC returns no map for one shape)
                out_map = [[(2, tags[0])]]
            occ.synchronize()
            # region -> surfaces after fragmentation (a rectangle may be split by its neighbours' edges)
            surf_of = {}
            for (name, *_), pieces in zip(self.rects, out_map):
                surf_of.setdefault(name, set()).update(t for d, t in pieces if d == 2)
            for name in self.regions:
                gmsh.model.addPhysicalGroup(2, sorted(surf_of[name]), name=name)
            # contacts: boundary curves lying on the segment
            for cname, x0, x1, y0, y1 in self.contacts:
                eps = 1e-6                                    # OCC bounding boxes carry a ~1e-7 tolerance
                cur = []
                for d, t in gmsh.model.getEntities(1):
                    bx0, by0, _, bx1, by1, _ = gmsh.model.getBoundingBox(d, t)
                    if (bx0 >= min(x0, x1) - eps and bx1 <= max(x0, x1) + eps and by0 >= min(y0, y1) - eps
                            and by1 <= max(y0, y1) + eps):
                        cur.append((d, t))
                if not cur:
                    raise ValueError(f"contact {cname}: no boundary curve on the segment")
                gmsh.model.addPhysicalGroup(1, [t for _, t in cur], name=cname)
            # interfaces between regions (named region1_region2), for DEVSIM interfaces
            names = self.regions
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    ca = {c for s in surf_of[a] for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)}
                    cb = {c for s in surf_of[b] for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)}
                    shared = sorted(ca & cb)
                    if shared:
                        gmsh.model.addPhysicalGroup(1, shared, name=f"{a}_{b}")
            if structured:
                self._transfinite(gmsh, out_map, size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
            # every region gets at least ~4 elements across its thinnest dimension (thin layers otherwise get
            # obtuse triangles whose circumcentres fall outside, and DEVSIM's control volumes leak across regions)
            fl = []
            for _, _, x0, x1, y0, y1 in self.rects:
                h = min(size, min(x1 - x0, y1 - y0) / 4)
                if h < size:
                    fb = gmsh.model.mesh.field.add("Box")
                    for k, v in (("VIn", h), ("VOut", size), ("XMin", x0), ("XMax", x1), ("YMin", y0), ("YMax", y1),
                                 ("Thickness", 4 * h)):
                        gmsh.model.mesh.field.setNumber(fb, k, v)
                    fl.append(fb)
            if (self.fields or fl) and not structured:
                for k, (x, y, s, r) in enumerate(self.fields):
                    p = gmsh.model.occ.addPoint(x, y, 0)
                    occ.synchronize()
                    fd = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(fd, "PointsList", [p])
                    ft = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                    gmsh.model.mesh.field.setNumber(ft, "SizeMin", s)
                    gmsh.model.mesh.field.setNumber(ft, "SizeMax", size)
                    gmsh.model.mesh.field.setNumber(ft, "DistMin", r)
                    gmsh.model.mesh.field.setNumber(ft, "DistMax", 4 * r + r)
                    fl.append(ft)
                fm = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fl)
                gmsh.model.mesh.field.setAsBackgroundMesh(fm)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(order)
            gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.option.setNumber("Mesh.ScalingFactor", UM)          # the file is in cm, as both engines use
            gmsh.write(str(path))
        finally:
            gmsh.finalize()
        return path

    def _transfinite(self, gmsh, out_map, size):
        import math
        hcurve = {}                                      # curve -> (axis, spacing); axis 0 = horizontal, 1 = vertical
        surfs = []
        for k, ((name, _, x0, x1, y0, y1), pieces) in enumerate(zip(self.rects, out_map)):
            hx, hy = self.sizes.get(k, (min(size, (x1 - x0) / 4, (y1 - y0) / 4),) * 2)
            for d, t in pieces:
                if d != 2:
                    continue
                cs = [abs(c) for _, c in gmsh.model.getBoundary([(2, t)], oriented=False)]
                if len(cs) != 4:
                    raise ValueError(f"structured mesh: fragment of {name!r} has {len(cs)} sides; split the rectangles"
                                     " so that every side meets exactly one neighbour side")
                surfs.append((t, cs))
                for c in cs:
                    bx0, by0, _, bx1, by1, _ = gmsh.model.getBoundingBox(1, c)
                    ax, h = (0, hx) if abs(bx1 - bx0) > abs(by1 - by0) else (1, hy)
                    hcurve[c] = (ax, min(h, hcurve.get(c, (ax, h))[1]))
        for _ in range(4 * len(surfs) + 4):              # equalise opposite sides until stable
            changed = False
            for t, cs in surfs:
                for ax in (0, 1):
                    side = [c for c in cs if hcurve[c][0] == ax]
                    h = min(hcurve[c][1] for c in side)
                    for c in side:
                        if hcurve[c][1] > h:
                            hcurve[c] = (ax, h); changed = True
            if not changed:
                break
        for c, (ax, h) in hcurve.items():
            # length from the end points (OCC bounding boxes are padded by ~1e-7, which would add a node)
            ends = [gmsh.model.getValue(0, abs(t), []) for _, t in gmsh.model.getBoundary([(1, c)], oriented=False)]
            L = math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1])
            gmsh.model.mesh.setTransfiniteCurve(c, max(2, int(math.ceil(L / h - 1e-9)) + 1))
        for t, _ in surfs:
            gmsh.model.mesh.setTransfiniteSurface(t)

    # ------------------------------------------------------------------------------------------- engines
    def to_devsim(self, path, device="d", mesh_name=None, materials: Dict[str, str] = None):
        """Load the .msh into DEVSIM (scaled um -> cm), with every region, contact and interface; creates the device."""
        from devsim import (add_gmsh_contact, add_gmsh_interface, add_gmsh_region, create_device, create_gmsh_mesh,
                            finalize_mesh)
        m = mesh_name or f"{self.name}_mesh"
        create_gmsh_mesh(mesh=m, file=str(path))
        for name in self.regions:
            material = self.material_of(name)
            add_gmsh_region(mesh=m, gmsh_name=name, region=name, material=(materials or {}).get(name, material))
        for cname, x0, x1, y0, y1 in self.contacts:
            region = self._region_at_segment(x0, x1, y0, y1)
            add_gmsh_contact(mesh=m, gmsh_name=cname, region=region, name=cname, material="metal")
        names = self.regions
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if self._touch(a, b):
                    add_gmsh_interface(mesh=m, gmsh_name=f"{a}_{b}", region0=a, region1=b, name=f"{a}_{b}")
        finalize_mesh(mesh=m)
        create_device(mesh=m, device=device)
        return device

    def to_skfem(self, path):
        """(MeshTri in cm, {region: element indices}, {contact: facet indices}) from the same .msh."""
        import meshio
        import numpy as np
        from skfem import MeshTri
        mm = meshio.read(str(path))
        pts = mm.points[:, :2]
        tri_blocks = [(i, c.data) for i, c in enumerate(mm.cells) if c.type == "triangle"]
        tris = np.vstack([d for _, d in tri_blocks])
        mesh = MeshTri(pts.T.copy(), tris.T.copy())
        phys = mm.cell_data_dict.get("gmsh:physical", {})
        names = {tag: name for name, (tag, dim) in mm.field_data.items()}
        subdomains, offset = {}, 0
        tri_phys = phys.get("triangle")
        for name, (tag, dim) in mm.field_data.items():
            if dim == 2:
                subdomains[name] = np.nonzero(tri_phys == tag)[0]
        boundaries = {}
        if "line" in phys:
            lines = np.vstack([c.data for c in mm.cells if c.type == "line"])
            facet_of = {tuple(sorted(f)): k for k, f in enumerate(mesh.facets.T)}
            for name, (tag, dim) in mm.field_data.items():
                if dim == 1 and name in {c[0] for c in self.contacts}:
                    sel = lines[phys["line"] == tag]
                    boundaries[name] = np.array([facet_of[tuple(sorted(l))] for l in sel])
        del names, offset
        return mesh, subdomains, boundaries

    def geant4_slab(self, x_um: float):
        """(Geant4 material, thickness um) of the region stack along y at x, for the single-material track tool;
        raises if the stack holds more than one material (the track tool simulates one slab)."""
        stack = sorted((y0, y1, m) for _, m, x0, x1, y0, y1 in self.rects if x0 <= x_um <= x1)
        mats = {m for *_, m in stack}
        if len(mats) != 1:
            raise ValueError(f"the stack at x = {x_um} um has {len(mats)} materials; the Geant4 track tool takes one")
        m = mats.pop()
        return G4_MATERIAL.get(m, m), stack[-1][1] - stack[0][0]

    # ------------------------------------------------------------------------------------------- helpers
    def _region_at_segment(self, x0, x1, y0, y1):
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        for name, _, a0, a1, b0, b1 in self.rects:
            if a0 - 1e-12 <= xm <= a1 + 1e-12 and b0 - 1e-12 <= ym <= b1 + 1e-12:
                return name
        raise ValueError("contact segment not on any region")

    def _touch(self, a, b):
        for _, _, ax0, ax1, ay0, ay1 in (r for r in self.rects if r[0] == a):
            for _, _, bx0, bx1, by0, by1 in (r for r in self.rects if r[0] == b):
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if (abs(ox) < 1e-12 and oy > 0) or (abs(oy) < 1e-12 and ox > 0):
                    return True
        return False
