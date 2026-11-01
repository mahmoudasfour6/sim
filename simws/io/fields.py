"""Field files shared between stages.

A field file is a VTU mesh (readable by ParaView/PyVista/meshio) plus a JSON sidecar `<name>.meta.json`
holding units for every field, the provenance stamp and free-form metadata. Writers must declare a unit for
every field; readers get them back, so a stage can never silently consume a field in the wrong unit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from simws.units import ureg

_CELL_TYPES = {3: "triangle", 4: "quad"}


@dataclass
class FieldSet:
    points: np.ndarray                     # (n, 2) or (n, 3), in `length_unit`
    cells: np.ndarray                      # (m, 3) triangles or (m, 4) quads, 0-based node indices
    length_unit: str
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def get(self, name: str, unit: Optional[str] = None) -> np.ndarray:
        data = np.asarray(self.point_data[name], dtype=float)
        if unit is None or unit == self.units[name]:
            return data
        return (data * ureg(self.units[name])).to(unit).magnitude

    def points_in(self, unit: str) -> np.ndarray:
        return (np.asarray(self.points, dtype=float) * ureg(self.length_unit)).to(unit).magnitude


def write(path, fs: FieldSet):
    import meshio
    missing = set(fs.point_data) - set(fs.units)
    if missing:
        raise ValueError(f"fields without a declared unit: {sorted(missing)}")
    for name, unit in fs.units.items():
        ureg.parse_units(unit)
    path = Path(path).with_suffix(".vtu")
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(fs.points, dtype=float)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    cells = np.asarray(fs.cells, dtype=int)
    meshio.write(str(path), meshio.Mesh(pts, [(_CELL_TYPES[cells.shape[1]], cells)],
                                        point_data={k: np.asarray(v, float) for k, v in fs.point_data.items()}))
    side = {"length_unit": fs.length_unit, "units": fs.units, "meta": fs.meta}
    path.with_suffix(".meta.json").write_text(json.dumps(side, indent=2, default=float) + "\n")
    return path


def read(path) -> FieldSet:
    import meshio
    path = Path(path).with_suffix(".vtu")
    m = meshio.read(str(path))
    side = json.loads(path.with_suffix(".meta.json").read_text())
    cells = next(c.data for c in m.cells if c.type in ("triangle", "quad"))
    pts = m.points[:, :2] if np.allclose(m.points[:, 2], 0) else m.points
    return FieldSet(pts, cells, side["length_unit"], dict(m.point_data), side["units"], side.get("meta", {}))
