"""0.9b checks of the shared geometry (simws.geometry), the conservative field transfer (simws.transfer) and the
preCICE bridge (simws.coupling.precice_bridge).

volumes            DEVSIM control volumes of a gmsh mesh sum to every region's area: structured (transfinite) to
                   1e-10 (gate); unstructured reported (obtuse triangles make the volumes overlap)
contacts           contact lengths on the skfem side equal the geometry's (gate 1e-12)
same_device        a 2D pn diode on the gmsh structured mesh against the same diode on DEVSIM's own mesh with the same
                   lines: terminal current at 0.3-0.6 V (gate 1e-8; the hypotenuse of a right triangle carries no
                   flux, so the two triangulations are the same finite-volume scheme)
geant4_slab        the material stack handed to the Geant4 track tool matches the geometry
transfer           conservation to 1e-8 (gate) and recorded L2 round-trip error for: a Geant4-style dose grid ->
                   DEVSIM node generation; a DEVSIM node field -> an unstructured skfem mesh; a P1 field between two
                   unrelated triangle meshes
precice            two-way electro-thermal coupling through preCICE: the same-mesh run reproduces the in-process loop
                   (1e-6), and on gmsh thermal meshes the heat transfer conserves power to 1e-8 and the thermal energy
                   balance closes to 1e-8; the T_max change between meshes is recorded as the transfer error
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

UM = 1e-4


def _mos(structured=True):
    from simws.geometry import Geometry
    g = Geometry("mos")
    if structured:
        g.rect("si", "Silicon", 0, 1.0, 0, 0.45, size=(0.02, 0.05)).rect("si", "Silicon", 0, 1.0, 0.45, 0.5,
                                                                          size=(0.02, 0.002))
        g.rect("ox", "Oxide", 0, 1.0, 0.5, 0.52, size=(0.02, 0.002))
    else:
        g.rect("si", "Silicon", 0, 1.0, 0, 0.5).rect("ox", "Oxide", 0, 1.0, 0.5, 0.52)
        g.refine(0.5, 0.5, size=0.005, radius=0.05)
    g.contact("sub", 0, 1.0, 0, 0).contact("gate", 0, 1.0, 0.52, 0.52)
    return g


def volumes(work) -> dict:
    Path(work).mkdir(parents=True, exist_ok=True)
    import simws.tcad  # noqa: F401
    from devsim import get_node_model_values, get_region_list, reset_devsim
    out = {}
    for label, st in (("structured", True), ("unstructured", False)):
        g = _mos(st)
        p = g.mesh(Path(work) / f"mos_{label}.msh", size=0.05, structured=st)
        reset_devsim(); g.to_devsim(p, device="d")
        area = {"si": 0.5, "ox": 0.02}
        rel = {r: sum(get_node_model_values(device="d", region=r, name="NodeVolume")) / (area[r] * UM * UM) - 1
               for r in get_region_list(device="d")}
        out[label] = {"rel_area_error": rel, "worst": max(abs(v) for v in rel.values())}
    out["pass"] = bool(out["structured"]["worst"] < 1e-10)
    return out


def contacts(work) -> dict:
    Path(work).mkdir(parents=True, exist_ok=True)
    g = _mos(True)
    p = g.mesh(Path(work) / "mos_c.msh", size=0.05, structured=True)
    mesh, sub, bnd = g.to_skfem(p)
    rows, worst = {}, 0.0
    for name, x0, x1, y0, y1 in g.contacts:
        f = mesh.facets[:, bnd[name]]
        L = float(np.sum(np.linalg.norm(mesh.p[:, f[0]] - mesh.p[:, f[1]], axis=0))) / UM
        ref = abs(x1 - x0) + abs(y1 - y0)
        rows[name] = {"length_um": L, "geometry_um": ref}
        worst = max(worst, abs(L / ref - 1))
    regions = {k: int(len(v)) for k, v in sub.items()}
    return {"contacts": rows, "region_triangles": regions, "worst_rel": worst, "pass": bool(worst < 1e-12)}


def _diode_current(build_mesh, Vs):
    import simws.tcad  # noqa: F401
    from devsim import get_contact_current
    from simws.materials import load
    from simws.tcad.physics import Physics, setup_region
    from simws.tcad.solve import ramp_bias
    build_mesh()
    setup_region("d", "si", load("silicon"), "ifelse(x < 0.505e-4, -1e17, 1e16)", "1.1e17",
                 Physics(bgn=False, srh=True, tau_n=1e-7, tau_p=1e-7, auger=False),
                 {"anode": False, "cathode": False})
    opts = dict(absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)
    out = []
    for V in Vs:
        ramp_bias("d", "anode", V, step=0.05, **opts)
        out.append(get_contact_current(device="d", contact="anode", equation="ElectronContinuityEquation")
                   + get_contact_current(device="d", contact="anode", equation="HoleContinuityEquation"))
    return np.array(out)


def same_device(work, Vs=(0.3, 0.45, 0.6)) -> dict:
    Path(work).mkdir(parents=True, exist_ok=True)
    from simws.geometry import Geometry
    hx, hy = 0.01, 0.1
    g = (Geometry("pn").rect("si", "Silicon", 0, 1.0, 0, 0.5, size=(hx, hy))
         .contact("anode", 0, 0, 0, 0.5).contact("cathode", 1.0, 1.0, 0, 0.5))
    p = g.mesh(Path(work) / "pn.msh", structured=True)

    def gm():
        from devsim import reset_devsim
        reset_devsim(); g.to_devsim(p, device="d")

    def native():
        from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                            finalize_mesh, reset_devsim)
        reset_devsim()
        create_2d_mesh(mesh="m")
        for i in range(int(round(1.0 / hx)) + 1):
            add_2d_mesh_line(mesh="m", dir="x", pos=i * hx * UM, ps=hx * UM)
        for j in range(int(round(0.5 / hy)) + 1):
            add_2d_mesh_line(mesh="m", dir="y", pos=j * hy * UM, ps=hy * UM)
        add_2d_mesh_line(mesh="m", dir="x", pos=-1e-8, ps=1e-8)          # DEVSIM 2D contacts sit between regions
        add_2d_mesh_line(mesh="m", dir="x", pos=UM + 1e-8, ps=1e-8)
        add_2d_region(mesh="m", material="Si", region="si", xl=0, xh=UM)
        add_2d_region(mesh="m", material="Si", region="pad_l", xl=-1e-8, xh=0)
        add_2d_region(mesh="m", material="Si", region="pad_r", xl=UM, xh=UM + 1e-8)
        add_2d_contact(mesh="m", name="anode", material="metal", region="si", xl=0, xh=0, yl=0, yh=0.5 * UM, bloat=1e-10)
        add_2d_contact(mesh="m", name="cathode", material="metal", region="si", xl=UM, xh=UM, yl=0, yh=0.5 * UM, bloat=1e-10)
        finalize_mesh(mesh="m"); create_device(mesh="m", device="d")

    Ig, In = _diode_current(gm, Vs), _diode_current(native, Vs)
    rel = np.abs(Ig / In - 1)
    return {"V": list(Vs), "I_gmsh": Ig.tolist(), "I_native": In.tolist(), "worst_rel": float(rel.max()),
            "pass": bool(rel.max() < 1e-8)}


def geant4_slab() -> dict:
    from simws.geometry import Geometry
    g = Geometry("slab").rect("a", "Silicon", 0, 1, 0, 2).rect("b", "Silicon", 0, 1, 2, 5)
    m, t = g.geant4_slab(0.5)
    try:
        _mos(False).geant4_slab(0.5)
        refused = False
    except ValueError:
        refused = True
    return {"material": m, "thickness_um": t, "mixed_stack_refused": refused,
            "pass": bool(abs(t - 5) < 1e-12 and refused)}


def transfer(work) -> dict:
    Path(work).mkdir(parents=True, exist_ok=True)
    import simws.tcad  # noqa: F401
    from devsim import get_element_node_list, get_node_model_values, reset_devsim
    from skfem import MeshTri
    from simws import transfer as T
    res = {}
    # (1) Geant4-style dose grid (per-volume density, um) -> DEVSIM node generation on the structured MOS mesh
    g = _mos(True)
    p = g.mesh(Path(work) / "mos_t.msh", size=0.05, structured=True)
    reset_devsim(); g.to_devsim(p, device="d")
    x = np.array(get_node_model_values(device="d", region="si", name="x"))
    y = np.array(get_node_model_values(device="d", region="si", name="y"))
    tris = np.array(get_element_node_list(device="d", region="si"), int)
    V = np.array(get_node_model_values(device="d", region="si", name="NodeVolume"))
    grid = T.rect_polys(np.linspace(0, UM, 41), np.linspace(0, 0.5 * UM, 21))
    cx = grid[:, :, 0].mean(1); cy = grid[:, :, 1].mean(1)
    dose = 1e20 * np.exp(-((cx - 0.4 * UM) ** 2) / (0.05 * UM) ** 2) * (1 + cy / UM)   # a track-like column
    G, rec = T.cells_to_control_volumes(grid, dose, np.column_stack([x, y]), tris, V)
    res["dose_grid_to_devsim"] = rec.as_dict()
    # (2) DEVSIM node field -> unstructured skfem mesh of the same rectangle
    gu = _mos(False)
    pu = gu.mesh(Path(work) / "mos_u.msh", size=0.05)
    mu, sub, _ = gu.to_skfem(pu)
    tgt = T.tri_polys(mu.p.T, mu.t.T[sub["si"]])
    field = 1e16 * np.exp(-((y - 0.5 * UM) / (0.05 * UM)) ** 2) + 1e15
    _, rec = T.remap_control_volumes(np.column_stack([x, y]), tris, field, V, tgt)
    res["devsim_to_skfem"] = rec.as_dict()
    # (3) P1 field between two unrelated meshes
    ms = MeshTri.init_tensor(np.linspace(0, 1, 41), np.linspace(0, 1, 41))
    mt = MeshTri.init_tensor(np.linspace(0, 1, 57) ** 1.3, np.linspace(0, 1, 50))
    f = lambda a, b: np.exp(-((a - 0.3) ** 2 + (b - 0.6) ** 2) / 0.02) + 0.3 * np.sin(3 * a)
    _, _, rec = T.remap_nodes(ms.p.T, ms.t.T, f(*ms.p), mt.p.T, mt.t.T)
    res["p1_to_p1"] = rec.as_dict()
    worst = max(r["conservation_rel_error"] for r in res.values())
    res["worst_conservation"] = worst
    res["pass"] = bool(worst < 1e-8 and all(r["lost_fraction"] < 1e-12 for k, r in res.items() if isinstance(r, dict)))
    return res


def precice(work) -> dict:
    Path(work).mkdir(parents=True, exist_ok=True)
    from simws.coupling import electrothermal as et
    from simws.coupling import precice_bridge as B
    case = et.ResistorCase(biases=[1.0, 5.0])
    ref = et.run(case)["points"]
    runs = {"same_mesh": "devsim", "gmsh_unstructured": {"size": 0.1}, "gmsh_structured": {"size": 0.07,
                                                                                            "structured": True}}
    out, ok = {}, True
    for label, spec in runs.items():
        r = B.run(case, thermal_mesh=spec, work=Path(work) / f"precice_{label}")
        rows = []
        for a, b in zip(r["points"], ref):
            rows.append({"bias_V": a["bias_V"], "T_max_K": a["T_max_K"], "T_max_inprocess_K": b["T_max_K"],
                         "dT_max_K": a["T_max_K"] - b["T_max_K"],
                         "I_rel_vs_inprocess": a["current_A_per_cm"] / b["current_A_per_cm"] - 1,
                         "transfer_conservation": a["transfer_conservation_rel_error"],
                         "energy_balance": a["energy_balance_rel_error"],
                         "coupling_iterations": a["coupling_iterations"]})
            ok &= a["transfer_conservation_rel_error"] < 1e-8 and a["energy_balance_rel_error"] < 1e-8
            if label == "same_mesh":
                ok &= abs(rows[-1]["I_rel_vs_inprocess"]) < 1e-6 and abs(rows[-1]["dT_max_K"]) / b["T_max_K"] < 1e-6
        out[label] = rows
    out["pass"] = bool(ok)
    return out


def run_all(work=None, with_precice=True) -> dict:
    work = Path(work or tempfile.mkdtemp(prefix="simws_geo_"))
    work.mkdir(parents=True, exist_ok=True)
    res = {"volumes": volumes(work), "contacts": contacts(work), "same_device": same_device(work),
           "geant4_slab": geant4_slab(), "transfer": transfer(work)}
    if with_precice:
        res["precice"] = precice(work)
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
    return res
