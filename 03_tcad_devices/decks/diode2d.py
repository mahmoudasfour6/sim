"""2-D test structure: 4H-SiC pn diode 2 um x 1 um, contacts 'anode' (top) and 'cathode' (bottom) -> diode2d.msh"""
import gmsh
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
W, H = 2.0, 1.0
s = gmsh.model.occ.addRectangle(0, 0, 0, W, H); gmsh.model.occ.synchronize()
crv = {}
for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
    if abs(ymin - H) < 1e-5 and abs(ymax - H) < 1e-5: crv["anode"] = c
    if abs(ymax) < 1e-5: crv["cathode"] = c
gmsh.model.addPhysicalGroup(2, [s], name="sic")
for k, c in crv.items(): gmsh.model.addPhysicalGroup(1, [c], name=k)
# finer mesh near the junction at y = 0.5
f = gmsh.model.mesh.field
b = f.add("Box"); f.setNumber(b, "VIn", 0.01); f.setNumber(b, "VOut", 0.05)
for k, v in (("XMin", 0), ("XMax", W), ("YMin", 0.4), ("YMax", 0.6), ("Thickness", 0.2)): f.setNumber(b, k, v)
f.setAsBackgroundMesh(b)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.model.mesh.generate(2); gmsh.write("diode2d.msh"); gmsh.finalize()
