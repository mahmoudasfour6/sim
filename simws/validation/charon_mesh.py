"""Exodus II meshes for the Charon cross-checks, written directly with netCDF4 (classic format, the layout Cubit
writes for Charon's own pndiode.exo: combined 'coord', one QUAD4 block, two side sets with distribution factors).

    python -m simws.validation.charon_mesh OUT.exo --dx-um 0.00125      (1 x 0.5 um diode, uniform along x)

Why: on Charon's 10 nm pndiode mesh the SRH peak in the depletion region (width ~ kT/qE, a few nm) is not resolved;
DEVSIM's node-lumped recombination underestimates the SRH current there and Charon's quadrature overestimates it
(300 K, 0.3 V: -3.0 % and +3.3 % about the converged value; 150 K, 0.75 V: -12 % and +14 %). The gate is run on a
mesh that resolves it, the same x nodes in both codes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def write_diode_mesh(path, L_um=1.0, W_um=0.5, dx_um=0.00125, ny=2, block="Silicon"):
    from netCDF4 import Dataset
    nx = int(round(L_um / dx_um))
    xs = np.linspace(0.0, L_um, nx + 1)
    ys = np.linspace(0.0, W_um, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")                 # node (i, j) -> id i*(ny+1) + j + 1
    nid = lambda i, j: i * (ny + 1) + j + 1
    conn = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]      # counter-clockwise
                     for i in range(nx) for j in range(ny)], dtype=np.int32)
    eid = lambda i, j: i * ny + j + 1
    # quad side k joins local nodes k and k+1: side 4 is the x = 0 edge, side 2 the x = L edge
    anode = np.array([eid(0, j) for j in range(ny)], dtype=np.int32)
    cathode = np.array([eid(nx - 1, j) for j in range(ny)], dtype=np.int32)

    d = Dataset(str(path), "w", format="NETCDF3_CLASSIC")
    d.api_version = np.float32(4.57); d.version = np.float32(4.57)
    d.floating_point_word_size = np.int32(8); d.file_size = np.int32(0)
    d.title = f"simws charon_mesh: {L_um} x {W_um} um diode, dx {dx_um} um"
    for k, n in (("len_string", 33), ("len_line", 81), ("four", 4), ("num_dim", 2), ("num_nodes", X.size),
                 ("num_elem", len(conn)), ("num_el_blk", 1), ("num_qa_rec", 1), ("num_side_sets", 2),
                 ("num_side_ss1", ny), ("num_df_ss1", 2 * ny), ("num_side_ss2", ny), ("num_df_ss2", 2 * ny),
                 ("num_el_in_blk1", len(conn)), ("num_nod_per_el1", 4)):
        d.createDimension(k, n)
    d.createDimension("time_step", None)

    def chars(var, dims, rows):
        v = d.createVariable(var, "S1", dims)
        width = d.dimensions[dims[-1]].size
        arr = np.zeros(v.shape, dtype="S1")
        flat = arr.reshape(-1, width)
        for r, s in enumerate(rows):
            b = s.encode()[: width - 1]
            flat[r, : len(b)] = np.frombuffer(b, dtype="S1")
        v[:] = arr

    d.createVariable("time_whole", "f8", ("time_step",))
    chars("qa_records", ("num_qa_rec", "four", "len_string"), ["simws", "0.9a", time.strftime("%m/%d/%Y"),
                                                              time.strftime("%H:%M:%S")])
    chars("coor_names", ("num_dim", "len_string"), ["x", "y"])
    chars("eb_names", ("num_el_blk", "len_string"), [block])
    d.createVariable("ss_status", "i4", ("num_side_sets",))[:] = [1, 1]
    v = d.createVariable("ss_prop1", "i4", ("num_side_sets",)); v.setncattr("name", "ID"); v[:] = [1, 2]
    chars("ss_names", ("num_side_sets", "len_string"), ["anode", "cathode"])
    for k, (els, side) in enumerate(((anode, 4), (cathode, 2)), start=1):
        d.createVariable(f"elem_ss{k}", "i4", (f"num_side_ss{k}",))[:] = els
        d.createVariable(f"side_ss{k}", "i4", (f"num_side_ss{k}",))[:] = np.full(ny, side, np.int32)
        d.createVariable(f"dist_fact_ss{k}", "f8", (f"num_df_ss{k}",))[:] = np.ones(2 * ny)
    d.createVariable("elem_map", "i4", ("num_elem",))[:] = np.arange(1, len(conn) + 1, dtype=np.int32)
    d.createVariable("eb_status", "i4", ("num_el_blk",))[:] = [1]
    v = d.createVariable("eb_prop1", "i4", ("num_el_blk",)); v.setncattr("name", "ID"); v[:] = [1]
    v = d.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1")); v.setncattr("elem_type", "QUAD4")
    v[:] = conn
    d.createVariable("coord", "f8", ("num_dim", "num_nodes"))[:] = np.vstack([X.ravel(), Y.ravel()])
    d.close()
    return {"nodes": int(X.size), "elements": int(len(conn)), "nx": nx, "dx_um": dx_um}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--dx-um", type=float, default=0.00125)
    a = ap.parse_args()
    print(write_diode_mesh(a.out, dx_um=a.dx_um))
