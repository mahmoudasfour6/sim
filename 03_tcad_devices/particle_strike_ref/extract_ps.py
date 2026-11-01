#!/usr/bin/env python3
"""Dump what is needed to replicate Charon's particleStrike nightly test in DEVSIM:
mesh geometry, contacts, global (current) responses and the strike generation field from the gold file."""
import sys
import numpy as np

try:
    import exodus3 as exodus
except ImportError:
    import exodus

for path in sys.argv[1:]:
    print(f"\n######## {path}")
    e = exodus.exodus(path, mode="r", array_type="numpy")
    x, y, z = (np.asarray(c) for c in e.get_coords())
    print(f"dim={e.num_dimensions()} nodes={len(x)} elems={e.num_elems()} times={e.num_times()}")
    print(f"bbox x[{x.min():.6g},{x.max():.6g}] y[{y.min():.6g},{y.max():.6g}]")
    ux, uy = np.unique(np.round(x, 9)), np.unique(np.round(y, 9))
    print(f"unique x ({len(ux)}): {np.array2string(ux, precision=6, max_line_width=200, threshold=400)}")
    print(f"unique y ({len(uy)}): {np.array2string(uy, precision=6, max_line_width=200, threshold=400)}")

    blk_ids = list(e.get_elem_blk_ids())
    for b in blk_ids:
        conn, n_el, n_per = e.get_elem_connectivity(b)
        print(f"block {b} name={e.get_elem_blk_name(b)!r} type={e.elem_type(b)} n_elem={n_el} nodes/elem={n_per}")
    conn, n_el, n_per = e.get_elem_connectivity(blk_ids[0])
    conn = np.asarray(conn).reshape(n_el, n_per) - 1
    cx, cy = x[conn].mean(axis=1), y[conn].mean(axis=1)

    for sid in e.get_side_set_ids():
        name = e.get_side_set_name(sid)
        try:
            _, nodes = e.get_side_set_node_list(sid)
            nodes = np.unique(np.asarray(nodes) - 1)
            print(f"sideset {sid} {name!r}: {len(nodes)} nodes, x[{x[nodes].min():.4g},{x[nodes].max():.4g}] "
                  f"y[{y[nodes].min():.4g},{y[nodes].max():.4g}]")
        except Exception as exc:
            print(f"sideset {sid} {name!r}: {exc}")

    print("nodal vars:", e.get_node_variable_names())
    print("element vars:", e.get_element_variable_names())
    gnames = e.get_global_variable_names()
    print("global vars:", gnames)
    if e.num_times() == 0:
        e.close(); continue
    t = np.asarray(e.get_times())
    cols = {g: np.asarray(e.get_global_variable_values(g)) for g in gnames}
    print("\n# time-series (global variables)")
    print("time_s," + ",".join(gnames))
    for i, ti in enumerate(t):
        print(f"{ti:.6e}," + ",".join(f"{cols[g][i]:.8e}" for g in gnames))

    gen_name = next((v for v in e.get_element_variable_names() if "strike" in v.lower()), None)
    if gen_name:
        gmax = np.array([np.asarray(e.get_element_variable_values(blk_ids[0], gen_name, k + 1)).max()
                         for k in range(len(t))])
        print(f"\n# max cell '{gen_name}' per step")
        for ti, gm in zip(t, gmax):
            print(f"{ti:.6e},{gm:.6e}")
        k = int(gmax.argmax())
        g = np.asarray(e.get_element_variable_values(blk_ids[0], gen_name, k + 1))
        print(f"\n# spatial generation at peak step t={t[k]:.6e}: cells with G > 1e-6*max "
              f"({int((g > 1e-6 * g.max()).sum())} of {len(g)})")
        print("cx,cy,G")
        for i in np.where(g > 1e-6 * g.max())[0]:
            print(f"{cx[i]:.6f},{cy[i]:.6f},{g[i]:.6e}")
        dop = next((v for v in e.get_element_variable_names() if v.lower() == "doping"), None)
        if dop:
            d = np.asarray(e.get_element_variable_values(blk_ids[0], dop, 1))
            print(f"\n# doping range: {d.min():.4e} .. {d.max():.4e}; sign at x<0.5: {np.sign(d[cx < 0.45]).mean():+.2f}")
    e.close()
