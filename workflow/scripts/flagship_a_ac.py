"""Flagship A, part 2 (0.9 closure): f_T, f_max and low-frequency noise of the SiGe HBT and the Si BJT from 77 to
400 K, at fixed collector current densities taken from the DC flagship run (results/flagship_a/flagship_a.json).

    python workflow/scripts/flagship_a_ac.py --outdir results/flagship_a_ac [--workers 4]

Jobs run through the hub on the Dask runtime (simws.hub.runtime), cryogenic core on, same mesh as the DC run.
Reported per T and device: f_T and f_max at JC = 1, 10, 100 A/m (1-100 uA/um), the AC vs charge-control f_T
difference (frozen-mobility linearisation check), and S_IC / 2qIC, S_IB / 2qIB at JC = 1 A/m.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from simws.hub.jobs import Hub
from simws.hub.runtime import Runtime
from simws.units import Q_E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dc", default="results/flagship_a/flagship_a.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--JC", default="1,10,100")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    dc = json.loads(Path(a.dc).read_text())
    hub = Hub()
    JC = [float(x) for x in a.JC.split(",")]
    GE = {"si_bjt": 0.0, "sige_hbt": 0.15}

    tasks, meta = [], []
    for row in dc["rows"]:
        T = row["T_K"]
        for dev, rid in row["result_ids"].items():
            if row["status"].get(dev) != "ok":
                continue
            r = hub.store.envelope(rid)
            q = r.q("hbt.IC")
            VB, IC = np.asarray(q.axes[0].values), q.array()
            ok = IC > 0
            V = [float(np.interp(np.log(j), np.log(IC[ok]), VB[ok])) for j in JC if IC[ok].min() <= j <= IC[ok].max()]
            if not V:
                continue
            for k, v in enumerate(V):
                p = {"case": {"ge_peak": GE[dev], "cryo": True}, "T": (float(T), "K"), "VBE": (np.array([round(v, 4)]), "V"),
                     "VCE": (1.0, "V"), "noise": bool(k == 0)}
                tasks.append(("devsim.hbt_ac", {"params": p, "priority": "batch"}))
                meta.append((T, dev, JC[k]))

    print(f"{len(tasks)} AC jobs on {a.workers} workers")
    with Runtime(n_workers=a.workers) as rt:
        res = rt.jobs(tasks)

    rows = {}
    for (T, dev, j), r in zip(meta, res):
        d = rows.setdefault(f"{T}", {}).setdefault(dev, {})
        if r.status != "ok":
            d[f"JC{j:g}"] = {"status": r.status, "message": (r.message or "")[:200]}
            continue
        get = lambda n: float(r.q(n).array()[0]) if n in {x.name for x in r.quantities} else None
        e = {"status": "ok", "VBE": float(r.q("hbt.fT").axes[0].values[0]), "IC_A_per_m": get("hbt.IC"),
             "IB_A_per_m": get("hbt.IB"), "fT_GHz": get("hbt.fT") / 1e9, "fmax_GHz": (get("hbt.fmax") or 0) / 1e9,
             "tau_ps": get("hbt.tau_charge") * 1e12, "fT_ac_vs_charge": r.diagnostics.get("fT_ac_vs_charge", [None])[0],
             "flags": r.flags, "result_id": r.id}
        if get("hbt.S_IC") is not None:
            e["S_IC_over_2qIC"] = get("hbt.S_IC") / (2 * Q_E * e["IC_A_per_m"])
            e["S_IB_over_2qIB"] = get("hbt.S_IB") / (2 * Q_E * e["IB_A_per_m"])
        d[f"JC{j:g}"] = e
    (out / "flagship_a_ac.json").write_text(json.dumps({"JC_A_per_m": JC, "rows": rows}, indent=1, default=float))
    hdr = f"{'T':>5} {'device':9s} " + " ".join(f"{'fT@' + format(j, 'g'):>9} {'fmax':>7}" for j in JC) + "  S_IC/2qIC  S_IB/2qIB"
    lines = [hdr]
    for T, devs in rows.items():
        for dev, d in devs.items():
            cells = []
            for j in JC:
                e = d.get(f"JC{j:g}", {})
                cells.append(f"{e['fT_GHz']:9.3g} {e['fmax_GHz']:7.3g}" if e.get("status") == "ok" else f"{'-':>9} {'-':>7}")
            e1 = d.get(f"JC{JC[0]:g}", {})
            n1 = f"{e1.get('S_IC_over_2qIC', float('nan')):9.3g}  {e1.get('S_IB_over_2qIB', float('nan')):9.3g}"
            lines.append(f"{T:>5} {dev:9s} " + " ".join(cells) + "  " + n1)
    (out / "flagship_a_ac_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
