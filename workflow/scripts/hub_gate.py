"""simws 0.8 upgrade gate: the hub must reproduce the 0.7 results it wraps.

Every check runs the same physics twice: once through the hub (job key, isolated adapter process, gatekeeper,
content-addressed store, knowledge graph) and once by calling the 0.7 function directly. They must agree to
round-off; any difference is a wiring error (units, conventions, parameter hand-over), not physics.

    python workflow/scripts/hub_gate.py --part devices --outdir <dir>     # tcad env
    python workflow/scripts/hub_gate.py --part dft --outdir <dir>         # materials env (Quantum ESPRESSO, PBE)
    python workflow/scripts/hub_gate.py --part pseudos --outdir <dir>     # PseudoDojo vs the bundled 0.7 set (PBE)
The 'pseudos' part is a physics comparison, not a round-off one: two different pseudopotential sets must give the
same PBE band edges within stated tolerances (masses 3 %, conduction-minimum position 1 %, gap 50 meV).
"""
import argparse
import json
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["devices", "dft", "pseudos"], required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--ranks", type=int, default=1)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

from simws.hub.jobs import Hub  # noqa: E402
from simws.hub.schema import Budget  # noqa: E402

hub = Hub(out / "store")
rows = {}


def pair(key, hub_value, direct_value, rtol=1e-9):
    rel = abs(hub_value / direct_value - 1) if direct_value else abs(hub_value)
    rows[key] = {"hub": float(hub_value), "direct": float(direct_value), "rel_diff": float(rel), "pass": bool(rel <= rtol)}
    print(f"{key:40s} hub {hub_value:+.10e}  direct {direct_value:+.10e}  rel {rel:.1e}")


def ok(r):
    if r.status != "ok":
        raise SystemExit(f"{r.job_id}: {r.status}: {r.message}")
    return r


if a.part == "devices":
    from simws.radiation.hbt_rad import StrikeSetup, strike
    from simws.radiation.tid import dose_response
    from simws.tcad.hbt import HBTCase
    from simws.tcad.mos import MOSCase
    from simws.validation.sige_checks import collector_enhancement

    # TID: t_ox 20 nm, 1 MV/cm irradiation field (V_irr = 2 V), 1 Mrad(SiO2) = 1e4 Gy(SiO2)
    r = ok(hub.submit("radiation.tid_mos", params={"t_ox": (20.0, "nm"), "V_irr": (2.0, "V"), "NA": (1e17, "cm**-3"),
                                                   "dose": ([0.0, 100.0, 300.0, 1000.0, 3000.0, 1e4], "Gy")}))
    d = dose_response(MOSCase(t_ox_nm=20.0), V_irr=2.0, doses_krad=[0, 10, 30, 100, 300, 1000])
    pair("tid.tox20.dVmg_1Mrad_V", r.q("tid.dV_mg").value[-1], d["dV_mg"][-1])
    pair("tid.tox20.N_it_1Mrad_cm2", r.q("tid.N_it").to("1/cm**2")[-1], d["N_it"][-1])
    # SiGe collector enhancement at VBE = 0.6 V: two hub jobs (Si and SiGe), ratio of intrinsic-base Jn
    jn = {ge: ok(hub.submit("devsim.hbt_gummel", params={"case": {"ge_peak": ge}, "VBE": ([0.6], "V"),
                                                          "intrinsic_jn": True})).q("hbt.Jn_intrinsic").value[0]
          for ge in (0.0, 0.15)}
    ce = collector_enhancement(VBE=(0.6,))["cases"]["VBE_0.6"]
    pair("sige.IC_ratio_0.6V", jn[0.15] / jn[0.0], ce["Jn_ratio_devsim"])
    # HBT strike at LET 0.1 MeV cm2/mg: collected collector charge
    s = ok(hub.submit("devsim.strike_hbt", params={"LET": (0.1, "MeV*cm**2/mg")}))
    sd = strike(HBTCase(), StrikeSetup(let=0.1))
    pair("sige.strike_QC_let0.1_fC", s.q("strike.Q_collector").value * 1e9, sd["Q_collector_fC"])
    # code verification of the production Poisson equation
    m = ok(hub.submit("devsim.mms_poisson", params={"levels": 4}))
    rows["mms.poisson_order"] = {"hub": m.q("mms.observed_order").value, "pass": m.verification["code"]["passed"]}
    print(f"{'mms.poisson_order':40s} {m.q('mms.observed_order').value:.4f}")
elif a.part == "pseudos":
    from simws.studies.dft_params import Config
    if Config.from_workspace(pseudo_set="pseudodojo").pseudo_set != "pseudodojo":
        raise SystemExit("PseudoDojo set not installed (tools/fetch_pseudodojo.sh)")
    bud = Budget(wall_s=8 * 3600, ranks=a.ranks)
    rs = {ps: ok(hub.submit("dft.si_bandedge", params={"functional": "PBE", "pseudo_set": ps},
                            member=f"functional=PBE|pseudos={ps}", budget=bud)) for ps in ("bundled", "pseudodojo")}
    tol = {"dft.m_l": ("rel", 0.03), "dft.m_t": ("rel", 0.03), "dft.k0_frac_GX": ("rel", 0.01), "dft.Eg_dft": ("abs", 0.05)}
    for name, (kind, t) in tol.items():
        b, p = float(rs["bundled"].q(name).value), float(rs["pseudodojo"].q(name).value)
        d = abs(p / b - 1) if kind == "rel" else abs(p - b)
        rows[f"pseudos.{name}"] = {"bundled": b, "pseudodojo": p, kind + "_diff": d, "tol": t, "pass": bool(d <= t)}
        print(f"{name:20s} bundled {b:.5f}  pseudodojo {p:.5f}  {kind} diff {d:.4f} (tol {t})")
    rows["pseudodojo_flags"] = {"flags": rs["pseudodojo"].flags}
else:
    import dataclasses
    import shutil
    import tempfile

    from simws.dft import bandedge
    from simws.studies.dft_params import A_EXP_SI, Config

    r = ok(hub.submit("dft.si_bandedge", params={"functional": "PBE"}, member="functional=PBE",
                      budget=Budget(wall_s=6 * 3600, ranks=a.ranks)))
    cfg = Config.from_workspace(ranks=a.ranks)
    sc, fo, fr, fo_fr = cfg.pseudos_for("PBE")
    st = bandedge.Setup(pseudo=sc, soc_pseudo=fr, a_angstrom=A_EXP_SI, functional=fo, ecutwfc=cfg.ecutwfc,
                        ranks=cfg.ranks, pseudo_dir=cfg.pseudo_dir)
    wd = Path(tempfile.mkdtemp(prefix="gate_qe_"))
    try:
        b = bandedge.compute(st if not (fr and fo_fr != fo) else dataclasses.replace(st, soc_pseudo=None), wd, dos=False)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    pair("dft.PBE.m_l", r.q("dft.m_l").value, b["electrons"]["m_l"], rtol=1e-6)
    pair("dft.PBE.m_t", r.q("dft.m_t").value, b["electrons"]["m_t"], rtol=1e-6)
    rows["hub_values"] = {q.name: q.value for q in r.quantities}

ev = hub.graph.events()
rows["graph"] = {"events": len(ev), "committed": sum(e["type"] == "committed" for e in ev),
                 "nodes": hub.graph.to_networkx().number_of_nodes()}
rows["pass"] = all(v.get("pass", True) for v in rows.values() if isinstance(v, dict))
(out / f"hub_gate_{a.part}.json").write_text(json.dumps(rows, indent=1, default=float))
print("hub gate", a.part, "pass:", rows["pass"])
if not rows["pass"]:
    raise SystemExit(1)
