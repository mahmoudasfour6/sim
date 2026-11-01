"""Two-way electro-thermal resistor sweep (DEVSIM + scikit-fem in one process)."""
import argparse
from pathlib import Path

from simws.coupling import electrothermal as et
from simws.io import fields
from simws.materials import load
from simws.provenance import stamp, write_json

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True)
ap.add_argument("--biases", default="0.5,1,2,3,4,5")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
case = et.ResistorCase(biases=[float(v) for v in a.biases.split(",")])
mat = load("silicon")
res = et.run(case, mat)
nodes = res.pop("nodes")
fields.write(out / "temperature_final_bias", fields.FieldSet(
    points=__import__("numpy").column_stack([nodes["x_cm"], nodes["y_cm"]]), cells=nodes["triangles"], length_unit="cm",
    point_data={"T": nodes["T_K"]}, units={"T": "K"}, meta={"bias_V": case.biases[-1]}))
prov = stamp(parameters={"case": case.__dict__, "provisional_material_params": mat.provisional()})
write_json(out / "summary.json", res, prov)
for p in res["points"]:
    print(f"V={p['bias_V']:.2f} V  I={p['current_A_per_cm']:.4f} A/cm  T_max={p['T_max_K']:.1f} K  "
          f"its={p['iterations']}  energy balance {p['energy_balance_rel_error']:.1e}")
