"""Reference values for the container check: the same numbers computed inside and outside the image must agree.
    python tools/reference_values.py out.json        (tcad env, PYTHONPATH=repo)"""
import json
import sys

import numpy as np

out = {}
from simws.materials import load
from simws.materials.semiconductor import intrinsic_density
out["n_i_si_300"] = float(intrinsic_density(load("silicon"), 300.0))
from simws.tcad import schottky_transport as S
out["J_schottky_ni"] = S.current(S.Contact(phi_B=1.75, N_D=1.97e19), 0.3)
from simws.validation import small_signal_checks as SS
out["C_dep_rel"] = SS.depletion_capacitance()["rows"][0]["C_F_cm2"]
from simws.validation import contact_checks as CC
out["te_closed_form_J"] = CC._diode(mu=1e8, A_p=0.0, Vs=(0.2,))["Jn"][0]
from simws.coupling import electrothermal as et
p = et.run(et.ResistorCase(biases=[2.0]))["points"][0]
out["electrothermal_Tmax_2V"] = p["T_max_K"]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
