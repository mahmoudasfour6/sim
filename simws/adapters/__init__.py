"""Built-in adapters. Each entry maps an adapter name to "module:Class"; engine imports live inside run(), so the
hub can read every manifest from any env."""
from simws.hub.adapter import register

register("materials.intrinsic", "simws.adapters.materials_intrinsic:Intrinsic")
register("devsim.hbt_gummel", "simws.adapters.tcad:HbtGummel")
register("devsim.mos_cv", "simws.adapters.tcad:MosCv")
register("devsim.hbt_ac", "simws.adapters.tcad:HbtAc")
register("devsim.strike_hbt", "simws.adapters.tcad:StrikeHbt")
register("devsim.mms_poisson", "simws.adapters.tcad:MmsPoisson")
register("radiation.tid_mos", "simws.adapters.radiation:TidMos")
register("radiation.ser_rate", "simws.adapters.radiation:SerRate")
register("dft.si_bandedge", "simws.adapters.dft:SiBandedge")
register("device.devsim", "simws.adapters.device:DeviceDevsim")
register("device.charon", "simws.adapters.device:DeviceCharon")
register("implant.rustbca", "simws.adapters.implant:ImplantRustbca")
register("implant.g4implant", "simws.adapters.implant:ImplantG4implant")
register("implant.channeling", "simws.adapters.implant:ImplantChanneling")
register("see.g4see", "simws.adapters.see:SeeG4see")
register("circuit.ngspice", "simws.adapters.circuit:CircuitNgspice")
register("device.devsim_eb", "simws.adapters.device:DeviceDevsimEb")
