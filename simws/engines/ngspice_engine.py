"""ngspice translator (architecture v1, contract amendment A9.4): a CircuitProblem on ngspice 47 with OSDI.

ngspice (Ngspice User's Manual, version 47) runs as an executable in batch mode. Verilog-A compact models are
compiled by OpenVAF-Reloaded (OSDI 0.3 output) and loaded with `pre_osdi`. Results come back as an ASCII raw
file. Both executables live in one conda env (tools/install_spice.sh); importing this module needs neither.

Element names are prefixed with the SPICE kind letter and an underscore (r1 -> R_r1), so the description's names
never clash with SPICE's first-letter rule. Current probes keep the SPICE sign: i(vin) is the current that
enters the source at its first node.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from simws.engines.base import CircuitCapability, CircuitResult, EngineNotAvailable, Unsupported

FAMILY = "circuit"
NAME = "ngspice"
NGSPICE_VERSION = "47"
OPENVAF_BUILD = "osdi_0.3-31-gf47d557"
RAW = "result.raw"

CAPABILITY = CircuitCapability(
    engine=NAME, env="spice", in_process=False,
    description="ngspice 47 with OSDI; Verilog-A models compiled by OpenVAF-Reloaded (OSDI 0.3)",
    element_kinds=["resistor", "capacitor", "inductor", "vsource", "isource", "diode", "bjt", "va", "subckt"],
    model_kinds=["diode", "npn", "pnp", "va"], analyses=["op", "dc", "ac", "tran"], max_elements=10000,
    tested_element_kinds=["resistor", "capacitor", "vsource", "diode", "va", "subckt"],
    tested_analyses=["op", "dc", "ac", "tran"], tested_temperature_C=(26.85, 27.0),
    notes="One analysis per problem; subcircuits only from model libraries.")

_LETTER = {"resistor": "R", "capacitor": "C", "inductor": "L", "vsource": "V", "isource": "I", "diode": "D",
           "bjt": "Q", "va": "N", "subckt": "X"}
_MODEL_TYPE = {"diode": "D", "npn": "NPN", "pnp": "PNP"}


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem)


def prefix() -> Path:
    """The conda env with ngspice and openvaf: SIMWS_SPICE_PREFIX, else <envs_root>/spice."""
    if os.environ.get("SIMWS_SPICE_PREFIX"):
        return Path(os.environ["SIMWS_SPICE_PREFIX"]).expanduser()
    from simws import config
    return config.path("envs_root") / "spice"


def _exe(name: str) -> Path:
    return prefix() / "bin" / name


def _ok(p: Path) -> bool:
    try:
        return p.is_file() and os.access(p, os.X_OK)
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    return _ok(_exe("ngspice"))


def openvaf_available() -> bool:
    return _ok(_exe("openvaf"))


def _g(x: float) -> str:
    return repr(float(x))


def spice_name(element) -> str:
    return f"{_LETTER[element.kind]}_{element.name}"


def _probe(problem, p: str) -> str:
    from simws.spec.circuit import parse_probe
    pr = parse_probe(p)
    if pr[0] == "v":
        return f"v({pr[1]})" if pr[2] is None else f"v({pr[1]},{pr[2]})"
    return f"i(V_{pr[1]})"


def _source(e) -> str:
    parts = [f"DC {_g(e.value if e.value is not None else 0.0)}"]
    if e.ac is not None:
        parts.append(f"AC {_g(e.ac)}")
    if e.pulse is not None:
        p = e.pulse
        parts.append(f"PULSE({_g(p.v1)} {_g(p.v2)} {_g(p.delay_s)} {_g(p.rise_s)} {_g(p.fall_s)} {_g(p.width_s)} "
                     f"{_g(p.period_s)})")
    if e.pwl is not None:
        parts.append("PWL(" + " ".join(f"{_g(t)} {_g(v)}" for t, v in e.pwl.points) + ")")
    return " ".join(parts)


def build_netlist(problem, osdi: Optional[Dict[str, str]] = None) -> str:
    """The ngspice netlist for `problem` (pure). osdi: Verilog-A module name -> compiled .osdi file."""
    osdi = osdi or {}
    L: List[str] = [problem.title.replace("\n", " ")]
    for lib in problem.libraries:
        path = os.path.expanduser(lib.path)
        L.append(f'.lib "{path}" {lib.section}' if lib.section else f'.include "{path}"')
    for e in problem.elements:
        line = [spice_name(e), *e.nodes]
        if e.kind in ("resistor", "capacitor", "inductor"):
            line.append(_g(e.value))
        elif e.kind in ("vsource", "isource"):
            line.append(_source(e))
        else:
            line.append(e.model)
        line += [f"{k}={_g(v)}" for k, v in e.params.items()]
        L.append(" ".join(line))
    for m in problem.models:
        kv = " ".join(f"{k}={_g(v)}" for k, v in m.params.items())
        if m.kind == "va":
            # A9.10: no parentheses for an OSDI model; ngspice 47 drops the first parameter of "module(k=v ...)"
            L.append(f".model {m.name} {m.va.module} {kv}".rstrip())
        else:
            L.append(f".model {m.name} {_MODEL_TYPE[m.kind]}({kv})")
    n = problem.numerics
    L.append(f".options reltol={_g(n.reltol)} abstol={_g(n.abstol_A)} vntol={_g(n.vntol_V)} gmin={_g(n.gmin_S)} "
             f"temp={_g(problem.temperature_C)} tnom={_g(problem.tnom_C)}")
    L.append(".control")
    for m in problem.models:
        if m.kind == "va":
            L.append(f"pre_osdi {osdi.get(m.va.module, m.va.module + '.osdi')}")
    L.append("set filetype=ascii")
    a = problem.analysis
    if a.kind == "op":
        L.append("op")
    elif a.kind == "dc":
        src = next(e for e in problem.elements if e.name.lower() == a.source.lower())
        L.append(f"dc {spice_name(src)} {_g(a.start)} {_g(a.stop)} {_g(a.step)}")
    elif a.kind == "ac":
        L.append(f"ac dec {a.points_per_decade} {_g(a.f_start_Hz)} {_g(a.f_stop_Hz)}")
    else:
        L.append(f"tran {_g(a.step_s)} {_g(a.stop_s)} 0 {_g(a.max_step_s if a.max_step_s is not None else a.step_s)}")
    L.append(f"write {RAW} " + " ".join(_probe(problem, p) for p in problem.probes))
    L += [".endc", ".end"]
    return "\n".join(L) + "\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_root() -> Path:
    return Path(os.environ.get("SIMWS_CACHE", "~/.cache/simws")).expanduser() / "osdi"


def compile_va(va, cache_dir: Optional[Path] = None) -> Path:
    """Compile a VerilogA description with OpenVAF into <cache_dir or the cache root>/<key>/<module>.osdi."""
    return _compile_va(va, cache_dir)[0]


def _compile_va(va, cache_dir: Optional[Path] = None):
    """(path, cached): cached is True when the file was already there."""
    if not openvaf_available():
        raise EngineNotAvailable(f"no OpenVAF at {_exe('openvaf')}; install it with tools/install_spice.sh")
    exe = _exe("openvaf")
    key = hashlib.sha256((_sha256(exe) + "\n" + va.source).encode()).hexdigest()
    folder = (Path(cache_dir) if cache_dir else cache_root()) / key
    out = folder / f"{va.module}.osdi"
    if out.exists():
        return out, True
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / f"{va.module}.va"
    src.write_text(va.source)
    r = subprocess.run([str(exe), src.name, "-o", out.name], cwd=folder, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"OpenVAF failed on module {va.module}:\n{(r.stdout + r.stderr)[-2000:]}")
    return out, False


def parse_raw(path) -> dict:
    """An ngspice ASCII raw file: {"plotname", "complex", "names", "types", "values" (n_points x n_vars,
    complex numbers when the plot is complex)}."""
    lines = Path(path).read_text(errors="replace").splitlines()
    head: Dict[str, str] = {}
    i = 0
    while i < len(lines) and not lines[i].startswith("Variables:"):
        if ":" in lines[i]:
            k, _, v = lines[i].partition(":")
            head[k.strip()] = v.strip()
        i += 1
    nvar, npts = int(head["No. Variables"]), int(head["No. Points"])
    cplx = "complex" in head.get("Flags", "")
    names, types = [], []
    for ln in lines[i + 1:i + 1 + nvar]:
        f = ln.split()
        names.append(f[1].lower())
        types.append(f[2] if len(f) > 2 else "")
    j = i + 1 + nvar
    if not lines[j].startswith("Values:"):
        raise ValueError(f"{path}: no 'Values:' section")
    tokens: List[str] = []
    for ln in lines[j + 1:]:
        f = ln.split()
        if not f:
            continue
        tokens.append(f[-1])               # the first line of a point also holds the point index
    if len(tokens) != nvar * npts:
        raise ValueError(f"{path}: {len(tokens)} values, expected {nvar} x {npts}")

    def num(t: str):
        if cplx:
            re_, _, im = t.partition(",")
            return complex(float(re_), float(im))
        return float(t)

    vals = [[num(tokens[p * nvar + v]) for v in range(nvar)] for p in range(npts)]
    return {"plotname": head.get("Plotname", ""), "complex": cplx, "names": names, "types": types, "values": vals}


def _library_flags(problem) -> List[str]:
    out = []
    for lib in problem.libraries:
        path = Path(os.path.expanduser(lib.path))
        digest = _sha256(path)[:12] if path.is_file() else "missing"
        out.append(f"circuit:external_library: {path} sha256 {digest}")
    return out


def run(problem, workdir) -> CircuitResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    if not available():
        raise EngineNotAvailable(f"no ngspice at {_exe('ngspice')}; install it with tools/install_spice.sh "
                                 f"(or point SIMWS_SPICE_PREFIX at the conda env that has it)")
    osdi, cached = {}, []
    for m in problem.models:
        if m.kind == "va":
            path, c = _compile_va(m.va)
            osdi[m.va.module] = str(path)
            cached.append(c)
    pid = problem.problem_id()
    folder = Path(workdir) / f"ngspice_{pid[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "circuit.cir").write_text(build_netlist(problem, osdi))
    log = folder / "ngspice.log"
    with open(log, "w") as fh:
        rc = subprocess.run([str(_exe("ngspice")), "-b", "circuit.cir"], cwd=folder, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    text = log.read_text(errors="replace")
    raw = folder / RAW
    if rc != 0 or not raw.exists() or "simulation interrupted" in text.lower():
        tail = "\n".join(text.splitlines()[-20:])
        raise RuntimeError(f"ngspice exited with code {rc}" + ("" if raw.exists() else ", no raw file written")
                           + f"; last lines of {log}:\n{tail}")
    data = parse_raw(raw)
    a = problem.analysis
    names = data["names"]
    vals = data["values"]
    # for dc, ac and tran the first vector of the raw file is the sweep axis
    if a.kind == "op":
        axis_name, axis_unit, axis = "", "", []
    else:
        axis = [float(v[0].real) if data["complex"] else float(v[0]) for v in vals]
        if a.kind == "dc":
            src = next(e for e in problem.elements if e.name.lower() == a.source.lower())
            axis_name, axis_unit = src.name.lower(), ("V" if src.kind == "vsource" else "A")
        elif a.kind == "ac":
            axis_name, axis_unit = "f", "Hz"
        else:
            axis_name, axis_unit = "t", "s"
    wanted = [_probe(problem, p).lower() for p in problem.probes]
    signals, imag, units = {}, {}, {}
    for p, w in zip(problem.probes, wanted):
        # ngspice writes v(a,b) as v(a,b) or as v(a)-v(b) depending on version; accept both
        alt = {w, w.replace(",", ")-v(")}
        try:
            k = next(n for n, name in enumerate(names) if name in alt)
        except StopIteration:
            raise RuntimeError(f"ngspice wrote no vector for probe {p} (vectors: {names})") from None
        col = [row[k] for row in vals]
        if data["complex"]:
            signals[p] = [float(c.real) for c in col]
            imag[p] = [float(c.imag) for c in col]
        else:
            signals[p] = [float(c) for c in col]
        units[p] = "A" if w.startswith("i(") else "V"
    return CircuitResult(engine=NAME, problem_id=pid, analysis=a.kind, axis_name=axis_name, axis_unit=axis_unit,
                         axis=axis, signals=signals, signals_imag=imag, units=units, flags=_library_flags(problem),
                         diagnostics={"ngspice": NGSPICE_VERSION, "openvaf": OPENVAF_BUILD if osdi else "",
                                      "osdi_cached": bool(cached) and all(cached), "n_points": len(vals)})
