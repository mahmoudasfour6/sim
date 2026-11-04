"""Find which part of the cold-diode decks makes Charon abort (it exits with MPI_ABORT 14 and no message).

    conda activate tcad; python tools/charon_bisect.py          (from ~/sim_workspace)

Runs Charon's own capability deck first (must pass), then our deck with one feature removed or changed at a time,
each with mpirun -np 1 ... --current exactly as the Charon interpreter launches it. Prints a pass/fail table and
copies every log to sim_backup/charon_dbg/bisect.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WS = Path.home() / "sim_workspace"
OUT = WS / "runs" / "charon_bisect"
ROOT = Path(os.environ.get("CHARON_ROOT", Path.home() / "tcad"))


def find(name, extra=""):
    p = subprocess.run(f"find -L {ROOT} -name {name} -type f {extra} 2>/dev/null | head -1", shell=True,
                       capture_output=True, text=True).stdout.strip()
    return p


EXE = os.environ.get("CHARON_EXE") or find("charon_mp.exe")
INTERP = os.environ.get("CHARON_INTERP") or find("charonInterpreter.py", "-not -path '*parseGenerator*'")
MESH = find("pndiode.exo", "-path '*capability*'")
CAP = find("pndiode.nlp.inp", "-path '*capability*'")
if not (EXE and INTERP and MESH and CAP):
    sys.exit(f"missing: exe={EXE!r} interp={INTERP!r} mesh={MESH!r} capability deck={CAP!r}")
MPIRUN = shutil.which("mpirun") or ""

sys.path.insert(0, str(WS))
from simws.validation import charon_xcheck as X  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(MESH, OUT / "pndiode.exo")
os.chdir(OUT)
X.write_decks(OUT, 77.0)
X.write_decks(OUT, 300.0)
nlp77, dd77 = (OUT / "xc_T77_nlp.inp").read_text(), (OUT / "xc_T77_dd_0000mV.inp").read_text()
nlp300 = (OUT / "xc_T300_nlp.inp").read_text()


def drop(text, *patterns):
    """Remove lines matching any regex; a pattern 'start X' removes the whole X block."""
    out, skip = [], None
    for line in text.splitlines():
        s = line.strip().lower()
        if skip:
            if s.startswith("end " + skip):
                skip = None
            continue
        hit = False
        for p in patterns:
            if p.startswith("block:") and s.startswith("start " + p[6:].lower()):
                skip, hit = p[6:].lower(), True
            elif not p.startswith("block:") and re.search(p, s):
                hit = True
        if not hit:
            out.append(line)
    return "\n".join(out) + "\n"


def from_mesh(dd):
    """dd deck that starts from the bare mesh (no nlp stage): equilibrium potential and densities."""
    dd = re.sub(r"Import State File \S+ at index 1", "Import State File pndiode.exo", dd)
    dd = dd.replace("initial conditions for ELECTRIC_POTENTIAL in silicon is Exodus File",
                    "initial conditions for ELECTRIC_POTENTIAL in silicon is Equilibrium Potential")
    return dd


II = ("incomplete ionization is on", "block:incomplete ionization donor", "block:incomplete ionization acceptor")
cap = Path(CAP).read_text()
variants = [
    ("capability pndiode.nlp (Charon's own, 300 K)", cap),
    ("our nlp 300 K", nlp300),
    ("our nlp 77 K", nlp77),
    ("nlp 77 K, no incomplete ionization", drop(nlp77, *II)),
    ("nlp 77 K, no II, no Fermi-Dirac", drop(nlp77, *II, "use fermi dirac")),
    ("nlp 77 K, no II, no DOS/gap blocks", drop(nlp77, *II, "block:effective dos", "block:band gap")),
    ("nlp 77 K, no II, no lifetime/mobility/recomb lines",
     drop(nlp77, *II, "block:carrier lifetime", "fixed mobility", "recombination is off")),
    ("nlp 300 K, no II", drop(nlp300, *II)),
    ("dd 0 V from mesh, 77 K (all features)", from_mesh(dd77)),
    ("dd 0 V from mesh, 77 K, no Fermi-Dirac", drop(from_mesh(dd77), "use fermi dirac")),
    ("dd 0 V from mesh, 77 K, no II", drop(from_mesh(dd77), *II)),
    ("dd 0 V from mesh, 77 K, no DOS/gap blocks", drop(from_mesh(dd77), "block:effective dos", "block:band gap")),
]

rows = []
for k, (label, text) in enumerate(variants):
    name = f"v{k:02d}"
    text = re.sub(r"Output State File \S+", f"Output State File {name}.exo", text)
    if k == 0:
        text = text.replace("pndiode.exo\n", "pndiode.exo\n", 1)
    (OUT / f"{name}.inp").write_text(text)
    r = subprocess.run([sys.executable, INTERP, "-i", f"{name}.inp"], capture_output=True, text=True)
    if r.returncode or not (OUT / f"{name}.inp.xml").exists():
        rows.append((name, label, "INTERPRETER FAIL", (r.stdout + r.stderr)[-300:]))
        continue
    cmd = ([MPIRUN, "-np", "1"] if MPIRUN else []) + [EXE, f"--i={name}.inp.xml", "--current"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    log = p.stdout + "\n--- stderr ---\n" + p.stderr
    (OUT / f"{name}.log").write_text(log)
    msg = [ln for ln in log.splitlines() if re.search(r"error|exception|throw|what\(\)|invalid|not found|"
                                                      r"must|cannot|unknown|failed", ln, re.I)
           and "[unused]" not in ln][:4]
    rows.append((name, label, "ok" if p.returncode == 0 else f"FAIL rc={p.returncode}", " | ".join(msg)[:300]))
    print(f"{name}  {rows[-1][2]:14s} {label}", flush=True)

print("\nSUMMARY")
for name, label, st, msg in rows:
    print(f"{name}  {st:16s} {label}" + (f"\n        {msg}" if msg and st != "ok" else ""))
win = subprocess.run("cmd.exe /c 'echo %USERNAME%'", shell=True, capture_output=True, text=True).stdout.strip()
dbg = Path(f"/mnt/c/Users/{win}/sim_backup/charon_dbg/bisect")
try:
    dbg.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("v*.log"):
        shutil.copy(f, dbg)
    for f in OUT.glob("v*.inp"):
        shutil.copy(f, dbg)
    (dbg / "summary.txt").write_text("\n".join(f"{a}\t{c}\t{b}\t{d}" for a, b, c, d in rows))
    print(f"\nlogs copied to {dbg}")
except Exception as e:  # noqa: BLE001
    print("could not copy logs:", e)
