"""Write every material parameter with unit, status and reference (for methods sections / audits)."""
import sys
from pathlib import Path

from simws.materials import available, load

out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
lines = ["# Material parameters (simws.materials)\n"]
for name in available():
    m = load(name)
    lines += [f"\n## {name}\n", "| parameter | value | unit | status | reference |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {p.value:g} | {p.unit} | {p.status} | {p.ref} |" for k, p in sorted(m.parameters.items())]
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
