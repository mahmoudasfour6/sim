"""Planner (architecture v1, section 4.3): which engines can run a Problem, and why the others cannot.
A plan is about capability, not about what is installed on this machine."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel

from simws import engines as _engines


class Plan(BaseModel):
    problem_id: str
    primary: Optional[str]                 # first supporting engine in `prefer` order, then registry (sorted) order
    checkers: List[str]                    # the other supporting engines, sorted
    refused: Dict[str, List[str]]          # engine -> reasons
    warnings: Dict[str, List[str]]         # supporting engine -> outside_tested_range strings (only non-empty)

    def summary(self) -> str:
        lines = [f"problem {self.problem_id}: " + (f"primary engine {self.primary}" if self.primary
                                                   else "no engine can run it")]
        if self.checkers:
            # A11.6: an engine on the primary's code is a different-physics check, not a second opinion
            code = _engines.code(self.primary) if self.primary in _engines.ENGINES else self.primary
            tag = lambda e: e + (" (same code: model-form check only)" if e in _engines.ENGINES  # noqa: E731
                                 and _engines.code(e) == code else "")
            lines.append("  checkers: " + ", ".join(tag(e) for e in self.checkers))
        for e, w in sorted(self.warnings.items()):
            lines.append(f"  {e} warns: " + "; ".join(w))
        for e, r in sorted(self.refused.items()):
            lines.append(f"  {e} refuses: " + "; ".join(r))
        return "\n".join(lines)


def plan(problem, prefer: Sequence[str] = ("devsim", "rustbca"), engines: Optional[Sequence[str]] = None) -> Plan:
    # A13.3: rustbca stays the implant primary (g4implant sorts first); devsim stays the device primary
    names = sorted(engines) if engines is not None else _engines.names()
    ok: List[str] = []
    refused: Dict[str, List[str]] = {}
    warnings: Dict[str, List[str]] = {}
    for n in names:
        reasons = _engines.unsupported(n, problem)
        if reasons:
            refused[n] = reasons
            continue
        ok.append(n)
        w = _engines.capability(n).outside_tested_range(problem)
        if w:
            warnings[n] = w
    order = [n for n in prefer if n in ok] + [n for n in ok if n not in prefer]
    primary = order[0] if order else None
    return Plan(problem_id=problem.problem_id(), primary=primary, checkers=sorted(n for n in ok if n != primary),
                refused=refused, warnings=warnings)
