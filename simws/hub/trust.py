"""Trust labels for stored results (architecture v1, section 6.1).

    t = assess(res, comparisons)        # pure function of the envelope and the comparison dicts, no I/O
    t = assess_stored(hub, result_id)   # reads the envelope and the comparisons recorded in the graph

Levels, weakest to strongest: sketch, unverified, verified, validated, measured. First matching rule wins.
"""
from __future__ import annotations

from typing import List, Sequence

from pydantic import BaseModel

from simws.hub.schema import ResultEnvelope

LEVELS = ("sketch", "unverified", "verified", "validated", "measured")
_RANGE_PREFIXES = ("extrapolated", "cryo_unvalidated")


class Trust(BaseModel):
    level: str
    in_range: bool
    reasons: List[str]


def _range_flags(res: ResultEnvelope) -> List[str]:
    flags = list(res.flags)
    for q in res.quantities:
        flags += q.flags
    out: List[str] = []
    for f in flags:
        if f.startswith(_RANGE_PREFIXES) and f not in out:
            out.append(f)
    return out


def assess(res: ResultEnvelope, comparisons: Sequence[dict] = ()) -> Trust:
    """Trust level of one result. `comparisons` are Comparison.model_dump() dicts; only those naming res.id count."""
    rng = _range_flags(res)
    in_range = not rng
    range_reasons = list(rng)

    if res.status != "ok":
        return Trust(level="unverified", in_range=in_range, reasons=[f"status:{res.status}"] + range_reasons)
    if "measured" in res.flags:
        return Trust(level="measured", in_range=in_range, reasons=["measured data"] + range_reasons)
    if "sketch" in res.flags:
        return Trust(level="sketch", in_range=in_range, reasons=["sketch: exploration only"] + range_reasons)

    mine = [c for c in comparisons if res.id and res.id in (c.get("a"), c.get("b"))]
    validated: List[str] = []
    verified: List[str] = []
    extra: List[str] = []
    for c in mine:
        other = c.get("b") if c.get("a") == res.id else c.get("a")
        kind = c.get("kind")
        if c.get("passed"):
            if kind == "measurement":                       # A2.2: only validation data validates
                lab = c.get("measurement_label")
                if lab == "validation":
                    validated.append(f"validated against measurement {other}")
                elif lab == "calibration":
                    extra.append(f"matches calibration data {other} (fitted to it: not validation)")
                else:
                    extra.append(f"matches unlabelled measured data {other} (not counted as validation)")
            elif kind == "engine_pair":                     # A2.3: only a different engine verifies
                if c.get("independent"):
                    verified.append(f"engine pair agreement with {other}")
                else:
                    extra.append(f"same-engine or sketch comparison with {other} (consistency only: not verification)")
        else:
            what = "measurement" if kind == "measurement" else "engine"
            tail = " (inconclusive)" if c.get("inconclusive") else ""
            extra.append(f"failed {what} comparison with {other}{tail}")
    for k, v in res.verification.items():
        if not isinstance(v, dict):
            continue
        if v.get("kind") == "code":
            if v.get("passed"):
                verified.append(f"code verification '{k}' passed")
            else:
                extra.append(f"code verification '{k}' not passed")
        elif v.get("kind") == "solution" and "numerical error quantified" not in extra:
            extra.append("numerical error quantified")

    # amendment A1.10: one failed (not merely inconclusive) measurement comparison blocks "validated", however many
    # others passed: a model that matches the fit temperatures and misses the held-out ones is not validated
    failed_meas = [c for c in mine if c.get("kind") == "measurement" and not c.get("passed") and not c.get("inconclusive")]
    if validated and failed_meas:
        extra.append(f"not validated: {len(failed_meas)} measurement comparison(s) failed, {len(validated)} passed")
        validated = []
    # amendment A13.7 (stricter only): one failed independent engine comparison blocks "verified", as A1.10 does for
    # "validated"; and a result flagged as a known failure is never above "unverified" (two codes agreeing on a model
    # known to be wrong verify the code, not the number)
    failed_pair = [c for c in mine if c.get("kind") == "engine_pair" and c.get("independent") and not c.get("passed")
                   and not c.get("inconclusive")]
    if verified and failed_pair:
        extra.append(f"not verified: {len(failed_pair)} engine comparison(s) failed, {len(verified)} passed")
        verified = []
    known = [f for f in res.flags if ":known_failure:" in f]
    if known and (validated or verified):
        extra.append("not verified: the result carries a known-failure flag (" + known[0].split(":", 2)[0] +
                     ":known_failure); agreement with another code does not make it right")
        validated, verified = [], []
    if validated:
        level, reasons = "validated", validated + verified
    elif verified:
        level, reasons = "verified", verified
    else:
        level, reasons = "unverified", ["no passed comparison or code verification"]
    return Trust(level=level, in_range=in_range, reasons=reasons + extra + range_reasons)


def assess_stored(hub, result_id: str) -> Trust:
    from simws import compare
    return assess(hub.result(result_id), compare.records_for(hub, result_id))
