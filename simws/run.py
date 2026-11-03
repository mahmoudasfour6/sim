"""Isolated adapter runner:  python -m simws.run <adapter name> <staging dir>

Reads <staging>/job.json, runs the adapter lifecycle, writes <staging>/result.json and exits 0 whenever an envelope
was written (the status inside says how it went). Exit 2 means the runner itself could not start.
Wall-time budget is enforced here with SIGALRM (timeout status) and again by the submitter with a hard kill.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def _alarm(signum, frame):
    raise TimeoutError("wall-time budget exceeded")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    name, staging = argv[0], Path(argv[1])
    from simws.hub import adapter as A
    from simws.hub.schema import JobEnvelope, ResultEnvelope

    job = JobEnvelope.model_validate_json((staging / "job.json").read_text())
    os.environ.setdefault("OMP_NUM_THREADS", str(job.budget.threads))
    try:
        ad = A.get(name)
    except Exception as exc:  # noqa: BLE001
        res = ResultEnvelope(job_id=job.id, status="engine_error", message=f"adapter load failed: {exc}")
    else:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(max(1, int(job.budget.wall_s)))
        try:
            res = ad.execute(job, staging)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
    import platform
    res.provenance.setdefault("python", sys.executable)
    res.provenance.setdefault("host", platform.node())
    res.provenance.setdefault("pid", os.getpid())
    (staging / "result.json").write_text(res.model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
