"""Every registered adapter's manifest is wired consistently (no engine needed)."""
import pytest

pytest.importorskip("pydantic")

from simws import conventions as cv  # noqa: E402
from simws.hub import adapter as A  # noqa: E402

ENVS = {"workflow", "tcad", "materials", "process_rad", "quantum_iii_v"}


def _envs():
    from simws.config import load
    return ENVS | {(load().get("dft") or {}).get("env", "materials")}


@pytest.mark.parametrize("name", A.names())
def test_manifest(name):
    m = A.manifest(name)
    assert m.name == name and m.env in _envs()
    outs = {o.name for o in m.outputs}
    assert outs, "an adapter must declare outputs"
    for e in m.physics_edges:
        assert e.output in outs, f"physics edge from undeclared output {e.output}"
    for axis, val in m.native_conventions.items():
        assert axis in cv.CANONICAL
        if axis in cv.ALTERNATIVES:
            assert val in cv.ALTERNATIVES[axis]
    for k, (lo, hi) in m.validity.items():
        assert lo < hi
    # outputs follow the '<domain>.<name>' naming used as graph quantity names
    assert all("." in o for o in outs)
