import pytest


@pytest.fixture
def hub(tmp_path):
    from simws.hub.jobs import Hub
    return Hub(tmp_path / "store", mode="inprocess")
