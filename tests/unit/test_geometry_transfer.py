"""0.9b shared geometry (gmsh -> DEVSIM / scikit-fem / Geant4) and conservative field transfer. The preCICE run is in
the 0.9b gate (it needs libprecice and two processes)."""
import numpy as np
import pytest

from simws.validation import geometry_checks as K


def test_structured_volumes_tile_regions(tmp_path):
    r = K.volumes(tmp_path)
    assert r["pass"], r


def test_contact_lengths(tmp_path):
    r = K.contacts(tmp_path)
    assert r["pass"], r


def test_gmsh_device_matches_native_mesh(tmp_path):
    r = K.same_device(tmp_path, Vs=(0.3, 0.5))
    assert r["pass"], r


def test_geant4_stack():
    assert K.geant4_slab()["pass"]


def test_transfers_conserve(tmp_path):
    r = K.transfer(tmp_path)
    assert r["pass"], r


def test_remap_conserves_with_unrelated_meshes():
    from skfem import MeshTri
    from simws import transfer as T
    ms = MeshTri.init_tensor(np.linspace(0, 1, 9), np.linspace(0, 1, 7))
    mt = MeshTri.init_tensor(np.linspace(0, 1, 13) ** 1.5, np.linspace(0, 1, 11) ** 0.8)
    v = np.random.default_rng(1).random(ms.t.shape[1])
    vt, rec = T.remap_cells(T.tri_polys(ms.p.T, ms.t.T), v, T.tri_polys(mt.p.T, mt.t.T))
    assert rec.conservation_rel_error < 1e-12 and rec.lost_fraction < 1e-12
    assert rec.roundtrip_l2_rel is not None and rec.roundtrip_l2_rel < 1


def test_precice_config(tmp_path):
    from simws.coupling import precice_bridge as B
    p = B.write_config(tmp_path, n_windows=2, tol_K=1e-6, max_iter=50)
    s = p.read_text()
    assert 'constraint="conservative"' in s and 'constraint="consistent"' in s and str(tmp_path / "exchange") in s
