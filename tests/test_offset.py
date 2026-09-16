"""Layer-2 surface offset tests."""

import importlib
import sys
import types

import numpy as np
import pytest
import trimesh


def _import_layer2(name):
    """Import a layer-2 module, stubbing still-missing sibling modules.

    The geometry package __init__ imports layer-1/3 siblings that are being
    written concurrently with this suite; any that do not exist yet get an
    empty stand-in so layer 2 imports on its own. No-op once they all exist.
    """
    for _ in range(16):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if not missing.startswith("femguide_core."):
                raise
            stub = types.ModuleType(missing)
            stub.__getattr__ = lambda attr: None
            sys.modules[missing] = stub
    raise ImportError(name)


_import_layer2("femguide_core.geometry.offset")

from femguide_core.geometry.booleans import NonManifoldError
from femguide_core.geometry.offset import offset_solid
from femguide_core.geometry.repair import is_manifold_solid


def _sphere_volume(radius: float) -> float:
    return 4.0 / 3.0 * np.pi * radius**3


def test_offset_sphere_outward(clean_sphere):
    grown = offset_solid(clean_sphere, 2.0)

    assert is_manifold_solid(grown)
    assert grown.is_watertight
    # r=20 sphere grown by 2 -> analytic volume of an r=22 sphere, within a
    # few percent (icosphere faceting keeps the mesh slightly inside).
    assert grown.volume == pytest.approx(_sphere_volume(22.0), rel=0.03)
    assert grown.volume > clean_sphere.volume


def test_offset_sphere_inward(clean_sphere):
    shrunk = offset_solid(clean_sphere, -2.0)

    assert is_manifold_solid(shrunk)
    assert shrunk.is_watertight
    assert shrunk.volume < clean_sphere.volume
    assert shrunk.volume == pytest.approx(_sphere_volume(18.0), rel=0.03)


def test_offset_zero_is_identity(clean_sphere):
    same = offset_solid(clean_sphere, 0.0)
    assert is_manifold_solid(same)
    assert same.volume == pytest.approx(clean_sphere.volume, rel=1e-9)


def test_offset_open_mesh_raises(clean_sphere):
    holed = trimesh.Trimesh(
        vertices=clean_sphere.vertices.copy(),
        faces=clean_sphere.faces[1:].copy(),
        process=False,
    )
    with pytest.raises(NonManifoldError):
        offset_solid(holed, 1.0)


def test_offset_femur_like(clean_femur_like):
    grown = offset_solid(clean_femur_like, 1.0)
    assert is_manifold_solid(grown)
    assert grown.volume > clean_femur_like.volume
