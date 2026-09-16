"""Layer-2 boolean tests: manifold3d union / difference / intersection."""

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


_import_layer2("femguide_core.geometry.booleans")

from femguide_core.geometry.booleans import (
    NonManifoldError,
    difference,
    from_manifold,
    intersection,
    to_manifold,
    union,
)
from femguide_core.geometry.repair import is_manifold_solid, repair


def _punched(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of the mesh with one face removed (an open hole)."""
    return trimesh.Trimesh(
        vertices=mesh.vertices.copy(), faces=mesh.faces[1:].copy(), process=False
    )


def _box(extents, translate=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(translate)
    return box


def _assert_solid(mesh: trimesh.Trimesh):
    assert is_manifold_solid(mesh)
    assert mesh.is_watertight


def test_to_from_manifold_roundtrip(clean_sphere):
    solid = to_manifold(clean_sphere)
    back = from_manifold(solid)
    _assert_solid(back)
    assert back.volume == pytest.approx(clean_sphere.volume, rel=1e-9)


def test_union_two_overlapping_boxes():
    a = _box((10, 10, 10))
    b = _box((10, 10, 10), translate=(5.0, 0.0, 0.0))
    result = union([a, b])

    _assert_solid(result)
    assert result.volume < a.volume + b.volume
    assert result.volume >= max(a.volume, b.volume)
    # Analytic: 10x10x10 twice, overlapping in a 5x10x10 slab.
    assert result.volume == pytest.approx(1500.0, rel=0.01)


def test_union_overlapping_spheres():
    a = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    b = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    b.apply_translation([8.0, 0.0, 0.0])
    result = union([a, b])

    _assert_solid(result)
    assert result.volume < a.volume + b.volume
    assert result.volume >= max(a.volume, b.volume)


def test_union_accepts_single_mesh(clean_sphere):
    result = union(clean_sphere)
    _assert_solid(result)
    assert result.volume == pytest.approx(clean_sphere.volume, rel=1e-9)


def test_difference_small_box_from_big_box():
    big = _box((20, 20, 20))
    small = _box((10, 10, 10))  # fully inside: leaves an internal cavity
    result = difference(big, small)

    _assert_solid(result)
    assert result.volume == pytest.approx(20**3 - 10**3, rel=0.01)


def test_difference_accepts_sequence_of_tools():
    big = _box((20, 20, 20))
    tools = [
        _box((4, 4, 30), translate=(-5.0, 0.0, 0.0)),
        _box((4, 4, 30), translate=(5.0, 0.0, 0.0)),
    ]
    result = difference(big, tools)

    _assert_solid(result)
    # Each tool removes a 4 x 4 x 20 column from the box.
    assert result.volume == pytest.approx(20**3 - 2 * (4 * 4 * 20), rel=0.01)


def test_intersection_analytic_overlap():
    a = _box((10, 10, 10))
    b = _box((10, 10, 10), translate=(5.0, 5.0, 5.0))
    result = intersection(a, b)

    _assert_solid(result)
    assert result.volume == pytest.approx(5.0**3, rel=0.01)


def test_intersection_disjoint_is_empty():
    a = _box((10, 10, 10))
    b = _box((10, 10, 10), translate=(100.0, 0.0, 0.0))
    result = intersection(a, b)
    assert len(result.faces) == 0


def test_non_manifold_input_raises(clean_sphere):
    holed = _punched(clean_sphere)

    with pytest.raises(NonManifoldError, match="repair"):
        to_manifold(holed, "holed sphere")
    with pytest.raises(NonManifoldError):
        union([clean_sphere, holed])
    with pytest.raises(NonManifoldError):
        difference(clean_sphere, holed)
    with pytest.raises(NonManifoldError):
        intersection(holed, clean_sphere)


def test_empty_input_raises():
    with pytest.raises(NonManifoldError):
        to_manifold(trimesh.Trimesh())


def test_dirty_input_raises_without_repair(dirty_sphere):
    with pytest.raises(NonManifoldError):
        to_manifold(dirty_sphere)


def test_end_to_end_repair_then_subtract_pin(dirty_sphere, clean_sphere):
    """THE core commercial workflow: repair segmented anatomy, then subtract a
    pin cylinder from it (stand-in for base-minus-pin-cylinders)."""
    repaired, report = repair(dirty_sphere)
    assert report.is_manifold_solid

    pin = trimesh.creation.cylinder(radius=3.0, height=60.0, sections=32)
    result = difference(repaired, pin)

    _assert_solid(result)
    # The pin bores straight through the sphere: volume strictly shrinks by
    # roughly the cylinder-through-sphere volume.
    assert result.volume < repaired.volume
    expected_removed = np.pi * 3.0**2 * 2.0 * np.sqrt(20.0**2 - 3.0**2)
    assert repaired.volume - result.volume == pytest.approx(
        expected_removed, rel=0.10
    )
