"""Tests for geometry layer 1 primitives.

Imports the module directly (not via the femguide_core package root, which
pulls in the other geometry layers).
"""

import math

import numpy as np
import pytest

try:
    from femguide_core.geometry.primitives import (
        safe_normalize,
        polyline_from_angles,
        polyline_plane_normal,
        segment_prism,
        polyline_prism_solids,
        offset_prism,
        oriented_cylinder,
        paired_cylinders,
    )
except ImportError:
    # The geometry package __init__ imports the other layers, which may not
    # exist yet during parallel development. Load primitives.py directly.
    import importlib.util
    from pathlib import Path

    _path = (
        Path(__file__).resolve().parents[1]
        / "core" / "femguide_core" / "geometry" / "primitives.py"
    )
    _spec = importlib.util.spec_from_file_location("_fg_primitives", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    safe_normalize = _mod.safe_normalize
    polyline_from_angles = _mod.polyline_from_angles
    polyline_plane_normal = _mod.polyline_plane_normal
    segment_prism = _mod.segment_prism
    polyline_prism_solids = _mod.polyline_prism_solids
    offset_prism = _mod.offset_prism
    oriented_cylinder = _mod.oriented_cylinder
    paired_cylinders = _mod.paired_cylinders


def _assert_solid(mesh):
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


# ---------------- safe_normalize ----------------

def test_safe_normalize_unit_vector():
    v = safe_normalize([3.0, 0.0, 4.0])
    assert np.allclose(v, [0.6, 0.0, 0.8])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_safe_normalize_zero_vector_fallback():
    assert np.allclose(safe_normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(safe_normalize([1e-12, 0.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(
        safe_normalize([0.0, 0.0, 0.0], fallback=(1.0, 0.0, 0.0)),
        [1.0, 0.0, 0.0],
    )


# ---------------- polyline_from_angles ----------------

def test_polyline_single_segment_90_degrees():
    pts = polyline_from_angles([90.0], [10.0])
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])
    assert np.allclose(pts[1], [0.0, 10.0, 0.0], atol=1e-12)


def test_polyline_two_segments_cumulative_angles():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 10.0])
    assert pts.shape == (3, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])
    assert np.allclose(pts[1], [10.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(pts[2], [10.0, 10.0, 0.0], atol=1e-12)


def test_polyline_45_degrees():
    pts = polyline_from_angles([45.0], [math.sqrt(2.0)])
    assert np.allclose(pts[1], [1.0, 1.0, 0.0])


def test_polyline_lies_in_z0_plane():
    pts = polyline_from_angles([8, 40, 45, 45, 42], [50, 20.5, 25, 11, 30])
    assert pts.shape == (6, 3)
    assert np.allclose(pts[:, 2], 0.0)
    # Segment lengths are preserved.
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(seg_lengths, [50, 20.5, 25, 11, 30])


def test_polyline_length_mismatch_raises():
    with pytest.raises(ValueError):
        polyline_from_angles([10.0, 20.0], [5.0])


def test_polyline_empty_raises():
    with pytest.raises(ValueError):
        polyline_from_angles([], [])


# ---------------- polyline_plane_normal ----------------

def test_plane_normal_of_planar_polyline_is_z():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 10.0])
    n = polyline_plane_normal(pts)
    assert np.allclose(np.abs(n), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_plane_normal_collinear_fallback():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(polyline_plane_normal(pts), [0.0, 0.0, 1.0])


def test_plane_normal_fewer_than_three_points_fallback():
    assert np.allclose(
        polyline_plane_normal([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        [0.0, 0.0, 1.0],
    )


def test_plane_normal_tilted_plane():
    # Points in the x-z plane -> normal along +-y.
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    n = polyline_plane_normal(pts)
    assert np.allclose(np.abs(n), [0.0, 1.0, 0.0], atol=1e-12)


# ---------------- segment_prism ----------------

def test_segment_prism_watertight_and_volume():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([10.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    thickness, extrusion = 2.0, 5.0
    mesh = segment_prism(p0, p1, normal, thickness, extrusion)
    _assert_solid(mesh)
    assert np.isclose(mesh.volume, thickness * extrusion * 10.0)


def test_segment_prism_extensions_add_length():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([10.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    mesh = segment_prism(p0, p1, normal, 2.0, 5.0, start_ext=3.0, end_ext=4.0)
    _assert_solid(mesh)
    assert np.isclose(mesh.volume, 2.0 * 5.0 * (10.0 + 3.0 + 4.0))
    # The extended prism spans [-3, 14] along x.
    assert np.isclose(mesh.bounds[0][0], -3.0)
    assert np.isclose(mesh.bounds[1][0], 14.0)


def test_segment_prism_centered_on_plane_along_normal():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([10.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    extrusion = 6.0
    mesh = segment_prism(p0, p1, normal, 1.5, extrusion)
    assert np.isclose(mesh.bounds[0][2], -extrusion / 2.0)
    assert np.isclose(mesh.bounds[1][2], extrusion / 2.0)


def test_segment_prism_thickness_sign_matches_prototype():
    # Prototype: thickVec = -normalize(cross(normal, segVec)) * thickness.
    # For normal=+z and segVec=+x, cross = +y, so the quad extends toward -y.
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([10.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    mesh = segment_prism(p0, p1, normal, 2.0, 4.0)
    assert np.isclose(mesh.bounds[0][1], -2.0)
    assert np.isclose(mesh.bounds[1][1], 0.0)


# ---------------- polyline_prism_solids ----------------

def test_polyline_prism_solids_count_and_watertight():
    pts = polyline_from_angles([0.0, 90.0, 45.0], [10.0, 8.0, 6.0])
    normal = polyline_plane_normal(pts)
    solids = polyline_prism_solids(pts, normal, 1.0, 5.0)
    assert len(solids) == 3
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    for solid, seg_len in zip(solids, seg_lengths):
        _assert_solid(solid)
        assert np.isclose(solid.volume, 1.0 * 5.0 * seg_len)


def test_polyline_prism_solids_extension_length_validation():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    with pytest.raises(ValueError):
        polyline_prism_solids(pts, normal, 1.0, 5.0, start_ext=[1.0])
    with pytest.raises(ValueError):
        polyline_prism_solids(pts, normal, 1.0, 5.0, end_ext=[1.0, 2.0, 3.0])


def test_polyline_prism_solids_with_extensions():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    solids = polyline_prism_solids(
        pts, normal, 2.0, 5.0, start_ext=[1.0, 0.5], end_ext=[0.0, 2.0]
    )
    assert len(solids) == 2
    assert np.isclose(solids[0].volume, 2.0 * 5.0 * (10.0 + 1.0))
    assert np.isclose(solids[1].volume, 2.0 * 5.0 * (8.0 + 0.5 + 2.0))


# ---------------- offset_prism ----------------

def test_offset_prism_watertight_and_asymmetric_quad_volume():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    length, width, extrusion = 6.0, 3.0, 2.0
    mesh = offset_prism(pts, normal, 0, length, width, extrusion, offset=4.0)
    _assert_solid(mesh)
    # The prototype's quad spans the full length but only halfW in width
    # (pC/pD have no width component), so area = length * width / 2.
    expected_area = length * (width / 2.0)
    assert np.isclose(mesh.volume, expected_area * extrusion)


def test_offset_prism_center_displaced_by_offset():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    seg_vec = safe_normalize(pts[1] - pts[0])
    plane_vec = safe_normalize(np.cross(normal, seg_vec))
    offset = 4.0
    mesh0 = offset_prism(pts, normal, 0, 6.0, 3.0, 2.0, offset=0.0)
    mesh1 = offset_prism(pts, normal, 0, 6.0, 3.0, 2.0, offset=offset)
    displacement = mesh1.center_mass - mesh0.center_mass
    assert np.allclose(displacement, plane_vec * offset, atol=1e-9)


def test_offset_prism_zero_offset_centroid():
    # With offset 0, the solid's centroid sits at the segment midpoint
    # shifted by -plane_vec * width/4 (the asymmetric-quad centroid).
    pts = polyline_from_angles([0.0], [10.0])
    pts = np.vstack([pts, [10.0, 5.0, 0.0]])  # make 3 points for a normal
    normal = np.array([0.0, 0.0, 1.0])
    width = 3.0
    mesh = offset_prism(pts, normal, 0, 6.0, width, 2.0, offset=0.0)
    seg_vec = safe_normalize(pts[1] - pts[0])
    plane_vec = safe_normalize(np.cross(normal, seg_vec))
    midpoint = (pts[0] + pts[1]) / 2.0
    expected = midpoint - plane_vec * (width / 4.0)
    assert np.allclose(mesh.center_mass, expected, atol=1e-9)


def test_offset_prism_invalid_seg_index_raises():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    with pytest.raises(IndexError):
        offset_prism(pts, normal, -1, 6.0, 3.0, 2.0, 0.0)
    with pytest.raises(IndexError):
        offset_prism(pts, normal, 2, 6.0, 3.0, 2.0, 0.0)


# ---------------- oriented_cylinder ----------------

def test_oriented_cylinder_watertight_and_volume():
    radius, height, sections = 2.0, 10.0, 20
    mesh = oriented_cylinder([0, 0, 0], [0, 0, 1], radius, height, sections)
    _assert_solid(mesh)
    # A 20-gon prism underestimates pi*r^2*h by ~1.6%.
    assert mesh.volume == pytest.approx(math.pi * radius**2 * height, rel=0.02)


def test_oriented_cylinder_axis_alignment():
    radius, height = 2.0, 10.0
    axis = safe_normalize([1.0, 2.0, 3.0])
    center = np.array([5.0, -1.0, 2.0])
    mesh = oriented_cylinder(center, axis, radius, height)
    _assert_solid(mesh)
    # Extent along the axis equals the height.
    proj = (mesh.vertices - center) @ axis
    assert np.isclose(proj.max() - proj.min(), height)
    assert np.isclose(proj.max(), height / 2.0)
    # Radial distance from the axis line never exceeds the radius.
    radial = (mesh.vertices - center) - np.outer(proj, axis)
    radial_dist = np.linalg.norm(radial, axis=1)
    assert radial_dist.max() == pytest.approx(radius, rel=1e-6)
    # Centered at the requested center.
    assert np.allclose(mesh.center_mass, center, atol=1e-9)


def test_oriented_cylinder_unnormalized_axis():
    # Axis direction is what matters, not its magnitude.
    m1 = oriented_cylinder([0, 0, 0], [0.0, 0.0, 7.0], 1.0, 4.0)
    m2 = oriented_cylinder([0, 0, 0], [0.0, 0.0, 1.0], 1.0, 4.0)
    assert np.allclose(m1.bounds, m2.bounds)


# ---------------- paired_cylinders ----------------

def test_paired_cylinders_centers_and_watertight():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    frac, radius, height, offset = 0.25, 1.5, 6.0, 5.0
    cyls = paired_cylinders(pts, normal, 0, frac, radius, height, offset)
    assert len(cyls) == 2
    anchor = pts[0] + frac * (pts[1] - pts[0])
    expected = [anchor + normal * offset, anchor - normal * offset]
    for cyl, center in zip(cyls, expected):
        _assert_solid(cyl)
        assert cyl.volume == pytest.approx(
            math.pi * radius**2 * height, rel=0.02
        )
        assert np.allclose(cyl.center_mass, center, atol=1e-9)


def test_paired_cylinders_axis_is_segment_direction():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    height = 6.0
    cyls = paired_cylinders(pts, normal, 1, 0.5, 1.0, height, 3.0)
    seg_vec = safe_normalize(pts[2] - pts[1])
    for cyl in cyls:
        proj = cyl.vertices @ seg_vec
        assert np.isclose(proj.max() - proj.min(), height)


def test_paired_cylinders_invalid_seg_index_raises():
    pts = polyline_from_angles([0.0, 90.0], [10.0, 8.0])
    normal = polyline_plane_normal(pts)
    with pytest.raises(IndexError):
        paired_cylinders(pts, normal, -1, 0.5, 1.0, 5.0, 2.0)
    with pytest.raises(IndexError):
        paired_cylinders(pts, normal, 2, 0.5, 1.0, 5.0, 2.0)
