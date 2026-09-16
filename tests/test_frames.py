"""Tests for landmark-derived coordinate frames.

Imports the module directly (not via the femguide_core package root, which
pulls in the other geometry layers).
"""

import numpy as np
import pytest

try:
    from femguide_core.geometry.frames import (
        frame_from_origin_and_axes,
        frame_from_three_points,
    )
except ImportError:
    # The geometry package __init__ imports the other layers, which may not
    # exist yet during parallel development. Load frames.py directly.
    import importlib.util
    from pathlib import Path

    _path = (
        Path(__file__).resolve().parents[1]
        / "core" / "femguide_core" / "geometry" / "frames.py"
    )
    _spec = importlib.util.spec_from_file_location("_fg_frames", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    frame_from_origin_and_axes = _mod.frame_from_origin_and_axes
    frame_from_three_points = _mod.frame_from_three_points


def _check_frame(frame, origin):
    assert frame.shape == (4, 4)
    rotation = frame[:3, :3]
    # Orthonormal.
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    # Right-handed.
    assert np.isclose(np.linalg.det(rotation), 1.0)
    # Origin column and homogeneous bottom row.
    assert np.allclose(frame[:3, 3], origin)
    assert np.allclose(frame[3], [0.0, 0.0, 0.0, 1.0])


def test_frame_identity_case():
    frame = frame_from_origin_and_axes([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert np.allclose(frame, np.eye(4))


def test_frame_axes_gram_schmidt():
    origin = [1.0, 2.0, 3.0]
    frame = frame_from_origin_and_axes(origin, [2.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    _check_frame(frame, origin)
    # X is the normalized x_axis.
    assert np.allclose(frame[:3, 0], [1.0, 0.0, 0.0])
    # Z = normalize(cross(x_axis, xy_vector)) = +z here.
    assert np.allclose(frame[:3, 2], [0.0, 0.0, 1.0])
    # Y = cross(Z, X) = +y; the xy_vector lies on the +Y side.
    assert np.allclose(frame[:3, 1], [0.0, 1.0, 0.0])


def test_frame_non_orthogonal_inputs_orthonormalized():
    origin = [0.0, 0.0, 0.0]
    x_axis = [1.0, 1.0, 0.0]
    xy_vector = [0.0, 3.0, 1.0]  # not perpendicular to x_axis
    frame = frame_from_origin_and_axes(origin, x_axis, xy_vector)
    _check_frame(frame, origin)
    assert np.allclose(frame[:3, 0], np.array(x_axis) / np.sqrt(2.0))
    # xy_vector lies in the frame's XY plane (no Z component)...
    z = frame[:3, 2]
    assert np.isclose(np.dot(z, xy_vector), 0.0, atol=1e-12)
    # ...on the +Y side.
    assert np.dot(frame[:3, 1], xy_vector) > 0


def test_frame_zero_x_axis_raises():
    with pytest.raises(ValueError):
        frame_from_origin_and_axes([0, 0, 0], [0.0, 0.0, 0.0], [0, 1, 0])
    with pytest.raises(ValueError):
        frame_from_origin_and_axes([0, 0, 0], [1e-12, 0.0, 0.0], [0, 1, 0])


def test_frame_parallel_xy_vector_raises():
    with pytest.raises(ValueError):
        frame_from_origin_and_axes([0, 0, 0], [1, 0, 0], [2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        frame_from_origin_and_axes([0, 0, 0], [1, 0, 0], [-3.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        frame_from_origin_and_axes([0, 0, 0], [1, 0, 0], [0.0, 0.0, 0.0])


def test_frame_from_three_points():
    origin = np.array([10.0, 0.0, 5.0])
    point_on_x = origin + np.array([0.0, 4.0, 0.0])
    point_in_xy = origin + np.array([1.0, 1.0, 0.0])
    frame = frame_from_three_points(origin, point_on_x, point_in_xy)
    _check_frame(frame, origin)
    assert np.allclose(frame[:3, 0], [0.0, 1.0, 0.0])
    # Matches the underlying axes constructor.
    expected = frame_from_origin_and_axes(
        origin, point_on_x - origin, point_in_xy - origin
    )
    assert np.allclose(frame, expected)


def test_frame_from_three_points_collinear_raises():
    with pytest.raises(ValueError):
        frame_from_three_points([0, 0, 0], [1, 0, 0], [2, 0, 0])
    with pytest.raises(ValueError):
        frame_from_three_points([0, 0, 0], [0, 0, 0], [0, 1, 0])
