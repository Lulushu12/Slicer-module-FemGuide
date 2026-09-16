"""Landmark-derived local coordinate frames (layer 3 helper).

New for the commercial core (docs/ARCHITECTURE.md section 4.1, layer 3,
"landmark-derived local coordinate frames"); there is no direct prototype
counterpart in ``LineGenerator.py``, but the frame construction mirrors the
prototype's ``_make_oriented_cylinder`` 4x4 convention: the rotation columns
are the local X, Y, Z axes expressed in world coordinates and the last column
is the world-space origin.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def frame_from_origin_and_axes(origin, x_axis, xy_vector) -> np.ndarray:
    """Right-handed orthonormal 4x4 frame from an origin and two directions.

    Gram-Schmidt construction: ``X = normalize(x_axis)``,
    ``Z = normalize(cross(x_axis, xy_vector))``, ``Y = cross(Z, X)``, so
    ``xy_vector`` lies in the frame's XY plane on the +Y side. Columns of the
    returned matrix are the X, Y, Z axes; the last column is ``origin``.

    Raises ``ValueError`` when ``x_axis`` is near zero or ``xy_vector`` is
    (anti)parallel to ``x_axis`` (cross-product norm below 1e-8).
    """
    origin = np.asarray(origin, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    xy_vector = np.asarray(xy_vector, dtype=float)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < _EPS:
        raise ValueError("x_axis is near zero; cannot build a frame.")
    x = x_axis / x_norm
    z_raw = np.cross(x_axis, xy_vector)
    z_norm = np.linalg.norm(z_raw)
    if z_norm < _EPS:
        raise ValueError(
            "xy_vector is parallel to x_axis (or near zero); "
            "cannot build a frame."
        )
    z = z_raw / z_norm
    y = np.cross(z, x)
    frame = np.eye(4)
    frame[:3, 0] = x
    frame[:3, 1] = y
    frame[:3, 2] = z
    frame[:3, 3] = origin
    return frame


def frame_from_three_points(origin, point_on_x, point_in_xy) -> np.ndarray:
    """4x4 frame from three landmark points.

    Convenience wrapper around ``frame_from_origin_and_axes`` with
    ``x_axis = point_on_x - origin`` and ``xy_vector = point_in_xy - origin``.
    """
    origin = np.asarray(origin, dtype=float)
    point_on_x = np.asarray(point_on_x, dtype=float)
    point_in_xy = np.asarray(point_in_xy, dtype=float)
    return frame_from_origin_and_axes(origin, point_on_x - origin,
                                      point_in_xy - origin)
