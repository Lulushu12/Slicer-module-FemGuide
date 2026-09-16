"""Solid splitting along a plane (layer 2).

This is the future mold parting-plane operation (docs/ARCHITECTURE.md 6.3):
the hip-spacer mold body is split into two capped, watertight halves along a
parting plane. Implemented on manifold3d per frozen decision D3.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh
from manifold3d import Manifold

from femguide_core.geometry.booleans import from_manifold, to_manifold


def split_by_plane(
    mesh: trimesh.Trimesh,
    plane_origin,
    plane_normal,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Split a manifold solid into two capped halves along a plane.

    Returns ``(above, below)``: ``above`` is the half on the side the normal
    points toward, ``below`` the other half. Both halves are capped and
    watertight (manifold3d closes the cut faces). A plane that misses the
    solid returns the whole solid on one side and an empty mesh on the other.

    The input is validated via ``to_manifold`` and raises
    :class:`~femguide_core.geometry.booleans.NonManifoldError` for
    non-manifold input — run ``repair()`` on segmentation output first.

    Parameters
    ----------
    mesh:
        Closed manifold solid to split.
    plane_origin:
        Any point (3,) on the cutting plane.
    plane_normal:
        Plane normal (3,); need not be unit length.
    """
    solid = to_manifold(mesh, "mesh")

    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("plane_normal must be a non-zero vector")
    unit = normal / length
    # manifold3d expresses the plane as (normal, offset from origin along it).
    origin_offset = float(np.dot(unit, origin))

    if hasattr(solid, "split_by_plane"):
        above_m, below_m = solid.split_by_plane(tuple(unit), origin_offset)
    else:  # pragma: no cover - older manifold3d without split_by_plane
        above_m = solid.trim_by_plane(tuple(unit), origin_offset)
        below_m = solid.trim_by_plane(tuple(-unit), -origin_offset)

    return from_manifold(above_m), from_manifold(below_m)
