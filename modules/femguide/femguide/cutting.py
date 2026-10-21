"""Cutting structure geometry: the slot and the walls that capture the blade.

The prototype's primary prism is the slot (clarified 2026-10-21), so the
commercial cutting structure is built as two solids from the same polyline:

- ``slot``: the blade's full path: one strip per segment, ``slot_width``
  wide and centred on the cut line, extending far past the guide in both
  directions along local z. Subtracting it opens the channel through the
  walls, the base and anything else in the blade's plane, whatever way the
  placement is oriented: everything on the blade's path must be cut.
- ``walls``: the material around it, ``slot_width + 2 * wall_thickness``
  wide and ``slot_depth`` tall, extended by ``wall_thickness`` past every
  slot end so the slot never breaks out of the wall.

At interior polyline vertices, per-segment boxes leave a wedge-shaped gap on
the outer side of each bend; a cylinder along local z at every interior
vertex fills it (rounded joints) for both solids.

Deviation from the prototype: the prototype offset its strip to one side of
the polyline. Here the slot is centred, so the polyline is exactly where the
blade cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import trimesh

from femguide_core.geometry import (
    oriented_cylinder,
    polyline_from_angles,
    safe_normalize,
    segment_prism,
    union,
)

from femguide.plan import CuttingParams

_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0])
# Half-length of the blade path along local z. Far larger than any guide,
# so the slot is open through all guide material in the blade's plane.
BLADE_PATH_HALF_MM = 250.0
_JOINT_SECTIONS = 32


@dataclass
class CuttingSolids:
    walls: trimesh.Trimesh
    slot: trimesh.Trimesh
    polyline: np.ndarray  # (N+1, 3) cut line in bone coordinates


def _centred_strips(points, width, height, start_ext, end_ext) -> List[trimesh.Trimesh]:
    """One box per segment, ``width`` wide and centred on the segment.

    Reuses the prototype-faithful ``segment_prism``, whose thickness goes to
    the ``-cross(normal, seg)`` side, by first shifting the segment half a
    width toward the opposite side.
    """
    solids = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        side = safe_normalize(np.cross(_LOCAL_NORMAL, safe_normalize(p1 - p0)))
        shift = side * (width / 2.0)
        solids.append(segment_prism(p0 + shift, p1 + shift, _LOCAL_NORMAL,
                                    width, height, start_ext[i], end_ext[i]))
    return solids


def _joints(points, width, height) -> List[trimesh.Trimesh]:
    return [oriented_cylinder(p, _LOCAL_NORMAL,
                              width / 2.0, height, sections=_JOINT_SECTIONS)
            for p in points[1:-1]]


def cutting_solids(params: CuttingParams) -> CuttingSolids:
    """Build the slot and wall solids for ``params``, placed in bone space."""
    points = polyline_from_angles(params.angles_deg, params.lengths_mm)
    wall = params.wall_thickness_mm
    slot_w = params.slot_width_mm
    wall_w = slot_w + 2.0 * wall

    slot_h = 2.0 * BLADE_PATH_HALF_MM
    slot_parts = _centred_strips(points, slot_w, slot_h, params.start_ext_mm,
                                 params.end_ext_mm)
    slot_parts += _joints(points, slot_w, slot_h)

    wall_start = [e + wall for e in params.start_ext_mm]
    wall_end = [e + wall for e in params.end_ext_mm]
    wall_parts = _centred_strips(points, wall_w, params.slot_depth_mm,
                                 wall_start, wall_end)
    wall_parts += _joints(points, wall_w, params.slot_depth_mm)

    placement = params.placement_matrix
    walls = union(wall_parts)
    slot = union(slot_parts)
    walls.apply_transform(placement)
    slot.apply_transform(placement)
    return CuttingSolids(walls=walls, slot=slot,
                         polyline=trimesh.transform_points(points, placement))
