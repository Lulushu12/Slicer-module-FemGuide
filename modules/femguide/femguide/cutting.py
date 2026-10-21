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

Auto-connect (when a bone is given): the walls are a block the surgeon
places, plus an automatic extension along local -z (toward the bone) that
reaches down into the patient-specific base under the block. The extension
is clipped to the base's shadow along local z, so it only ever lands on the
base: never on bone outside the outline the surgeon drew (where soft tissue
usually is). Then everything closer to the bone than the bone-fit clearance
is cut away, so the block may even be pushed into the bone. Wherever the
block is moved, the connection is recomputed on every regeneration (D4).

Without a base the extension reaches the bone itself (and the plan already
carries a NO_BASE warning). A block with no base (or bone) under it gets no
extension, and validation reports it as disconnected.

The clearance is measured along local z, so on bone sloped at an angle a to
the walls the true perpendicular gap is clearance * cos(a).

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
    NonManifoldError,
    difference,
    intersection,
    oriented_cylinder,
    polyline_from_angles,
    remove_small_bodies,
    safe_normalize,
    segment_prism,
    union,
)

from femguide_core.geometry.surface import DUST_MM3

from femguide.plan import CuttingParams

_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0])
# Half-length of the blade path along local z. Far larger than any guide,
# so the slot is open through all guide material in the blade's plane.
BLADE_PATH_HALF_MM = 250.0
_JOINT_SECTIONS = 32
# Auto-connect: the extension overshoots the clearance surface by this much so
# the trim, not the extension's end, shapes the walls' feet.
_EXTENSION_OVERSHOOT_MM = 1.0
# Margin around the walls when cropping the bone for the trim.
_CROP_MARGIN_MM = 2.0
# Footprint samples per segment (and across the wall width) used to find how
# far the bone lies below the walls.
_FOOTPRINT_SAMPLES = 9


@dataclass
class CuttingSolids:
    walls: trimesh.Trimesh
    slot: trimesh.Trimesh
    polyline: np.ndarray  # (N+1, 3) cut line in bone coordinates
    extension_mm: float = 0.0  # auto-connect extension below the block


def _centred_strips(points, width, height, start_ext, end_ext,
                    z_shift=0.0) -> List[trimesh.Trimesh]:
    """One box per segment, ``width`` wide and centred on the segment.

    Reuses the prototype-faithful ``segment_prism``, whose thickness goes to
    the ``-cross(normal, seg)`` side, by first shifting the segment half a
    width toward the opposite side.
    """
    solids = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        side = safe_normalize(np.cross(_LOCAL_NORMAL, safe_normalize(p1 - p0)))
        shift = side * (width / 2.0) + _LOCAL_NORMAL * z_shift
        solids.append(segment_prism(p0 + shift, p1 + shift, _LOCAL_NORMAL,
                                    width, height, start_ext[i], end_ext[i]))
    return solids


def _joints(points, width, height, z_shift=0.0) -> List[trimesh.Trimesh]:
    return [oriented_cylinder(p + _LOCAL_NORMAL * z_shift, _LOCAL_NORMAL,
                              width / 2.0, height, sections=_JOINT_SECTIONS)
            for p in points[1:-1]]


def _footprint_samples(points, wall_w, start_ext, end_ext) -> np.ndarray:
    """Local (x, y) sample points covering the walls' footprint."""
    samples = []
    for i in range(len(points) - 1):
        u = safe_normalize(points[i + 1] - points[i])
        side = np.cross(_LOCAL_NORMAL, u)
        a = points[i] - u * start_ext[i]
        b = points[i + 1] + u * end_ext[i]
        for t in np.linspace(0.0, 1.0, _FOOTPRINT_SAMPLES):
            for w in np.linspace(-wall_w / 2.0, wall_w / 2.0, 3):
                samples.append((a + t * (b - a) + side * w)[:2])
    return np.array(samples)


def _bone_heights(samples_xy, triangles_local, below_z) -> np.ndarray:
    """For each local (x, y), the highest bone-surface z at or below
    ``below_z`` straight under it (NaN where there is none)."""
    tri = triangles_local
    lo, hi = tri[:, :, :2].min(axis=1), tri[:, :, :2].max(axis=1)
    smin, smax = samples_xy.min(axis=0), samples_xy.max(axis=0)
    keep = np.all((hi >= smin) & (lo <= smax), axis=1)
    tri = tri[keep]
    heights = np.full(len(samples_xy), np.nan)
    if len(tri) == 0:
        return heights
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1 = (b - a)[:, :2], (c - a)[:, :2]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(det) > 1e-12
    a, b, c, v0, v1, det = a[ok], b[ok], c[ok], v0[ok], v1[ok], det[ok]
    for k, p in enumerate(samples_xy):
        d = p[None, :] - a[:, :2]
        beta = (d[:, 0] * v1[:, 1] - d[:, 1] * v1[:, 0]) / det
        gamma = (v0[:, 0] * d[:, 1] - v0[:, 1] * d[:, 0]) / det
        inside = (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1)
        if not inside.any():
            continue
        z = (a[inside, 2] + beta[inside] * (b[inside, 2] - a[inside, 2])
             + gamma[inside] * (c[inside, 2] - a[inside, 2]))
        z = z[z <= below_z]
        if len(z):
            heights[k] = z.max()
    return heights


def _extension_needed(params: CuttingParams, points, wall_w, target, extra) -> float:
    """How far the walls must extend below the block to reach ``target``
    (the base, or the bone when there is no base) everywhere under their
    footprint, plus ``extra``; 0 if nothing of it lies below them."""
    inverse = np.linalg.inv(params.placement_matrix)
    triangles_local = trimesh.transform_points(
        target.triangles.reshape(-1, 3), inverse).reshape(-1, 3, 3)
    wall_start = [e + params.wall_thickness_mm for e in params.start_ext_mm]
    wall_end = [e + params.wall_thickness_mm for e in params.end_ext_mm]
    samples = _footprint_samples(points, wall_w, wall_start, wall_end)
    half = params.slot_depth_mm / 2.0
    heights = _bone_heights(samples, triangles_local, below_z=half)
    if np.all(np.isnan(heights)):
        return 0.0
    gap = -half - np.nanmin(heights)         # block bottom to lowest surface below
    return max(0.0, gap + extra + _EXTENSION_OVERSHOOT_MM)


def _shadow_column(solid, placement, z_low, z_high):
    """``solid``'s shadow along local z, as a prism spanning [z_low, z_high]
    in the local frame, placed back in bone space."""
    from manifold3d import Manifold

    from femguide_core.geometry import from_manifold, to_manifold

    local = to_manifold(solid, "base").transform(np.linalg.inv(placement)[:3, :])
    column = Manifold.extrude(local.project(), z_high - z_low).translate((0.0, 0.0, z_low))
    return from_manifold(column.transform(placement[:3, :]))


def _trim_to_bone(walls, bone, placement, clearance) -> trimesh.Trimesh:
    """Cut away everything inside the bone or within ``clearance`` of it
    (measured along local z), using the bone cropped to the walls'
    surroundings. Subtracting the bone itself as well as its shifted copy
    keeps the walls out of the bone even under noise that overhangs relative
    to local z. Dust left by the cut is dropped."""
    lo, hi = walls.bounds
    margin = _CROP_MARGIN_MM + abs(clearance)
    box = trimesh.creation.box(bounds=(lo - margin, hi + margin))
    crop = intersection(bone, box)
    if len(crop.faces) == 0:
        return walls
    moved = crop.copy()
    moved.apply_translation(placement[:3, 2] * clearance)
    return remove_small_bodies(difference(walls, [crop, moved]), DUST_MM3)


def cutting_solids(params: CuttingParams, bone: trimesh.Trimesh = None,
                   bone_clearance_mm: float = 0.0,
                   base: trimesh.Trimesh = None) -> CuttingSolids:
    """Build the slot and wall solids for ``params``, placed in bone space.

    With ``bone``, the walls auto-connect (see module docstring): down into
    ``base`` when given, else down to the bone; then trimmed to
    ``bone_clearance_mm`` from the bone. A bone that is not a closed solid
    cannot be trimmed against; the walls are then left unconnected and
    validation blocks the plan.
    """
    points = polyline_from_angles(params.angles_deg, params.lengths_mm)
    wall = params.wall_thickness_mm
    slot_w = params.slot_width_mm
    wall_w = slot_w + 2.0 * wall
    depth = params.slot_depth_mm
    placement = params.placement_matrix

    slot_h = 2.0 * BLADE_PATH_HALF_MM
    slot_parts = _centred_strips(points, slot_w, slot_h, params.start_ext_mm,
                                 params.end_ext_mm)
    slot_parts += _joints(points, slot_w, slot_h)
    slot = union(slot_parts)
    slot.apply_transform(placement)

    wall_start = [e + wall for e in params.start_ext_mm]
    wall_end = [e + wall for e in params.end_ext_mm]

    def wall_prism(height, z_shift):
        parts = _centred_strips(points, wall_w, height, wall_start, wall_end,
                                z_shift=z_shift)
        parts += _joints(points, wall_w, height, z_shift=z_shift)
        solid = union(parts)
        solid.apply_transform(placement)
        return solid

    walls = wall_prism(depth, 0.0)
    extension = 0.0
    if bone is not None:
        try:
            if base is not None:
                extension = _extension_needed(params, points, wall_w, base, 0.0)
            else:
                extension = _extension_needed(params, points, wall_w, bone,
                                              bone_clearance_mm)
            if extension > 0.0:
                # Overlap the block slightly so the union has no coincident faces.
                overlap = 0.1
                below = wall_prism(extension + overlap,
                                   -depth / 2.0 - extension / 2.0 + overlap / 2.0)
                if base is not None:
                    below = intersection(below, _shadow_column(
                        base, placement, -depth / 2.0 - extension - 1.0,
                        -depth / 2.0 + overlap + 1.0))
                if len(below.faces):
                    walls = union([walls, below])
            walls = _trim_to_bone(walls, bone, placement, bone_clearance_mm)
        except NonManifoldError:
            extension = 0.0
    return CuttingSolids(walls=walls, slot=slot,
                         polyline=trimesh.transform_points(points, placement),
                         extension_mm=extension)
