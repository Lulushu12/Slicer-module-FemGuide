"""Layer 1 parametric primitives (docs/ARCHITECTURE.md section 4.1).

Pure numpy + trimesh reimplementation of the geometry math from the Slicer
prototype ``modules/femguide/LineGenerator.py``. No vtk/slicer/qt imports.

Prototype-to-function map:

- ``safe_normalize``            <- prototype ``safe_normalize``
- ``polyline_from_angles``      <- prototype ``generatePolyline``
- ``polyline_plane_normal``     <- prototype ``computePolylinePlaneNormal``
- ``segment_prism`` /
  ``polyline_prism_solids``     <- prototype ``build_primary_prism``
- ``offset_prism``              <- prototype ``build_offset_prism``
- ``oriented_cylinder``         <- prototype ``_make_oriented_cylinder``
- ``paired_cylinders``          <- prototype ``build_cylinders``

All solids returned are watertight ``trimesh.Trimesh`` objects with
outward-facing winding, suitable as input to layer 2 (manifold3d booleans).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

_EPS = 1e-8


def safe_normalize(v, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Normalize a vector, returning ``fallback`` when it is near zero.

    Same semantics as the prototype's ``safe_normalize``: a vector with norm
    below 1e-8 yields the fallback (default +Z) instead of dividing by zero.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.array(fallback, dtype=float)
    return v / n


def polyline_from_angles(angle_increments_deg, lengths) -> np.ndarray:
    """Planar polyline in the z=0 plane from cumulative angle increments.

    Reimplements the prototype's ``generatePolyline``: starting at the origin
    with heading 0, each segment i first adds ``angle_increments_deg[i]``
    (degrees) to the running heading, then advances by ``lengths[i]`` along
    ``(cos(heading), sin(heading), 0)``.

    Returns an ``(N+1, 3)`` float array of points for N segments.
    Raises ``ValueError`` when the two lists differ in length or are empty.
    """
    angle_increments_deg = list(angle_increments_deg)
    lengths = list(lengths)
    if len(angle_increments_deg) != len(lengths):
        raise ValueError("Lengths list must match angle increments list in size.")
    if not angle_increments_deg:
        raise ValueError("At least one segment (angle, length) is required.")
    points = [(0.0, 0.0, 0.0)]
    current_angle = 0.0
    x, y = 0.0, 0.0
    for inc, length in zip(angle_increments_deg, lengths):
        current_angle += math.radians(inc)
        x += length * math.cos(current_angle)
        y += length * math.sin(current_angle)
        points.append((x, y, 0.0))
    return np.array(points, dtype=float)


def polyline_plane_normal(points) -> np.ndarray:
    """Unit normal of the plane spanned by the first three polyline points.

    Reimplements the prototype's ``computePolylinePlaneNormal``. ``points`` is
    an (N, 3) array-like. With fewer than 3 points, or when the first three
    points are (nearly) collinear (cross-product norm below 1e-8), the
    fallback ``(0, 0, 1)`` is returned.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < _EPS:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def _quad_prism(quad, normal, extrusion) -> trimesh.Trimesh:
    """Extrude a planar quad (4x3 array, corners in order) along ``normal``.

    Matches the prototype's vtkLinearExtrusionFilter + translate combination:
    the quad is swept by ``normal * extrusion`` and then shifted by
    ``-normal * extrusion / 2`` so the extrusion is centered on the quad's
    plane. Built directly as 8 corners and 12 triangles; the winding is
    flipped if needed so all faces point outward.
    """
    quad = np.asarray(quad, dtype=float)
    normal = np.asarray(normal, dtype=float)
    half = normal * (extrusion / 2.0)
    bottom = quad - half
    top = quad + half
    vertices = np.vstack([bottom, top])
    # Quad corners 0..3, bottom = 0..3, top = 4..7.
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom cap
            [4, 5, 6], [4, 6, 7],  # top cap
            [0, 1, 5], [0, 5, 4],  # side A-B
            [1, 2, 6], [1, 6, 5],  # side B-C
            [2, 3, 7], [2, 7, 6],  # side C-D
            [3, 0, 4], [3, 4, 7],  # side D-A
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def segment_prism(p0, p1, normal, thickness, extrusion, start_ext=0.0,
                  end_ext=0.0) -> trimesh.Trimesh:
    """One segment of the prototype's primary prism (``build_primary_prism``).

    ``p0`` is extended backward by ``start_ext`` and ``p1`` forward by
    ``end_ext`` along the segment direction. The in-plane thickness vector is
    ``-normalize(cross(normal, seg_dir)) * thickness`` (the prototype's sign),
    giving the quad ``p0, p0+thick, p1+thick, p1``. The quad is extruded by
    ``extrusion`` total along ``normal``, centered on the polyline plane
    (+-extrusion/2), matching the prototype's post-extrusion translate.

    Returns a watertight box-like solid.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    normal = np.asarray(normal, dtype=float)
    seg_vec = safe_normalize(p1 - p0)
    p0 = p0 - seg_vec * start_ext
    p1 = p1 + seg_vec * end_ext
    thick_vec = -safe_normalize(np.cross(normal, seg_vec)) * thickness
    quad = np.array([p0, p0 + thick_vec, p1 + thick_vec, p1])
    return _quad_prism(quad, normal, extrusion)


def polyline_prism_solids(points, normal, thickness, extrusion, start_ext=None,
                          end_ext=None):
    """Per-segment solids of the prototype's primary prism.

    The prototype's ``build_primary_prism`` extruded all segment quads in one
    polydata; here each segment becomes its own watertight solid (one
    ``segment_prism`` per segment), to be unioned by layer 2. ``start_ext`` /
    ``end_ext`` are per-segment sequences (default all zeros); a
    ``ValueError`` is raised when their length does not match the segment
    count.

    Returns a list of ``trimesh.Trimesh``.
    """
    points = np.asarray(points, dtype=float)
    n_segments = points.shape[0] - 1
    if n_segments < 1:
        raise ValueError("At least two points are required.")
    if start_ext is None:
        start_ext = [0.0] * n_segments
    if end_ext is None:
        end_ext = [0.0] * n_segments
    start_ext = list(start_ext)
    end_ext = list(end_ext)
    if len(start_ext) != n_segments:
        raise ValueError(
            "start_ext must have one entry per segment "
            "(%d given, %d segments)." % (len(start_ext), n_segments)
        )
    if len(end_ext) != n_segments:
        raise ValueError(
            "end_ext must have one entry per segment "
            "(%d given, %d segments)." % (len(end_ext), n_segments)
        )
    solids = []
    for i in range(n_segments):
        solids.append(
            segment_prism(points[i], points[i + 1], normal, thickness,
                          extrusion, start_ext[i], end_ext[i])
        )
    return solids


def offset_prism(points, normal, seg_index, length, width_inplane, extrusion,
                 offset) -> trimesh.Trimesh:
    """The prototype's secondary/tertiary prism (``build_offset_prism``).

    Centered on the midpoint of segment ``seg_index`` displaced by
    ``normalize(cross(normal, seg_dir)) * offset``, then extruded by
    ``extrusion`` total along ``normal``, centered as in ``segment_prism``.

    NOTE (inherited prototype behavior, replicated exactly): the base quad is
    asymmetric about the displaced center. With ``halfL = seg_dir * length/2``
    and ``halfW = plane_vec * width_inplane/2`` the corners are::

        pA = center - halfL - halfW
        pB = center + halfL - halfW
        pC = center + halfL            # no width component
        pD = center - halfL            # no width component

    i.e. the quad spans the full ``length`` along the segment but only
    ``width_inplane / 2`` in-plane, entirely on the ``-plane_vec`` side of the
    center line.

    Raises ``IndexError`` for an invalid ``seg_index``.
    """
    points = np.asarray(points, dtype=float)
    n_segments = points.shape[0] - 1
    if seg_index < 0 or seg_index >= n_segments:
        raise IndexError(
            "seg_index %d out of range for %d segments." % (seg_index, n_segments)
        )
    normal = np.asarray(normal, dtype=float)
    p0 = points[seg_index]
    p1 = points[seg_index + 1]
    seg_vec = safe_normalize(p1 - p0)
    plane_vec = safe_normalize(np.cross(normal, seg_vec))
    center = (p0 + p1) / 2.0 + plane_vec * offset
    half_length = seg_vec * (length / 2.0)
    half_width = plane_vec * (width_inplane / 2.0)
    quad = np.array(
        [
            center - half_length - half_width,
            center + half_length - half_width,
            center + half_length,
            center - half_length,
        ]
    )
    return _quad_prism(quad, normal, extrusion)


def oriented_cylinder(center, axis, radius, height, sections=20) -> trimesh.Trimesh:
    """Solid cylinder centered at ``center`` with its axis along ``axis``.

    Reimplements the prototype's ``_make_oriented_cylinder``. The prototype
    mapped VTK's cylinder Y axis onto the segment direction via a 4x4 local
    frame; here the axis is taken directly and trimesh's Z-aligned cylinder is
    rotated onto it. ``sections`` (default 20) matches the prototype's
    ``SetResolution(20)``.
    """
    center = np.asarray(center, dtype=float)
    axis_unit = safe_normalize(axis)
    mesh = trimesh.creation.cylinder(radius=radius, height=height,
                                     sections=sections)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis_unit)
    transform = np.array(transform, dtype=float)
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    return mesh


def paired_cylinders(points, normal, seg_index, frac_along, radius, height,
                     offset_from_plane):
    """Two cylinders above/below the polyline plane on a chosen segment.

    Reimplements the prototype's ``build_cylinders``: with
    ``anchor = p0 + frac_along * (p1 - p0)`` on segment ``seg_index``, the two
    cylinder centers are ``anchor +- normal * offset_from_plane`` and both
    axes run along the segment direction.

    Returns a list of two ``trimesh.Trimesh``.
    Raises ``IndexError`` for an invalid ``seg_index``.
    """
    points = np.asarray(points, dtype=float)
    n_segments = points.shape[0] - 1
    if seg_index < 0 or seg_index >= n_segments:
        raise IndexError(
            "seg_index %d out of range for %d segments." % (seg_index, n_segments)
        )
    normal = np.asarray(normal, dtype=float)
    p0 = points[seg_index]
    p1 = points[seg_index + 1]
    seg_vec = safe_normalize(p1 - p0)
    anchor = p0 + frac_along * (p1 - p0)
    center1 = anchor + normal * offset_from_plane
    center2 = anchor - normal * offset_from_plane
    return [
        oriented_cylinder(center1, seg_vec, radius, height),
        oriented_cylinder(center2, seg_vec, radius, height),
    ]
