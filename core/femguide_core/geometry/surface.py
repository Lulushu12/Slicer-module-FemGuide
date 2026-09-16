"""Layer 3 anatomy operations (docs/ARCHITECTURE.md sections 4.1 and 6.1).

The "draw footprint on bone -> conforming base -> subtract pin cylinders"
pipeline shared by FemGuide (6.1 item 2) and Glenoid Guide (6.2):

1. :func:`snap_points_to_surface` — closest points on a mesh surface.
2. :func:`project_curve_to_surface` — a drawn closed curve, densified and
   snapped onto the bone surface.
3. :func:`extract_patch` — the sub-surface enclosed by that curve (flood fill
   behind a barrier band of faces near the curve).
4. :func:`conforming_solid_from_patch` — the open patch offset into a
   watertight "molded base" solid whose underside is a negative of the bone.
5. :func:`molded_base` — the full pipeline in one call; pin cylinders are then
   subtracted by layer 2 (:func:`femguide_core.geometry.booleans.difference`).

Pure numpy + trimesh + manifold3d. The frozen dependency set has no scipy /
networkx / rtree, so ``trimesh.proximity`` (rtree) is unusable here; closest
points are computed with a vectorized exact point-to-triangle projection
(Ericson's region test) behind a cheap centroid-distance prefilter that
replaces the missing spatial index. Mesh sizes in this project are
10^3–10^5 faces, where this stays well under a second per query batch.
Face adjacency and boundary loops are likewise built locally from sorted edge
pairs via ``trimesh.grouping.group_rows`` (pure numpy).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import trimesh

from femguide_core.geometry.repair import is_manifold_solid, repair

# Chunking budget for brute-force closest-point queries: at most this many
# (point, triangle) pairs are expanded at once (~8 MB per scalar temporary).
_PAIR_BUDGET = 1_000_000

# Barrier tolerance for extract_patch, as a fraction of the mesh's median
# edge length. Every face the dense curve actually crosses has distance ~0 and
# is always caught; the factor only controls how far the barrier band bleeds
# sideways. 0.5 keeps the band tight (patch boundary error ~ half a face)
# while still swallowing the whole vertex fan when the curve passes near a
# vertex, which is what prevents diagonal (vertex-only) flood-fill leaks.
_BARRIER_EDGE_FACTOR = 0.5

# extract_patch declares the flood fill "escaped" past this fraction of all
# faces — an enclosed footprint is always far smaller than the whole bone.
_ESCAPE_FRACTION = 0.6

# Laplacian smoothing rounds for displacement normals (same idea as
# offset.py's _smoothed_vertex_normals, duplicated locally on purpose — no
# imports of private helpers across modules).
_SMOOTHING_ROUNDS = 2


# ---------------------------------------------------------------------------
# Brute-force closest point on surface (no scipy / rtree)
# ---------------------------------------------------------------------------

def _paired_closest(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point on triangle ``(a[k], b[k], c[k])`` to point ``p[k]``.

    All inputs are (K, 3); returns ``(closest, dist2)`` of shapes (K, 3) and
    (K,). Implementation is Ericson's closest-point-on-triangle region test
    ("Real-Time Collision Detection", 5.1.5), vectorized over the K pairs.
    Degenerate (zero-area) triangles fall through the guarded divisions to a
    vertex or edge, which is still a point of the triangle.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c
    d1 = np.einsum("ki,ki->k", ab, ap)
    d2 = np.einsum("ki,ki->k", ac, ap)
    d3 = np.einsum("ki,ki->k", ab, bp)
    d4 = np.einsum("ki,ki->k", ac, bp)
    d5 = np.einsum("ki,ki->k", ab, cp)
    d6 = np.einsum("ki,ki->k", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return num / np.where(np.abs(den) < 1e-30, 1.0, den)

    # Start from the interior (barycentric) candidate, then overwrite with
    # edge candidates, then vertex candidates — later writes win, so the
    # highest-priority (vertex) regions are applied last.
    denom = _safe_div(np.ones_like(va), va + vb + vc)
    v = (vb * denom)[:, None]
    w = (vc * denom)[:, None]
    closest = a + v * ab + w * ac

    edge_bc = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    t = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[:, None]
    closest = np.where(edge_bc[:, None], b + t * (c - b), closest)

    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    t = _safe_div(d2, d2 - d6)[:, None]
    closest = np.where(edge_ac[:, None], a + t * ac, closest)

    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    t = _safe_div(d1, d1 - d3)[:, None]
    closest = np.where(edge_ab[:, None], a + t * ab, closest)

    closest = np.where(((d6 >= 0.0) & (d5 <= d6))[:, None], c, closest)
    closest = np.where(((d3 >= 0.0) & (d4 <= d3))[:, None], b, closest)
    closest = np.where(((d1 <= 0.0) & (d2 <= 0.0))[:, None], a, closest)

    dist2 = np.einsum("ki,ki->k", closest - p, closest - p)
    return closest, dist2


def _face_geometry(
    mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangle corners plus centroids and enclosing radii for prefiltering.

    ``radii[m]`` is the largest corner-to-centroid distance of triangle
    ``m``, so the whole triangle lies within ``radii[m]`` of its centroid and
    ``|point - centroid| - radii[m]`` is a valid lower bound on the exact
    point-to-triangle distance.
    """
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    centroids = tri.mean(axis=1)
    radii = np.sqrt(
        np.einsum("mvi,mvi->mv", tri - centroids[:, None, :], tri - centroids[:, None, :])
    ).max(axis=1)
    return a, b, c, centroids, radii


def _snap_to_faces(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Exact closest surface points via centroid prefilter + paired Ericson.

    Brute force over all (point, triangle) pairs would be exact but slow at
    10^5 faces; instead, for each point the cheap centroid distance yields an
    upper bound ``min(|p - centroid| + radius)`` on the true distance, and
    only triangles whose lower bound ``|p - centroid| - radius`` does not
    exceed it can host the closest point. The exact Ericson test then runs on
    those few surviving pairs only. Chunked so the scalar centroid-distance
    grid stays within :data:`_PAIR_BUDGET` pairs.
    """
    out = np.empty_like(points)
    step = max(1, _PAIR_BUDGET // max(1, len(centroids)))
    for start in range(0, len(points), step):
        chunk = points[start : start + step]
        diff = chunk[:, None, :] - centroids[None, :, :]
        dist = np.sqrt(np.einsum("pmi,pmi->pm", diff, diff))
        upper = (dist + radii[None, :]).min(axis=1)
        point_idx, tri_idx = np.nonzero(dist - radii[None, :] <= upper[:, None])
        closest, dist2 = _paired_closest(
            chunk[point_idx], a[tri_idx], b[tri_idx], c[tri_idx]
        )
        best = np.full(len(chunk), np.inf)
        np.minimum.at(best, point_idx, dist2)
        winners = dist2 <= best[point_idx]
        out[start + point_idx[winners]] = closest[winners]
    return out


def _barrier_faces(
    mesh: trimesh.Trimesh, samples: np.ndarray, tolerance: float
) -> np.ndarray:
    """Boolean mask of faces within ``tolerance`` of any curve sample.

    Same prefilter idea as :func:`_snap_to_faces`: a face can only be within
    ``tolerance`` of a sample if the sample lies within
    ``tolerance + radius`` of the face centroid; the exact point-to-triangle
    test runs only on those pairs.
    """
    a, b, c, centroids, radii = _face_geometry(mesh)
    barrier = np.zeros(len(centroids), dtype=bool)
    step = max(1, _PAIR_BUDGET // max(1, len(centroids)))
    for start in range(0, len(samples), step):
        chunk = samples[start : start + step]
        diff = chunk[:, None, :] - centroids[None, :, :]
        dist2 = np.einsum("pmi,pmi->pm", diff, diff)
        reach = (tolerance + radii[None, :]) ** 2
        point_idx, tri_idx = np.nonzero(dist2 <= reach)
        if len(point_idx) == 0:
            continue
        _, exact2 = _paired_closest(
            chunk[point_idx], a[tri_idx], b[tri_idx], c[tri_idx]
        )
        hit = exact2 < tolerance * tolerance
        barrier[tri_idx[hit]] = True
    return barrier


def snap_points_to_surface(mesh: trimesh.Trimesh, points) -> np.ndarray:
    """Closest points on the surface of ``mesh`` for arbitrary 3D points.

    ``points`` is (N, 3) array-like (a single (3,) point is accepted and
    treated as N=1). Returns the (N, 3) array of exact closest points on the
    triangulated surface — for a point already on the surface this is the
    point itself. Implemented as a chunked brute-force point-to-triangle
    projection (see module docstring: no scipy/rtree available for
    ``trimesh.proximity``), which is exact and fast enough at the 10^3–10^5
    face sizes this project uses.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("mesh has no faces to snap to")
    a, b, c, centroids, radii = _face_geometry(mesh)
    return _snap_to_faces(pts, a, b, c, centroids, radii)


# ---------------------------------------------------------------------------
# Curve projection
# ---------------------------------------------------------------------------

def project_curve_to_surface(
    mesh: trimesh.Trimesh, curve_points, samples_per_segment: int = 10
) -> np.ndarray:
    """Project a CLOSED polyline onto the mesh surface as a dense loop.

    ``curve_points`` (N, 3) is treated as a closed polyline — the last point
    connects back to the first. Each of the N segments is densified with
    ``samples_per_segment`` linearly interpolated samples (the segment start
    included, its end excluded, so nothing is duplicated), and every sample is
    snapped to the surface with :func:`snap_points_to_surface`.

    Returns an ordered (N * samples_per_segment, 3) closed loop of on-surface
    points; the first point is NOT repeated at the end.
    """
    pts = np.asarray(curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("curve_points must be an (N, 3) array with N >= 2")
    samples_per_segment = int(samples_per_segment)
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be >= 1")
    starts = pts
    ends = np.roll(pts, -1, axis=0)
    t = (np.arange(samples_per_segment, dtype=np.float64) / samples_per_segment)
    # (N, S, 3): start + t * (end - start), then flattened in curve order.
    dense = starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]
    return snap_points_to_surface(mesh, dense.reshape(-1, 3))


# ---------------------------------------------------------------------------
# Enclosed patch extraction (flood fill behind a barrier band)
# ---------------------------------------------------------------------------

def _face_adjacency_pairs(mesh: trimesh.Trimesh) -> np.ndarray:
    """(K, 2) pairs of face indices sharing an interior edge.

    Built locally from sorted edge rows via ``trimesh.grouping.group_rows``
    (``mesh.edges`` lists the three directed edges of face ``i`` at rows
    ``3i .. 3i+2``, so an edge row index // 3 is its face). trimesh 5's
    ``mesh.face_adjacency`` happens to work without scipy, but this project's
    frozen dependency set makes no promises about optional trimesh backends,
    so the pure-numpy construction is used unconditionally.
    """
    undirected = np.sort(mesh.edges, axis=1)
    groups = trimesh.grouping.group_rows(undirected, require_count=2)
    if len(groups) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(groups, dtype=np.int64) // 3


def _flood(pairs: np.ndarray, allowed: np.ndarray, seed_face: int) -> np.ndarray:
    """Boolean mask of faces reachable from ``seed_face`` across shared edges,
    moving only through faces where ``allowed`` is True."""
    visited = np.zeros(len(allowed), dtype=bool)
    if not allowed[seed_face]:
        return visited
    visited[seed_face] = True
    live = pairs[allowed[pairs[:, 0]] & allowed[pairs[:, 1]]]
    while True:
        left = visited[live[:, 0]]
        right = visited[live[:, 1]]
        frontier = left != right
        if not frontier.any():
            break
        touched = live[frontier]
        visited[touched[:, 0]] = True
        visited[touched[:, 1]] = True
        live = live[~(left & right) & ~frontier]
    return visited


def _kept_neighbor_counts(pairs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(keep), dtype=np.int64)
    np.add.at(counts, pairs[:, 0], keep[pairs[:, 1]].astype(np.int64))
    np.add.at(counts, pairs[:, 1], keep[pairs[:, 0]].astype(np.int64))
    return counts


def _pinch_vertices(faces_kept: np.ndarray, n_vertices: int) -> np.ndarray:
    """Vertices where the kept face set's boundary passes more than once.

    A clean patch boundary crosses each of its vertices exactly twice (one
    incoming and one outgoing edge). A vertex incident to more than two
    boundary edges is a bowtie/pinch: two patch lobes touching at that vertex
    only, which would make the stitched side wall traverse the same
    inner-outer edge twice.
    """
    if len(faces_kept) == 0:
        return np.empty(0, dtype=np.int64)
    edges = np.sort(faces_kept[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    incidence = np.bincount(boundary.reshape(-1), minlength=n_vertices)
    return np.flatnonzero(incidence > 2)


def _tidy_patch_faces(
    mesh: trimesh.Trimesh,
    pairs: np.ndarray,
    keep: np.ndarray,
    barrier: np.ndarray,
    seed_face: int,
) -> np.ndarray:
    """Clean a ragged patch face set so its boundary is stitchable.

    Whole-face selection along a noisy curve leaves three kinds of debris,
    all confined to the barrier band around the outline:

    - micro-holes and concave notches (a face surrounded by kept faces that
      the inside-side test misclassified) — filled: a barrier face with two
      or more kept edge-neighbors is added;
    - spurs (a kept barrier face hanging on by a single edge) — pruned;
    - bowtie/pinch boundary vertices (two lobes touching at one vertex) —
      resolved by keeping only the largest edge-connected fan of kept faces
      around the vertex (deterministic tie-break by lowest face index).

    Fill and prune only touch barrier faces, so the pass cannot creep beyond
    the band or eat the flooded core; a final flood keeps the seed-connected
    component. Bounded iteration; if a pathological set is still pinched
    afterwards, the repair/manifold verdict downstream is the backstop.
    """
    keep = keep.copy()
    for _ in range(8):
        changed = False
        for _ in range(64):  # fill notches / micro-holes inside the band
            add = barrier & ~keep & (_kept_neighbor_counts(pairs, keep) >= 2)
            if not add.any():
                break
            keep |= add
            changed = True
        for _ in range(64):  # prune spur faces hanging on one edge
            drop = keep & barrier & (_kept_neighbor_counts(pairs, keep) <= 1)
            drop[seed_face] = False
            if not drop.any():
                break
            keep &= ~drop
            changed = True

        pinched = _pinch_vertices(mesh.faces[keep], len(mesh.vertices))
        if len(pinched) == 0:
            if not changed:
                break
            continue
        for vertex in pinched:
            incident = [
                int(f) for f in np.flatnonzero(keep & (mesh.faces == vertex).any(axis=1))
            ]
            # Union-find the incident faces into fans: two faces belong to
            # the same fan when they share an edge (vertex, w).
            parent = {f: f for f in incident}

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            edge_users: Dict[int, List[int]] = {}
            for face_index in incident:
                for w in mesh.faces[face_index]:
                    if int(w) != int(vertex):
                        edge_users.setdefault(int(w), []).append(face_index)
            for users in edge_users.values():
                for other in users[1:]:
                    parent[_find(other)] = _find(users[0])
            fans: Dict[int, List[int]] = {}
            for face_index in incident:
                fans.setdefault(_find(face_index), []).append(face_index)
            ordered = sorted(fans.values(), key=lambda fan: (-len(fan), min(fan)))
            for fan in ordered[1:]:
                keep[fan] = False
    return _flood(pairs, keep, seed_face)


def _curve_side(
    dense_curve: np.ndarray, points: np.ndarray, surface_normals: np.ndarray
) -> np.ndarray:
    """Signed side of each point relative to a dense closed on-surface curve.

    For each point, take the nearest dense curve sample ``q_i``, the local
    curve tangent ``q_{i+1} - q_{i-1}`` (cyclic), and return
    ``cross(tangent, point - q_i) . surface_normal``. Points on the same side
    of the curve get the same sign; the absolute sign depends on the drawn
    winding direction, so callers calibrate it against a known-interior point.
    """
    diff = points[:, None, :] - dense_curve[None, :, :]
    nearest = np.einsum("psi,psi->ps", diff, diff).argmin(axis=1)
    n_samples = len(dense_curve)
    tangents = (
        dense_curve[(nearest + 1) % n_samples] - dense_curve[(nearest - 1) % n_samples]
    )
    return np.einsum(
        "pi,pi->p",
        np.cross(tangents, points - dense_curve[nearest]),
        surface_normals,
    )


def extract_patch(mesh: trimesh.Trimesh, closed_curve_points) -> trimesh.Trimesh:
    """Sub-surface of ``mesh`` enclosed by a drawn closed curve.

    Algorithm (flood fill behind a barrier band):

    1. Project the closed curve onto the surface densely
       (:func:`project_curve_to_surface`, sample spacing well below the
       barrier tolerance so the band has no gaps).
    2. BARRIER faces: every face whose minimum distance to any dense curve
       sample is below a tolerance derived from the mesh's local edge length
       (``0.5 * median edge length`` — every face the curve actually crosses
       is at distance ~0 and always included; the factor only controls the
       sideways bleed of the band).
    3. Interior seed: the dense samples' centroid snapped to the surface;
       the seed face is the nearest non-barrier face to that point.
    4. Flood fill from the seed face across shared edges (adjacency built
       locally from sorted edge pairs — no scipy/networkx), never entering
       barrier faces.
    5. The patch reaches back out to the drawn outline: barrier faces whose
       centroid lies on the INSIDE of the curve are added. The side of a
       point is the sign of ``cross(curve tangent, point - nearest curve
       sample) . surface normal``, with the sign calibrated on the interior
       seed point, so the drawn winding direction does not matter. (If the
       calibration is degenerate the fallback is the barrier faces
       edge-adjacent to the flooded set.)
    6. The band is tidied (micro-holes and notches filled, spur faces
       pruned, bowtie vertices split — see :func:`_tidy_patch_faces`) and
       only the connected component containing the seed face is kept,
       discarding inside-side barrier faces picked up elsewhere (e.g. on the
       far side of a thin bone).

    Returns a new ``trimesh.Trimesh`` (``process=False``, unreferenced
    vertices dropped). Raises ``ValueError`` when the curve has fewer than 3
    points, or when the flood fill escapes (fills more than 60% of the mesh
    faces), which means the projected curve did not close into a barrier —
    e.g. a degenerate or self-crossing outline, or one smaller than the local
    triangle size.

    Honest limitation: the patch boundary is ragged at mesh-triangle
    resolution — faces are kept or dropped whole, so the outline is honored
    to within about one triangle, not cut exactly along the curve. That is
    acceptable at segmentation resolution (triangles far smaller than the
    footprint); the upgrade path is exact edge splitting along the projected
    curve.
    """
    pts = np.asarray(closed_curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError(
            "closed_curve_points must be an (N, 3) array with N >= 3 to "
            "enclose a patch"
        )
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    tolerance = _BARRIER_EDGE_FACTOR * float(np.median(mesh.edges_unique_length))

    # Densify so consecutive samples are spaced well below the tolerance —
    # otherwise the barrier band could have gaps between samples.
    seg_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    max_seg = float(seg_lengths.max()) if len(seg_lengths) else 0.0
    samples_per_segment = int(min(400, max(10, np.ceil(max_seg / (0.4 * tolerance)))))
    dense = project_curve_to_surface(mesh, pts, samples_per_segment)

    barrier = _barrier_faces(mesh, dense, tolerance)
    if barrier.all():
        raise ValueError(
            "every face lies within the barrier tolerance of the curve; the "
            "curve is too large or the mesh too coarse to enclose a patch"
        )

    # Interior seed point / face. A single point against all faces is cheap,
    # so the seed face uses the exact distance with no prefilter.
    seed_point = snap_points_to_surface(mesh, dense.mean(axis=0))[0]
    a, b, c, _, _ = _face_geometry(mesh)
    _, seed_dist2 = _paired_closest(
        np.broadcast_to(seed_point, a.shape).copy(), a, b, c
    )
    seed_dist2 = seed_dist2.copy()
    seed_dist2[barrier] = np.inf
    seed_face = int(seed_dist2.argmin())

    # Flood fill across shared edges, never entering barrier faces.
    pairs = _face_adjacency_pairs(mesh)
    flooded = _flood(pairs, ~barrier, seed_face)

    if flooded.sum() > _ESCAPE_FRACTION * len(mesh.faces):
        raise ValueError(
            "flood fill escaped the drawn outline (filled {} of {} faces, "
            "over the {:.0%} limit): the projected curve did not close into "
            "a barrier band on the surface — the outline is degenerate, "
            "self-crossing, or smaller than the local triangle size".format(
                int(flooded.sum()), len(mesh.faces), _ESCAPE_FRACTION
            )
        )

    # Reach back out to the outline: add the barrier faces whose centroid
    # lies on the inside of the curve. Whole faces are kept or dropped by
    # their centroid, so the boundary jitters about +-half a face around the
    # curve but is unbiased — the flooded set alone would stop a full
    # barrier-band-width short of the outline.
    keep = flooded.copy()
    barrier_idx = np.flatnonzero(barrier)
    centroids = np.asarray(mesh.triangles, dtype=np.float64).mean(axis=1)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    seed_side = _curve_side(
        dense, seed_point[None, :], face_normals[seed_face][None, :]
    )[0]
    if abs(seed_side) > 1e-9:
        sides = _curve_side(
            dense, centroids[barrier_idx], face_normals[barrier_idx]
        )
        keep[barrier_idx[sides * seed_side > 0.0]] = True
    else:  # degenerate calibration: fall back to the rim adjacent to flooded
        keep[pairs[barrier[pairs[:, 0]] & flooded[pairs[:, 1]], 0]] = True
        keep[pairs[barrier[pairs[:, 1]] & flooded[pairs[:, 0]], 1]] = True

    # Tidy the ragged band (fill micro-holes/notches, prune spurs, split
    # bowtie vertices) and keep only the seed-connected component —
    # inside-side barrier faces picked up away from the footprint (thin
    # bone, folds) are not part of the patch.
    keep = _tidy_patch_faces(mesh, pairs, keep, barrier, seed_face)

    faces = mesh.faces[keep]
    referenced, reindexed = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[referenced],
        faces=reindexed.reshape(-1, 3),
        process=False,
    )


# ---------------------------------------------------------------------------
# Patch -> conforming molded-base solid
# ---------------------------------------------------------------------------

def _smoothed_normals(mesh: trimesh.Trimesh, rounds: int) -> np.ndarray:
    """Vertex normals, Laplacian-smoothed over the 1-ring.

    Same approach as layer 2's surface offset (offset.py), duplicated locally
    rather than importing a private helper: each round replaces a normal with
    the renormalized average of itself and its edge-connected neighbors. The
    base vertex normals are area-weighted face-normal sums computed here in
    pure numpy (``mesh.vertex_normals`` tries a scipy sparse path first and
    logs a noisy fallback without it).
    """
    weighted = np.asarray(mesh.face_normals, dtype=np.float64) * np.asarray(
        mesh.area_faces, dtype=np.float64
    )[:, None]
    normals = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    np.add.at(normals, mesh.faces.reshape(-1), np.repeat(weighted, 3, axis=0))
    lengths0 = np.linalg.norm(normals, axis=1)
    ok0 = lengths0 > 1e-12
    normals[ok0] /= lengths0[ok0, None]
    normals[~ok0] = (0.0, 0.0, 1.0)
    edges = mesh.edges_unique
    for _ in range(max(0, rounds)):
        accum = normals.copy()
        np.add.at(accum, edges[:, 0], normals[edges[:, 1]])
        np.add.at(accum, edges[:, 1], normals[edges[:, 0]])
        lengths = np.linalg.norm(accum, axis=1)
        ok = lengths > 1e-12
        accum[ok] /= lengths[ok, None]
        accum[~ok] = normals[~ok]
        normals = accum
    return normals


def _boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """Ordered boundary-vertex cycles (edges appearing in exactly one face).

    Same idea as layer 2's repair pass, implemented locally: boundary edges
    are directed edges whose undirected form occurs once; they are walked into
    cycles with deterministic (sorted) choices. Consecutive loop entries are
    directed boundary edges in face-winding order, which the wall stitching
    relies on for outward orientation. Open chains are ignored.
    """
    edges = mesh.edges
    undirected = np.sort(edges, axis=1)
    groups = trimesh.grouping.group_rows(undirected, require_count=1)
    boundary = edges[groups]

    successors: Dict[int, List[int]] = {}
    for start_v, end_v in boundary:
        successors.setdefault(int(start_v), []).append(int(end_v))

    loops: List[List[int]] = []
    used = set()
    for start in sorted(successors):
        for first in sorted(successors[start]):
            if (start, first) in used:
                continue
            loop = [start]
            a, b = start, first
            closed = False
            for _ in range(len(boundary) + 1):
                used.add((a, b))
                loop.append(b)
                if b == start:
                    closed = True
                    break
                candidates = sorted(
                    c for c in successors.get(b, []) if (b, c) not in used
                )
                if not candidates:
                    break
                a, b = b, candidates[0]
            if closed and len(loop) >= 4:
                loops.append(loop[:-1])
    return loops


def conforming_solid_from_patch(
    patch: trimesh.Trimesh, thickness: float, clearance: float = 0.0
) -> trimesh.Trimesh:
    """Turn an open surface patch into a watertight molded-base solid.

    Construction:

    - inner surface: the patch vertices displaced by ``clearance`` along
      smoothed outward vertex normals (the bone-fit clearance — printer
      dependent snug vs slip fit, architecture 6.1), faces inverted so the
      underside — the negative of the bone — points back at the bone;
    - outer surface: the same vertices displaced by ``clearance + thickness``,
      original winding kept;
    - side walls: each boundary loop stitched between inner and outer with a
      quad strip (two triangles per boundary edge), wound outward.

    The assembled solid is then passed through layer 2's mandatory
    :func:`~femguide_core.geometry.repair.repair` pass and must come out as a
    manifold solid; otherwise ``ValueError`` is raised with the repair report
    in the message (typical cause: a patch so ragged or so thick relative to
    local curvature that the offset self-intersects — see offset.py's
    limitation note, which applies to this vertex-normal displacement too).

    ``thickness`` must be > 0; ``clearance`` may be 0 (or negative for an
    interference fit, at the caller's risk).
    """
    if thickness <= 0.0:
        raise ValueError("thickness must be > 0 (got {})".format(thickness))
    if patch is None or len(patch.faces) == 0:
        raise ValueError("patch has no faces")

    normals = _smoothed_normals(patch, _SMOOTHING_ROUNDS)
    vertices = np.asarray(patch.vertices, dtype=np.float64)
    inner = vertices + float(clearance) * normals
    outer = vertices + float(clearance + thickness) * normals
    n_vertices = len(vertices)

    faces_inner = patch.faces[:, ::-1]  # inverted: underside faces the bone
    faces_outer = patch.faces + n_vertices

    wall_faces: List[List[int]] = []
    for loop in _boundary_loops(patch):
        for i, va in enumerate(loop):
            vb = loop[(i + 1) % len(loop)]
            # (a -> b) is a directed boundary edge in face-winding order, so
            # the surface interior lies to its left; these two triangles face
            # away from the patch (outward).
            wall_faces.append([va + n_vertices, vb, vb + n_vertices])
            wall_faces.append([va + n_vertices, va, vb])
    if not wall_faces and not patch.is_watertight:
        raise ValueError(
            "patch has open boundary edges but no closed boundary loop could "
            "be walked; cannot stitch a side wall"
        )

    blocks = [faces_inner, faces_outer]
    if wall_faces:
        blocks.append(np.asarray(wall_faces, dtype=patch.faces.dtype))
    solid = trimesh.Trimesh(
        vertices=np.vstack([inner, outer]),
        faces=np.vstack(blocks),
        process=False,
    )

    repaired, report = repair(solid)
    if not report.is_manifold_solid:
        raise ValueError(
            "conforming_solid_from_patch could not produce a manifold solid "
            "(thickness={}, clearance={}); the offset likely self-intersects "
            "for this patch/thickness (vertex-normal displacement "
            "limitation).\n{}".format(thickness, clearance, report)
        )
    return repaired


def molded_base(
    mesh: trimesh.Trimesh,
    curve_points,
    thickness: float,
    clearance: float = 0.0,
) -> trimesh.Trimesh:
    """Full pipeline: drawn closed footprint -> conforming molded base solid.

    ``project_curve_to_surface`` -> ``extract_patch`` ->
    ``conforming_solid_from_patch``. This is the shared FemGuide / Glenoid
    Guide base generator (architecture 6.1 item 2 and 6.2): the surgeon's
    closed outline drawn on the bone becomes a watertight solid whose
    underside is a negative of the bone surface (displaced by ``clearance``)
    so the guide seats uniquely. Pin holes are layer 2's job: subtract
    cylinders from the result with ``difference()``.

    ``mesh`` must be clean enough to snap against; segmentation output should
    go through ``repair()`` first (the dirty-input path is repair -> this).
    Raises ``ValueError`` for a curve with fewer than 3 points, an outline
    that does not enclose a patch, or a thickness <= 0.
    """
    on_surface = project_curve_to_surface(mesh, curve_points)
    patch = extract_patch(mesh, on_surface)
    return conforming_solid_from_patch(patch, thickness, clearance)
