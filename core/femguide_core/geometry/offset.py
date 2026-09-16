"""Surface offset (shelling) of a closed manifold solid (layer 2).

Implementation: displace every vertex along a smoothed vertex normal by a
signed distance. This is the simple, fast approach and its limitation is
stated honestly in :func:`offset_solid`'s docstring — an SDF / voxel-based
offset is the upgrade path if guide flanges ever need large offsets.
"""

from __future__ import annotations

import numpy as np
import trimesh

from femguide_core.geometry.booleans import NonManifoldError, to_manifold
from femguide_core.geometry.repair import is_manifold_solid

_SMOOTHING_ROUNDS = 2


def _smoothed_vertex_normals(mesh: trimesh.Trimesh, rounds: int) -> np.ndarray:
    """Angle-weighted vertex normals, Laplacian-smoothed over the 1-ring.

    Each round replaces a vertex normal with the renormalized average of
    itself and its edge-connected neighbors, which damps the spiky normals a
    noisy or coarsely triangulated surface produces.
    """
    normals = np.array(mesh.vertex_normals, dtype=np.float64)
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


def offset_solid(mesh: trimesh.Trimesh, distance: float) -> trimesh.Trimesh:
    """Offset the surface of a closed manifold solid by ``distance``.

    Positive ``distance`` grows the solid outward, negative shrinks it. The
    input is validated via ``to_manifold`` and a
    :class:`~femguide_core.geometry.booleans.NonManifoldError` is raised if it
    is not a manifold solid (run ``repair()`` first on segmentation output).

    Method and honest limitation: each vertex is displaced along a smoothed
    vertex normal. That is valid for modest distances relative to the local
    curvature radius; large offsets (or offsets across thin/sharp features)
    can self-intersect or invert local geometry, and a negative distance
    larger than the local thickness will collapse the solid. If those regimes
    are ever needed, the upgrade path is an SDF / voxel-based offset
    (level-set of the distance field), not more smoothing here.

    The displaced mesh gets a light cleanup (merge vertices, drop degenerate
    faces, fix winding) and its manifoldness is verified; a result manifold3d
    would reject also raises :class:`NonManifoldError`.
    """
    to_manifold(mesh, "mesh")  # validation only; raises NonManifoldError

    normals = _smoothed_vertex_normals(mesh, _SMOOTHING_ROUNDS)
    vertices = np.array(mesh.vertices, dtype=np.float64) + float(distance) * normals
    result = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)

    # Light cleanup: the displacement can collapse near-coincident vertices.
    result.merge_vertices()
    result.update_faces(result.nondegenerate_faces())
    result.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(result)

    if not is_manifold_solid(result):
        raise NonManifoldError(
            "offset_solid produced a non-manifold result for distance {}; "
            "the offset is too large for the local curvature of the input "
            "(vertex-normal displacement limitation — see docstring)".format(
                distance
            )
        )
    return result
