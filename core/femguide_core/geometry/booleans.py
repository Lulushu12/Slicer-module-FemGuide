"""Mesh booleans on manifold3d (layer 2, frozen decision D3).

All boolean operations run through ``manifold3d.Manifold`` — never VTK
booleans on segmented anatomy. Inputs are validated strictly: anything that
manifold3d rejects as a non-manifold or empty mesh raises
:class:`NonManifoldError`. There is deliberately **no silent auto-repair**
here; per the architecture, the repair pass
(:func:`femguide_core.geometry.repair.repair`) is explicit at call sites so
that a mesh being mutated before a boolean is always visible in the plan
pipeline, never a hidden side effect.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import trimesh
from manifold3d import Error, Manifold, Mesh64, OpType


class NonManifoldError(ValueError):
    """Raised when a mesh is not a valid manifold solid for boolean input."""


def _describe(status: Error) -> str:
    return str(status).replace("Error.", "")


def to_manifold(mesh: trimesh.Trimesh, name: str = "mesh") -> Manifold:
    """Convert a trimesh into a ``manifold3d.Manifold``, validating strictly.

    Raises :class:`NonManifoldError` if the mesh is empty, is not an oriented
    2-manifold, or fails conversion for any other reason. No repair is
    attempted here — run ``repair()`` from
    :mod:`femguide_core.geometry.repair` on segmentation output first.

    ``name`` labels the offending input in the error message (e.g. ``"a"``,
    ``"cut tool"``) so multi-input boolean failures are attributable.
    """
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        raise NonManifoldError(
            "input '{}' is empty (no faces); run repair() from "
            "femguide_core.geometry.repair on it first".format(name)
        )
    try:
        # Fresh C-ordered copies: nanobind rejects some numpy views/subclasses.
        vertices = np.array(mesh.vertices, dtype=np.float64, order="C", copy=True)
        faces = np.array(mesh.faces, dtype=np.uint64, order="C", copy=True)
        solid = Manifold(Mesh64(vert_properties=vertices, tri_verts=faces))
    except Exception as exc:
        raise NonManifoldError(
            "input '{}' could not be converted to a manifold3d.Manifold "
            "({}); run repair() from femguide_core.geometry.repair on it "
            "first".format(name, exc)
        ) from exc
    status = solid.status()
    if status != Error.NoError or solid.is_empty():
        raise NonManifoldError(
            "input '{}' is not a valid manifold solid "
            "(manifold3d status: {}); run repair() from "
            "femguide_core.geometry.repair on it first".format(
                name, _describe(status)
            )
        )
    return solid


def from_manifold(solid: Manifold) -> trimesh.Trimesh:
    """Convert a ``manifold3d.Manifold`` back into a ``trimesh.Trimesh``.

    Uses the double-precision export (``to_mesh64``). An empty manifold (for
    example the intersection of non-overlapping solids) becomes an empty
    trimesh with zero faces.
    """
    mesh = solid.to_mesh64()
    vertices = np.array(mesh.vert_properties, dtype=np.float64, copy=True)
    if vertices.ndim == 2 and vertices.shape[1] > 3:
        vertices = np.ascontiguousarray(vertices[:, :3])
    faces = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    return trimesh.Trimesh(
        vertices=vertices.reshape(-1, 3), faces=faces.reshape(-1, 3), process=False
    )


def _as_sequence(
    meshes: Union[trimesh.Trimesh, Sequence[trimesh.Trimesh]]
) -> Sequence[trimesh.Trimesh]:
    if isinstance(meshes, trimesh.Trimesh):
        return [meshes]
    return list(meshes)


def union(
    meshes: Union[trimesh.Trimesh, Sequence[trimesh.Trimesh]]
) -> trimesh.Trimesh:
    """N-ary boolean union via manifold3d's batch boolean.

    Accepts a single mesh (returned round-tripped through manifold3d) or a
    sequence of meshes. Every input is validated with :func:`to_manifold` and
    a :class:`NonManifoldError` is raised for invalid ones — there is no
    silent auto-repair; call ``repair()`` explicitly on dirty input first.
    """
    items = _as_sequence(meshes)
    if not items:
        raise ValueError("union() needs at least one mesh")
    solids = [to_manifold(m, "meshes[{}]".format(i)) for i, m in enumerate(items)]
    return from_manifold(Manifold.batch_boolean(solids, OpType.Add))


def difference(
    a: trimesh.Trimesh,
    b: Union[trimesh.Trimesh, Sequence[trimesh.Trimesh]],
) -> trimesh.Trimesh:
    """Boolean difference ``a - b`` via manifold3d.

    ``b`` may be a single mesh or a sequence of meshes; a sequence is
    subtracted in one batch operation (e.g. subtracting several pin cylinders
    from a molded base at once). Inputs are validated with
    :func:`to_manifold`; dirty segmentation meshes must go through
    ``repair()`` explicitly first — no silent auto-repair happens here.
    """
    tools = _as_sequence(b)
    if not tools:
        raise ValueError("difference() needs at least one mesh to subtract")
    solids = [to_manifold(a, "a")]
    solids.extend(to_manifold(m, "b[{}]".format(i)) for i, m in enumerate(tools))
    return from_manifold(Manifold.batch_boolean(solids, OpType.Subtract))


def intersection(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Boolean intersection of two solids via manifold3d.

    Inputs are validated with :func:`to_manifold` and invalid ones raise
    :class:`NonManifoldError` — no silent auto-repair; the repair pass is
    explicit at call sites. Non-overlapping inputs yield an empty mesh.
    """
    solids = [to_manifold(a, "a"), to_manifold(b, "b")]
    return from_manifold(Manifold.batch_boolean(solids, OpType.Intersect))
