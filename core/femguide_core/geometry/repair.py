"""Mandatory mesh-repair pass for noisy segmentation output (layer 2).

Segmentation output is assumed dirty: duplicated vertices, degenerate and
duplicate faces, punched holes, small disconnected debris. Per frozen decision
D3 every boolean runs on manifold3d, which rejects anything that is not an
oriented 2-manifold — so :func:`repair` is the mandatory pass that turns
segmentation-grade meshes into manifold solids before any boolean.

Everything here is numpy + trimesh + manifold3d only. The pipeline avoids
trimesh helpers that require optional graph engines (scipy / networkx), so it
works on the frozen dependency set: connected components, hole capping and
per-component inversion fixes are implemented locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from manifold3d import Error, Manifold, Mesh64


@dataclass
class RepairReport:
    """What :func:`repair` did to a mesh, and what came out the other end.

    All ``output_*`` / ``is_*`` fields describe the mesh *after* repair.
    """

    input_vertices: int = 0
    input_faces: int = 0
    output_vertices: int = 0
    output_faces: int = 0
    holes_filled: int = 0
    components_removed: int = 0
    decimated: bool = False
    is_watertight: bool = False
    is_manifold_solid: bool = False
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "manifold solid" if self.is_manifold_solid else "NOT a manifold solid"
        lines = [
            "RepairReport: {} -> {} vertices, {} -> {} faces".format(
                self.input_vertices,
                self.output_vertices,
                self.input_faces,
                self.output_faces,
            ),
            "  holes filled: {}, debris components removed: {}, decimated: {}".format(
                self.holes_filled, self.components_removed, self.decimated
            ),
            "  result: watertight={}, {}".format(self.is_watertight, status),
        ]
        for note in self.notes:
            lines.append("  note: " + note)
        return "\n".join(lines)


def is_manifold_solid(mesh: trimesh.Trimesh) -> bool:
    """True iff manifold3d accepts ``mesh`` as a valid, non-empty manifold.

    Builds a ``manifold3d.Manifold`` from the raw vertex/face arrays and checks
    that construction succeeded (``status() == Error.NoError``) and the result
    is not empty. Any exception during conversion counts as "not a manifold
    solid". This is the exact acceptance test the boolean layer applies, so a
    mesh passing here is safe to hand to :mod:`femguide_core.geometry.booleans`.
    """
    try:
        if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            return False
        # Fresh C-ordered copies: nanobind rejects some numpy views/subclasses.
        vertices = np.array(mesh.vertices, dtype=np.float64, order="C", copy=True)
        faces = np.array(mesh.faces, dtype=np.uint64, order="C", copy=True)
        solid = Manifold(Mesh64(vert_properties=vertices, tri_verts=faces))
        return solid.status() == Error.NoError and not solid.is_empty()
    except Exception:
        return False


def _face_components(faces: np.ndarray, n_vertices: int) -> Tuple[np.ndarray, int]:
    """Label each face with a vertex-connectivity component id (union-find).

    Returns ``(labels, n_components)`` where ``labels[i]`` is the component of
    face ``i``. Pure numpy/python; no scipy or networkx required.
    """
    parent = np.arange(n_vertices)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for f in faces:
        a = find(int(f[0]))
        for v in f[1:]:
            b = find(int(v))
            if a != b:
                parent[b] = a
    roots = np.array([find(int(f[0])) for f in faces])
    uniq, labels = np.unique(roots, return_inverse=True)
    return labels, len(uniq)


def _boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """Extract closed boundary loops as ordered vertex-index cycles.

    Boundary edges are directed edges whose undirected form appears in exactly
    one face. Loops are walked deterministically (sorted choices) so repair is
    reproducible. Open chains (which cannot be capped) are ignored.
    """
    edges = mesh.edges  # directed, one triple of edges per face
    undirected = np.sort(edges, axis=1)
    groups = trimesh.grouping.group_rows(undirected, require_count=1)
    boundary = edges[groups]

    successors: Dict[int, List[int]] = {}
    for a, b in boundary:
        successors.setdefault(int(a), []).append(int(b))

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
            if closed and len(loop) >= 4:  # at least a triangle plus repeat
                loops.append(loop[:-1])
    return loops


def _cap_boundary_loops(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """Close every boundary loop with a triangle or a centroid fan.

    Triangular holes get the single missing triangle; larger loops get a new
    centroid vertex and a fan. Cap triangles are wound opposite to the
    boundary's directed edges so they close the surface with consistent
    orientation. Returns ``(capped_mesh, n_loops_capped)``.
    """
    loops = _boundary_loops(mesh)
    if not loops:
        return mesh, 0
    vertices = mesh.vertices.copy()
    face_blocks = [mesh.faces]
    for loop in loops:
        if len(loop) == 3:
            face_blocks.append(np.array([[loop[2], loop[1], loop[0]]]))
        else:
            centroid_index = len(vertices)
            centroid = vertices[loop].mean(axis=0)
            vertices = np.vstack([vertices, centroid[None, :]])
            fan = [
                [loop[(i + 1) % len(loop)], loop[i], centroid_index]
                for i in range(len(loop))
            ]
            face_blocks.append(np.array(fan))
    capped = trimesh.Trimesh(
        vertices=vertices, faces=np.vstack(face_blocks), process=False
    )
    return capped, len(loops)


def _fix_inversion(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Flip any connected component whose signed volume is negative.

    Assumes winding is already consistent within each component (run
    ``trimesh.repair.fix_winding`` first); this makes every closed component
    point outward so the whole mesh is a positive-volume solid.
    """
    if len(mesh.faces) == 0:
        return mesh
    labels, n_components = _face_components(mesh.faces, len(mesh.vertices))
    triangles = mesh.triangles
    signed = (
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
        / 6.0
    )
    faces = mesh.faces.copy()
    for component in range(n_components):
        mask = labels == component
        if signed[mask].sum() < 0.0:
            faces[mask] = faces[mask][:, ::-1]
    return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=faces, process=False)


def _basic_cleanup(mesh: trimesh.Trimesh) -> None:
    """In-place: merge duplicate vertices, drop degenerate/duplicate faces and
    unreferenced vertices."""
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()


# Debris components smaller than this fraction of the largest component's face
# count (and smaller than _DEBRIS_MIN_FACES) are discarded by repair().
_DEBRIS_FRACTION = 0.1
_DEBRIS_MIN_FACES = 8


def repair(
    mesh: trimesh.Trimesh,
    keep_largest_component: bool = True,
    target_faces: Optional[int] = None,
) -> Tuple[trimesh.Trimesh, RepairReport]:
    """The mandatory pre-boolean repair pass for noisy segmentation output.

    Pipeline (in order):

    1. Copy the input (never modified in place).
    2. Merge duplicate vertices; remove degenerate and duplicate faces and
       unreferenced vertices.
    3. If ``keep_largest_component``: discard tiny disconnected debris —
       components whose face count is below 10% of the largest component's
       (and below 8 faces are always debris). Comparably sized components are
       kept, so a bone made of two touching closed shells survives while
       far-away segmentation specks do not.
    4. Make triangle winding consistent (``trimesh.repair.fix_winding``).
    5. Fill holes: every closed boundary loop is capped — triangular holes get
       the missing triangle, larger loops a centroid fan — then the cleanup of
       step 2 runs again and winding is re-fixed.
    6. Optional decimation to about ``target_faces`` using trimesh's quadric
       decimation when its backend (``fast_simplification``) is installed;
       otherwise skipped gracefully and noted in the report.
    7. Final orientation pass: any closed component with negative signed
       volume is flipped outward.

    Returns ``(repaired_mesh, RepairReport)``. If the result still is not a
    manifold solid after all steps, it is returned anyway with the report
    saying so — callers decide what to do.
    """
    report = RepairReport(
        input_vertices=len(mesh.vertices), input_faces=len(mesh.faces)
    )
    result = mesh.copy()

    _basic_cleanup(result)

    if keep_largest_component and len(result.faces) > 0:
        labels, n_components = _face_components(result.faces, len(result.vertices))
        if n_components > 1:
            counts = np.bincount(labels)
            threshold = max(_DEBRIS_FRACTION * counts.max(), _DEBRIS_MIN_FACES)
            keep = counts >= threshold
            removed = int(np.count_nonzero(~keep))
            if removed:
                result.update_faces(keep[labels])
                result.remove_unreferenced_vertices()
                report.components_removed = removed

    trimesh.repair.fix_winding(result)

    if not result.is_watertight:
        result, capped = _cap_boundary_loops(result)
        report.holes_filled = capped
        _basic_cleanup(result)
        trimesh.repair.fix_winding(result)
        if not result.is_watertight:
            report.notes.append(
                "mesh is still not watertight after boundary-loop capping"
            )

    if target_faces is not None and len(result.faces) > target_faces:
        try:
            decimated = result.simplify_quadric_decimation(face_count=int(target_faces))
            trimesh.repair.fix_winding(decimated)
            if len(decimated.faces) > 0 and decimated.is_watertight:
                result = decimated
                report.decimated = True
            else:
                report.notes.append(
                    "decimation produced a non-watertight mesh; kept the "
                    "undecimated result"
                )
        except BaseException as exc:  # missing backend raises ImportError
            report.notes.append(
                "decimation skipped ({}: {})".format(type(exc).__name__, exc)
            )

    result = _fix_inversion(result)

    report.output_vertices = len(result.vertices)
    report.output_faces = len(result.faces)
    report.is_watertight = bool(result.is_watertight)
    report.is_manifold_solid = is_manifold_solid(result)
    if not report.is_manifold_solid:
        report.notes.append(
            "result is still not a manifold solid; booleans will reject it"
        )
    return result, report
