"""Shared test fixtures.

The "dirty" fixtures simulate segmentation output: duplicated vertices,
degenerate and duplicate faces, punched holes, small disconnected debris.
Real segmented femur/scapula datasets are an open item (docs/ARCHITECTURE.md
section 12); until they land, these synthetic meshes are the layer-2 proving
ground. Everything is seeded so failures reproduce.
"""

import numpy as np
import pytest
import trimesh


def _make_dirty(mesh: trimesh.Trimesh, rng: np.random.Generator) -> trimesh.Trimesh:
    """Corrupt a clean watertight mesh the way segmentation output tends to be."""
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # Surface noise, ~2% of the bounding-box diagonal.
    scale = float(np.linalg.norm(mesh.extents)) * 0.02
    vertices += rng.normal(scale=scale * 0.15, size=vertices.shape)

    # Punch a few holes by deleting random faces.
    n_holes = max(3, len(faces) // 200)
    delete = rng.choice(len(faces), size=n_holes, replace=False)
    keep = np.ones(len(faces), dtype=bool)
    keep[delete] = False
    faces = faces[keep]

    # Duplicate some faces (non-manifold overlap) and add degenerate faces.
    dup = faces[rng.choice(len(faces), size=5, replace=False)]
    degen_v = rng.choice(len(vertices), size=5)
    degenerate = np.column_stack([degen_v, degen_v, degen_v])
    faces = np.vstack([faces, dup, degenerate])

    # Un-merge some vertices: split shared vertices into duplicates.
    split_faces = rng.choice(len(faces), size=10, replace=False)
    for fi in split_faces:
        v = faces[fi][0]
        vertices = np.vstack([vertices, vertices[v]])
        faces[fi][0] = len(vertices) - 1

    # Disconnected debris: a tiny far-away tetrahedron.
    base = mesh.bounds[1] + mesh.extents
    tet_v = base + np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    tet_f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]) + len(vertices)
    vertices = np.vstack([vertices, tet_v])
    faces = np.vstack([faces, tet_f])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260916)


@pytest.fixture
def clean_sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=3, radius=20.0)


@pytest.fixture
def clean_femur_like() -> trimesh.Trimesh:
    """Watertight femur-ish stand-in: a long capsule shaft with a head bump."""
    shaft = trimesh.creation.capsule(radius=14.0, height=120.0, count=[48, 48])
    head = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    head.apply_translation([12.0, 0.0, 70.0])
    combined = trimesh.util.concatenate([shaft, head])
    combined.merge_vertices()
    return combined


@pytest.fixture
def dirty_sphere(clean_sphere, rng) -> trimesh.Trimesh:
    return _make_dirty(clean_sphere, rng)


@pytest.fixture
def dirty_femur_like(clean_femur_like, rng) -> trimesh.Trimesh:
    return _make_dirty(clean_femur_like, rng)
