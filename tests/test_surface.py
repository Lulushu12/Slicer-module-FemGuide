"""Layer 3 (anatomy operations) tests: the shared FemGuide / Glenoid Guide
"draw footprint on bone -> conforming base -> subtract pin cylinders"
pipeline (docs/ARCHITECTURE.md sections 4.1 and 6.1).

All deterministic: fixtures are seeded, curves are analytic.
"""

import numpy as np
import pytest
import trimesh

from femguide_core.geometry.surface import (
    snap_points_to_surface,
    project_curve_to_surface,
    extract_patch,
    conforming_solid_from_patch,
    molded_base,
)
from femguide_core.geometry.repair import is_manifold_solid, repair
from femguide_core.geometry.booleans import difference
from femguide_core.geometry.primitives import paired_cylinders

RADIUS = 20.0
# Max chord sag of the icosphere(subdivisions=3, radius=20) triangulation is
# ~0.06, so 0.3 is a comfortable on-surface tolerance.
CHORD_TOL = 0.3


def _n_components(mesh: trimesh.Trimesh) -> int:
    """Vertex-connectivity component count (pure python union-find — no
    scipy/networkx in the frozen dependency set, so no mesh.split here)."""
    parent = list(range(len(mesh.vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in mesh.faces:
        a = find(int(face[0]))
        for v in face[1:]:
            b = find(int(v))
            if a != b:
                parent[b] = a
    return len({find(int(face[0])) for face in mesh.faces})


def _latitude_circle(theta_deg: float, n_points: int = 24) -> np.ndarray:
    """The circle x^2 + y^2 = (R sin theta)^2 at z = R cos theta — exactly on
    the analytic sphere, approximately on its triangulation."""
    theta = np.radians(theta_deg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.column_stack(
        [
            RADIUS * np.sin(theta) * np.cos(phi),
            RADIUS * np.sin(theta) * np.sin(phi),
            np.full(n_points, RADIUS * np.cos(theta)),
        ]
    )


def _cap_area(theta_deg: float) -> float:
    """Analytic spherical cap area 2*pi*R*h with h = R(1 - cos theta)."""
    return 2.0 * np.pi * RADIUS * RADIUS * (1.0 - np.cos(np.radians(theta_deg)))


def _fine_femur(clean_femur_like: trimesh.Trimesh) -> trimesh.Trimesh:
    """The femur-like fixture with its shaft refined to segmentation-like
    resolution.

    trimesh's capsule builds the whole 120 mm cylindrical section as a single
    band of 120 mm-tall triangles — an artifact of the synthetic fixture, far
    coarser than any segmented bone. extract_patch honestly cannot enclose an
    outline smaller than the local triangles, so refine to a max edge of 4 mm
    (real segmentation meshes are already at ~sub-mm/voxel resolution).
    """
    vertices, faces = trimesh.remesh.subdivide_to_size(
        clean_femur_like.vertices, clean_femur_like.faces, max_edge=4.0
    )
    fine = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fine.merge_vertices()
    return fine


def _shaft_circle() -> np.ndarray:
    """A circle hovering beside the femur shaft (plane x = -30, tangent-ish to
    the r=14 shaft), projecting onto one side of it."""
    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return np.column_stack(
        [np.full(16, -30.0), 8.0 * np.cos(phi), 10.0 + 8.0 * np.sin(phi)]
    )


# ---------------------------------------------------------------------------
# snap_points_to_surface
# ---------------------------------------------------------------------------

def test_snap_outside_and_inside_points_land_on_sphere(clean_sphere):
    points = np.array(
        [
            [30.0, 0.0, 0.0],
            [0.0, -45.0, 10.0],
            [3.0, 4.0, 5.0],  # inside
            [0.0, 0.0, 0.5],  # near center
            [-25.0, 25.0, -25.0],
        ]
    )
    snapped = snap_points_to_surface(clean_sphere, points)
    assert snapped.shape == points.shape
    radii = np.linalg.norm(snapped, axis=1)
    assert np.all(np.abs(radii - RADIUS) < CHORD_TOL)


def test_snap_points_already_on_surface_stay_put(clean_sphere):
    on_surface = clean_sphere.vertices[[0, 17, 100, 333, 641]]
    snapped = snap_points_to_surface(clean_sphere, on_surface)
    assert np.linalg.norm(snapped - on_surface, axis=1).max() < 1e-9


# ---------------------------------------------------------------------------
# project_curve_to_surface
# ---------------------------------------------------------------------------

def test_project_curve_gives_ordered_dense_on_surface_loop(clean_sphere):
    n_points, samples = 12, 10
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    hovering = np.column_stack(
        [8.0 * np.cos(phi), 8.0 * np.sin(phi), np.full(n_points, 25.0)]
    )
    loop = project_curve_to_surface(clean_sphere, hovering, samples_per_segment=samples)

    assert loop.shape == (n_points * samples, 3)
    radii = np.linalg.norm(loop, axis=1)
    assert np.all(np.abs(radii - RADIUS) < CHORD_TOL)
    # Ordered closed loop, first point not repeated: consecutive gaps (with
    # wraparound) stay small versus the ~38 mm loop circumference.
    gaps = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
    assert gaps.max() < 1.5
    # Adjacent samples may legitimately snap to the same closest point on a
    # coarse triangulation, but the loop as a whole must actually go around.
    assert len(np.unique(np.round(loop, 6), axis=0)) > 0.8 * len(loop)


# ---------------------------------------------------------------------------
# extract_patch
# ---------------------------------------------------------------------------

def test_extract_patch_spherical_cap(clean_sphere):
    theta = 40.0
    patch = extract_patch(clean_sphere, _latitude_circle(theta))

    assert isinstance(patch, trimesh.Trimesh)
    assert len(patch.faces) > 0
    # Whole-face granularity: boundary is honored to ~half a triangle, area
    # to well within 15% of the analytic cap.
    assert abs(patch.area - _cap_area(theta)) < 0.15 * _cap_area(theta)
    # A 40-degree cap is a small fraction of the sphere.
    assert len(patch.faces) < 0.25 * len(clean_sphere.faces)
    assert _n_components(patch) == 1
    # No unreferenced vertices survive.
    assert len(np.setdiff1d(np.arange(len(patch.vertices)), patch.faces)) == 0
    # On-surface: every patch vertex is a sphere vertex (radius 20 exactly).
    assert np.allclose(np.linalg.norm(patch.vertices, axis=1), RADIUS, atol=1e-6)


def test_extract_patch_rejects_too_few_points(clean_sphere):
    with pytest.raises(ValueError):
        extract_patch(clean_sphere, np.array([[0.0, 0.0, 25.0], [1.0, 0.0, 25.0]]))


def test_extract_patch_escape_raises(clean_sphere):
    # A near-degenerate 3-point speck: its projected outline is far smaller
    # than the local triangles, so no closed barrier band forms and the flood
    # fill escapes — extract_patch must say so, deterministically.
    tiny = np.array([[0.1, 0.0, 25.0], [0.0, 0.1, 25.0], [-0.1, 0.0, 25.0]])
    with pytest.raises(ValueError, match="did not close"):
        extract_patch(clean_sphere, tiny)


# ---------------------------------------------------------------------------
# conforming_solid_from_patch
# ---------------------------------------------------------------------------

def test_conforming_solid_from_cap_patch(clean_sphere):
    theta, thickness, clearance = 40.0, 3.0, 0.5
    patch = extract_patch(clean_sphere, _latitude_circle(theta))
    solid = conforming_solid_from_patch(patch, thickness, clearance=clearance)

    assert solid.is_watertight
    assert is_manifold_solid(solid)
    # Volume ~ patch area x thickness, plus outward spherical spreading: the
    # mid-surface sits at radius ~22 so the exact factor is ~(22/20)^2 = 1.21.
    assert 0.75 * patch.area * thickness < solid.volume < 1.45 * patch.area * thickness
    # The whole solid lives in the shell [R + clearance, R + clearance +
    # thickness]; boundary-vertex normals tilt slightly, hence the margin.
    radii = np.linalg.norm(solid.vertices, axis=1)
    assert radii.min() > RADIUS + clearance - 0.25
    assert radii.max() < RADIUS + clearance + thickness + 0.25


def test_conforming_solid_requires_positive_thickness(clean_sphere):
    patch = extract_patch(clean_sphere, _latitude_circle(40.0))
    with pytest.raises(ValueError):
        conforming_solid_from_patch(patch, 0.0)
    with pytest.raises(ValueError):
        conforming_solid_from_patch(patch, -2.0, clearance=0.5)


# ---------------------------------------------------------------------------
# molded_base (full pipeline)
# ---------------------------------------------------------------------------

def test_molded_base_on_femur_shaft(clean_femur_like):
    femur = _fine_femur(clean_femur_like)
    base = molded_base(femur, _shaft_circle(), thickness=3.0, clearance=0.4)
    assert base.is_watertight
    assert is_manifold_solid(base)
    assert base.volume > 0.0
    # The base hugs the -x side of the shaft, clear of the bone axis.
    assert base.vertices[:, 0].max() < 0.0


def test_commercial_workflow_base_minus_pins(clean_sphere):
    """Layers 1+2+3 compose: molded base (layer 3) minus paired pin
    cylinders (layer 1) via manifold3d difference (layer 2)."""
    theta = 40.0
    base = molded_base(clean_sphere, _latitude_circle(theta), 3.0, clearance=0.5)
    assert base.is_watertight
    assert is_manifold_solid(base)

    # Two pins through the polar cap base: anchor above the pole, axes along
    # +x, one either side of the polyline plane — both pierce the shell,
    # which spans radii ~20.5..23.5 around the pole.
    pins = paired_cylinders(
        np.array([[-10.0, 0.0, 22.2], [10.0, 0.0, 22.2]]),
        np.array([0.0, 0.0, 1.0]),
        seg_index=0,
        frac_along=0.5,
        radius=1.2,
        height=60.0,
        offset_from_plane=0.9,
    )
    guide = difference(base, pins)

    assert guide.is_watertight
    assert is_manifold_solid(guide)
    assert 0.0 < guide.volume < base.volume


def test_molded_base_after_repair_of_dirty_segmentation(dirty_femur_like):
    """Segmentation-grade end-to-end: mandatory repair pass, then the base
    pipeline on the repaired mesh (footprint drawn on the femoral head, whose
    fixture resolution is segmentation-like)."""
    repaired, report = repair(dirty_femur_like)
    assert report.is_manifold_solid

    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    head_circle = np.column_stack(
        [12.0 + 7.0 * np.cos(phi), 7.0 * np.sin(phi), np.full(16, 100.0)]
    )
    base = molded_base(repaired, head_circle, thickness=3.0, clearance=0.4)
    assert is_manifold_solid(base)
    assert base.is_watertight
    assert base.volume > 0.0
