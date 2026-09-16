"""Layer-2 repair pass tests against the dirty segmentation-like fixtures."""

import numpy as np
import pytest
import trimesh

from femguide_core.geometry.booleans import to_manifold
from femguide_core.geometry.repair import RepairReport, is_manifold_solid, repair


def _body_count(mesh: trimesh.Trimesh) -> int:
    """Count connected solid bodies via manifold3d's decompose."""
    return len(to_manifold(mesh).decompose())


def _decimation_supported() -> bool:
    try:
        import fast_simplification  # noqa: F401

        return True
    except ImportError:
        return False


def test_is_manifold_solid_on_clean_and_dirty(clean_sphere, dirty_sphere):
    assert is_manifold_solid(clean_sphere)
    assert not is_manifold_solid(dirty_sphere)


def test_is_manifold_solid_rejects_empty():
    assert not is_manifold_solid(trimesh.Trimesh())


def test_repair_dirty_sphere(clean_sphere, dirty_sphere):
    repaired, report = repair(dirty_sphere)

    assert is_manifold_solid(repaired)
    assert repaired.is_watertight
    assert report.is_manifold_solid
    assert report.is_watertight

    # Debris tetrahedron removed: exactly one solid body remains.
    assert _body_count(repaired) == 1
    assert report.components_removed >= 1

    # Volume close to the clean source despite noise + hole capping.
    assert repaired.volume == pytest.approx(clean_sphere.volume, rel=0.15)


def test_repair_dirty_femur_like(clean_femur_like, dirty_femur_like):
    repaired, report = repair(dirty_femur_like)

    assert is_manifold_solid(repaired)
    assert repaired.is_watertight
    assert report.is_manifold_solid

    # The far-away debris tetrahedron is gone: every vertex sits within the
    # clean fixture's bounds plus a small noise margin, well short of where
    # the debris was planted (bounds[1] + extents).
    margin = 0.1 * np.linalg.norm(clean_femur_like.extents)
    assert np.all(repaired.vertices >= clean_femur_like.bounds[0] - margin)
    assert np.all(repaired.vertices <= clean_femur_like.bounds[1] + margin)
    assert report.components_removed >= 1

    assert repaired.volume == pytest.approx(clean_femur_like.volume, rel=0.15)


def test_repair_report_fields_sane(dirty_sphere):
    repaired, report = repair(dirty_sphere)

    assert isinstance(report, RepairReport)
    assert report.input_vertices == len(dirty_sphere.vertices)
    assert report.input_faces == len(dirty_sphere.faces)
    assert report.output_vertices == len(repaired.vertices)
    assert report.output_faces == len(repaired.faces)
    assert report.output_vertices > 0
    assert report.output_faces > 0
    # The fixture punches at least 3 holes and adds one debris component.
    assert report.holes_filled >= 3
    assert report.components_removed >= 1
    # __str__ mentions the headline outcome.
    text = str(report)
    assert "RepairReport" in text
    assert "manifold solid" in text


def test_repair_clean_passthrough(clean_sphere):
    repaired, report = repair(clean_sphere)

    assert is_manifold_solid(repaired)
    assert repaired.is_watertight
    assert report.holes_filled == 0
    assert report.components_removed == 0
    assert repaired.volume == pytest.approx(clean_sphere.volume, rel=1e-6)
    # Input must not be mutated.
    assert len(clean_sphere.faces) == report.input_faces


def test_repair_keep_all_components(dirty_sphere):
    repaired, report = repair(dirty_sphere, keep_largest_component=False)
    assert report.components_removed == 0
    # Debris tetrahedron survives, so there is more than one body if the
    # result is a manifold at all; watertightness must still hold.
    assert repaired.is_watertight


def test_repair_target_faces(dirty_sphere):
    target = 500
    repaired, report = repair(dirty_sphere, target_faces=target)

    assert is_manifold_solid(repaired)
    if _decimation_supported():
        assert report.decimated
        assert len(repaired.faces) < len(dirty_sphere.faces)
        assert len(repaired.faces) <= int(target * 1.5)
    else:
        # Skipped gracefully and noted in the report.
        assert not report.decimated
        assert any("decimation" in note for note in report.notes)
