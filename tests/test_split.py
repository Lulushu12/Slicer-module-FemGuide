"""Layer-2 plane-splitting tests (future mold parting-plane operation)."""

import numpy as np
import pytest
import trimesh

from femguide_core.geometry.booleans import NonManifoldError
from femguide_core.geometry.repair import is_manifold_solid
from femguide_core.geometry.split import split_by_plane


def _assert_solid(mesh: trimesh.Trimesh):
    assert is_manifold_solid(mesh)
    assert mesh.is_watertight


def test_split_box_off_center_axis_aligned():
    box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))  # centered at 0
    above, below = split_by_plane(box, plane_origin=(3.0, 0.0, 0.0),
                                  plane_normal=(1.0, 0.0, 0.0))

    _assert_solid(above)
    _assert_solid(below)

    # Analytic: plane x=3 leaves 7 x 20 x 20 above, 13 x 20 x 20 below.
    assert above.volume == pytest.approx(7.0 * 20.0 * 20.0, rel=0.01)
    assert below.volume == pytest.approx(13.0 * 20.0 * 20.0, rel=0.01)
    assert above.volume + below.volume == pytest.approx(box.volume, rel=0.01)

    # 'above' really is on the +normal side, 'below' on the other.
    assert above.vertices[:, 0].min() >= 3.0 - 1e-6
    assert below.vertices[:, 0].max() <= 3.0 + 1e-6


def test_split_sphere_oblique_plane(clean_sphere):
    origin = (2.0, 1.0, 0.0)
    normal = (1.0, 1.0, 1.0)  # non-unit on purpose; must be normalized inside
    above, below = split_by_plane(clean_sphere, origin, normal)

    _assert_solid(above)
    _assert_solid(below)
    assert above.volume + below.volume == pytest.approx(
        clean_sphere.volume, rel=0.01
    )

    # Halves lie on the correct sides of the plane.
    unit = np.asarray(normal, dtype=float)
    unit /= np.linalg.norm(unit)
    d_above = (above.vertices - np.asarray(origin)) @ unit
    d_below = (below.vertices - np.asarray(origin)) @ unit
    assert d_above.min() >= -1e-6
    assert d_below.max() <= 1e-6


def test_split_plane_missing_solid(clean_sphere):
    above, below = split_by_plane(
        clean_sphere, plane_origin=(100.0, 0.0, 0.0), plane_normal=(1.0, 0.0, 0.0)
    )
    assert len(above.faces) == 0
    _assert_solid(below)
    assert below.volume == pytest.approx(clean_sphere.volume, rel=1e-6)


def test_split_non_manifold_raises(clean_sphere):
    holed = trimesh.Trimesh(
        vertices=clean_sphere.vertices.copy(),
        faces=clean_sphere.faces[1:].copy(),
        process=False,
    )
    with pytest.raises(NonManifoldError):
        split_by_plane(holed, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def test_split_zero_normal_raises(clean_sphere):
    with pytest.raises(ValueError):
        split_by_plane(clean_sphere, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
