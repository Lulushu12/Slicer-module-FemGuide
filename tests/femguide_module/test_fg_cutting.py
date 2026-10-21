"""Slot and wall geometry, checked against hand-computed volumes."""

import numpy as np
import pytest
import trimesh

from femguide_core.geometry import difference, intersection, is_manifold_solid
from femguide.cutting import cutting_solids
from femguide.plan import CuttingParams

from fg_scenario import translation

L, S, W, D = 20.0, 1.5, 2.0, 10.0  # length, slot width, wall, slot depth


def straight(**kwargs):
    return CuttingParams(angles_deg=(0.0,), lengths_mm=(L,), blade_thickness_mm=S,
                         blade_clearance_mm=0.0, wall_thickness_mm=W,
                         slot_depth_mm=D, **kwargs)


def test_walls_volume_and_closed_ends():
    walls = cutting_solids(straight()).walls
    assert is_manifold_solid(walls)
    # Width slot + 2 walls, length segment + an end wall at each end.
    assert walls.volume == pytest.approx((S + 2 * W) * (L + 2 * W) * D, rel=1e-9)


def test_slot_is_centred_and_open_along_the_blade_path():
    slot = cutting_solids(straight()).slot
    (x0, y0, z0), (x1, y1, z1) = slot.bounds
    assert (y0, y1) == pytest.approx((-S / 2, S / 2))
    assert (x0, x1) == pytest.approx((0.0, L))
    # The slot is the blade's whole path: far past the walls both ways.
    assert z0 < -100.0 and z1 > 100.0


def test_slot_is_captured_inside_walls():
    solids = cutting_solids(straight())
    inside = intersection(solids.walls, solids.slot)
    assert inside.volume == pytest.approx(S * L * D, rel=1e-9)
    cut = difference(solids.walls, solids.slot)
    assert is_manifold_solid(cut)
    assert cut.volume == pytest.approx((S + 2 * W) * (L + 2 * W) * D - S * L * D,
                                       rel=1e-9)


def test_extensions_lengthen_slot_and_walls_follow():
    solids = cutting_solids(straight(start_ext_mm=(3.0,), end_ext_mm=(4.0,)))
    assert solids.slot.bounds[:, 0] == pytest.approx((-3.0, L + 4.0))
    assert solids.walls.bounds[:, 0] == pytest.approx((-3.0 - W, L + 4.0 + W))


def _inside(solid, point, size=0.05):
    """True if a tiny cube at ``point`` lies entirely inside ``solid``."""
    probe = trimesh.creation.box(extents=(size, size, size))
    probe.apply_translation(point)
    overlap = intersection(probe, solid)
    return len(overlap.faces) > 0 and overlap.volume == pytest.approx(size ** 3, rel=1e-6)


def test_bent_polyline_channel_is_continuous_through_bends():
    params = CuttingParams(angles_deg=(0, 45, 45), lengths_mm=(10, 8, 8))
    solids = cutting_solids(params)
    cut = difference(solids.walls, solids.slot)
    assert is_manifold_solid(cut)
    # The whole cut line, bends included, lies inside the slot channel, at
    # mid-height and near both wall faces. (The line's two outer endpoints
    # sit exactly on the slot's end faces, so they are skipped.)
    pts = solids.polyline
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, 9):
            on_line = a + t * (b - a)
            if np.allclose(on_line, pts[0]) or np.allclose(on_line, pts[-1]):
                continue
            for z in (-params.slot_depth_mm / 2 + 0.1, 0.0,
                      params.slot_depth_mm / 2 - 0.1):
                assert _inside(solids.slot, on_line + [0, 0, z])
    # And the walls are solid material beside the channel.
    mid = (pts[0] + pts[1]) / 2
    beside = mid + [0, params.slot_width_mm / 2 + params.wall_thickness_mm / 2, 0]
    assert _inside(cut, beside)
    assert not _inside(cut, mid)


def test_placement_moves_everything_rigidly():
    base = cutting_solids(straight())
    moved = cutting_solids(straight(placement=translation(5.0, -2.0, 30.0)))
    assert moved.walls.bounds == pytest.approx(base.walls.bounds + [5.0, -2.0, 30.0])
    assert moved.slot.volume == pytest.approx(base.slot.volume)
    assert moved.polyline == pytest.approx(base.polyline + [5.0, -2.0, 30.0])
