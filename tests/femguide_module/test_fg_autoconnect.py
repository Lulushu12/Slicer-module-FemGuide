"""Auto-connect: wherever the surgeon moves the cutting structure, the walls
connect down into the base, never enter the bone, never land on bone outside
the drawn outline, and the blade path stays clear."""

import dataclasses

import numpy as np
import pytest

from femguide_core.geometry import intersection, snap_points_to_surface
from femguide.cutting import cutting_solids
from femguide.regenerate import regenerate
from femguide.validate import validate

from fg_scenario import blade_path_blocked, translation, with_cutting


def _volume(mesh):
    return mesh.volume if len(mesh.faces) else 0.0


@pytest.mark.parametrize("z", [22.0, 26.5, 32.0, 40.0])
def test_any_height_connects_without_entering_bone(plan, bone, z):
    moved = with_cutting(plan, placement=translation(-6.3, -1.0, z))
    g = regenerate(moved, bone)
    report = validate(moved, g, bone)
    assert not report.blocks, report
    assert report.metrics["bodies"] == 1.0
    assert _volume(intersection(g.walls, bone)) < 1e-6
    # Clearance is kept (measured along the walls' axis, here world z).
    lifted = bone.copy()
    lifted.apply_translation([0.0, 0.0, plan.base.clearance_mm - 0.02])
    assert _volume(intersection(g.walls, lifted)) < 1e-6
    assert blade_path_blocked(g) == []


def test_pushing_into_bone_trims_the_walls(plan, bone):
    pushed = with_cutting(plan, placement=translation(-6.3, -1.0, 22.0))
    g = regenerate(pushed, bone)
    # The placed block reached 3 mm into the bone; the walls were cut back
    # to follow the bone at the clearance instead.
    block = cutting_solids(pushed.cutting).walls
    assert _volume(intersection(block, bone)) > 50.0
    assert g.walls.volume < block.volume - 50.0
    assert validate(pushed, g, bone).metrics["bone_intrusion_mm3"] == 0.0


def test_lifting_grows_the_connection(plan, bone):
    low = regenerate(plan, bone)
    high = regenerate(with_cutting(plan, placement=translation(-6.3, -1.0, 40.0)), bone)
    assert high.extension_mm > low.extension_mm + 10.0
    assert high.walls.bounds[0][2] < 24.0  # still reaches down into the base


def test_overhanging_walls_never_land_outside_the_outline(plan, bone):
    # Moved sideways so the walls overhang the base edge (outline at 35 deg).
    over = with_cutting(plan, placement=translation(9.7, -1.0, 26.5))
    g = regenerate(over, bone)
    report = validate(over, g, bone)
    assert report.metrics["bodies"] == 1.0
    v = g.walls.vertices
    polar = np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))
    outside = v[polar > 37.0]
    assert len(outside) > 0  # the walls do overhang
    dist = np.linalg.norm(snap_points_to_surface(bone, outside) - outside, axis=1)
    assert dist.min() > 1.0


def test_construct_off_the_bone_is_disconnected(plan, bone):
    away = with_cutting(plan, placement=translation(53.7, -1.0, 26.5))
    g = regenerate(away, bone)
    assert g.extension_mm == 0.0
    assert "DISCONNECTED_PARTS" in {f.code for f in validate(away, g, bone).findings}


def test_without_a_base_the_walls_connect_to_the_bone(plan, bone):
    bare = dataclasses.replace(plan, base=None, pins=())
    bare = with_cutting(bare, placement=translation(-6.3, -1.0, 32.0))
    g = regenerate(bare, bone)
    report = validate(bare, g, bone)
    assert g.extension_mm > 0.0
    assert {f.code for f in report.findings} == {"NO_BASE"}
    assert g.walls.bounds[0][2] < 21.0  # reaches down to the bone (top at z=20)
