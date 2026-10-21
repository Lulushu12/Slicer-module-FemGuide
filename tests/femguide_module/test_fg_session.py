"""Editing session: what the wizard does, tested without Slicer."""

import math

import numpy as np
import pytest
import trimesh

from femguide.plan import FemGuidePlan
from femguide.regenerate import regenerate
from femguide.session import PROTOTYPE_ANGLES_DEG, PlanSession, orthonormalized

from fg_scenario import sphere_point


@pytest.fixture
def session(plan, bone):
    return PlanSession(bone, plan)


def test_start_uses_prototype_cut(bone):
    s = PlanSession.start(bone, "CASE-9")
    assert s.plan.cutting.angles_deg == PROTOTYPE_ANGLES_DEG
    assert s.plan.base is None and s.plan.pins == ()


def test_wrong_bone_refused(plan):
    with pytest.raises(ValueError):
        PlanSession(trimesh.creation.icosphere(subdivisions=2, radius=20.0), plan)


def test_edits_undo_redo(session):
    original = session.plan
    session.set_cutting(wall_thickness_mm=3.0)
    session.set_base(thickness_mm=4.0)
    assert session.plan.base.thickness_mm == 4.0
    assert session.undo().cutting.wall_thickness_mm == 3.0
    assert session.undo() == original
    assert session.redo().cutting.wall_thickness_mm == 3.0


def test_changing_segment_count_resets_extensions(session):
    session.set_cutting(start_ext_mm=(1.0, 2.0))
    session.set_cutting(angles_deg=(0.0, 10.0, 20.0), lengths_mm=(5.0, 5.0, 5.0))
    assert session.plan.cutting.start_ext_mm == (0.0, 0.0, 0.0)


def test_placement_round_off_is_cleaned(session):
    m = session.plan.cutting.placement_matrix.copy()
    m[:3, :3] += 1e-4 * np.random.default_rng(0).normal(size=(3, 3))
    session.set_placement(m)  # would fail the rigid check without cleanup
    r = session.plan.cutting.placement_matrix[:3, :3]
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.allclose(orthonormalized(m)[:3, 3], m[:3, 3])


def test_nudge_translates_along_own_axes(session):
    before = session.plan.cutting.placement_matrix
    session.nudge(translate_mm=(2.0, 0.0, -1.0))
    after = session.plan.cutting.placement_matrix
    assert np.allclose(after[:3, 3] - before[:3, 3],
                       2.0 * before[:3, 0] - 1.0 * before[:3, 2])


def test_nudge_rotates_about_cut_line_middle(session):
    center = session.polyline_center_local()
    before = session.plan.cutting.placement_matrix @ center
    session.nudge(rotate_deg=(0.0, 0.0, 90.0))
    m = session.plan.cutting.placement_matrix
    assert np.allclose(m @ center, before)
    assert np.allclose(m[:3, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_drop_onto_base_gives_a_valid_guide(session):
    # Without pins: on a circular outline the cut's direction is arbitrary,
    # and validation rightly blocks a cut that runs into existing pins.
    session.set_pin_positions([])
    session.set_placement(np.eye(4))       # somewhere useless
    session.drop_onto_base()
    m = session.plan.cutting.placement_matrix
    assert np.dot(m[:3, 2], [0.0, 0.0, 1.0]) > 0.99   # local z = outward normal
    report = session.validate()
    assert not report.blocks, report


def test_drop_onto_base_needs_an_outline(bone):
    with pytest.raises(ValueError):
        PlanSession.start(bone, "C").drop_onto_base()


def test_pin_positions_sync_with_point_list(session):
    first = session.plan.pins[0]
    session.set_pin_positions([first.position, sphere_point(25.0, 0.0), sphere_point(25.0, 1.0)])
    assert len(session.plan.pins) == 3
    assert session.plan.pins[0] == first                  # settings kept
    assert session.plan.pins[2].direction is None          # new pins are auto
    session.set_pin_positions([sphere_point(25.0, 0.0)])
    assert len(session.plan.pins) == 1


def test_pin_angles_read_out_and_set(session):
    session.set_pin(0, direction=None)
    tilt, _ = session.pin_angles(0)
    assert tilt < 1.0                                   # auto = along the normal
    session.set_pin_angles(0, 30.0, 90.0)
    tilt, azimuth = session.pin_angles(0)
    assert tilt == pytest.approx(30.0, abs=1e-6)
    assert azimuth == pytest.approx(90.0, abs=1e-6)


def test_base_is_reused_while_only_the_cut_changes(session):
    first = session.regenerate()
    session.nudge(translate_mm=(0.5, 0.0, 0.0))
    second = session.regenerate()
    assert second.base is first.base
    session.set_base(thickness_mm=3.5)
    assert session.regenerate().base is not first.base


def test_cache_does_not_change_results(session, bone):
    session.nudge(rotate_deg=(0.0, 0.0, 5.0))
    session.regenerate()
    cached = session.regenerate()
    fresh = regenerate(session.plan, bone)
    assert np.array_equal(cached.guide.vertices, fresh.guide.vertices)
    assert np.array_equal(cached.guide.faces, fresh.guide.faces)


def test_export_through_session(session, tmp_path):
    record = session.export(tmp_path / "TEST-001.stl")
    assert record.exists()
    assert FemGuidePlan.from_json(session.plan.to_json()) == session.plan


def test_aligning_pins_with_the_cut_clears_crossings(session):
    # Auto-direction pins on the sphere aim at the centre, through the cut.
    session.set_pin(0, direction=None)
    session.set_pin(1, direction=None)
    codes = {f.code for f in session.validate().findings}
    assert "PIN_INTERSECTS_SLOT" in codes
    before = session.plan
    session.align_pins_with_cut()
    assert session.undo() == before          # one undo step for all pins
    session.redo()
    assert np.allclose(session.plan.pins[0].direction,
                       -session.plan.cutting.placement_matrix[:3, 2])
    assert session.validate().findings == []


def test_align_refuses_a_cut_facing_away(session):
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    flip[:3, 3] = session.plan.cutting.placement_matrix[:3, 3]
    session.set_placement(flip)
    with pytest.raises(ValueError):
        session.align_pin_with_cut(0)
