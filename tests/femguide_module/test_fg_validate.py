"""FemGuide validation rules: one test per block and warning."""

import dataclasses

import numpy as np
import trimesh

from femguide_core.export import mesh_sha256
from femguide.plan import Pin
from femguide.regenerate import regenerate
from femguide.validate import ValidationLimits, validate

from fg_scenario import sphere_point, translation, with_base, with_cutting


def codes(plan, bone, **kw):
    report = validate(plan, regenerate(plan, bone), bone, **kw)
    return {f.code for f in report.findings}, report


def with_pins(plan, *pins):
    return dataclasses.replace(plan, pins=pins)


# -- the healthy scenario --------------------------------------------------

def test_scenario_is_clean(plan, bone):
    found, report = codes(plan, bone)
    assert found == set()
    assert report.can_export()
    assert report.metrics["bodies"] == 1.0
    assert report.metrics["attachment_mm3"] > 100.0


# -- seating, connection, attachment ----------------------------------------

def test_walls_into_bone_block(plan, bone):
    found, _ = codes(with_cutting(plan, placement=translation(-6.3, -1.0, 22.0)), bone)
    assert found == {"GUIDE_INTERSECTS_BONE"}


def test_floating_cutting_structure_blocks(plan, bone):
    found, _ = codes(with_cutting(plan, placement=translation(-6.3, -1.0, 40.0)), bone)
    assert found == {"DISCONNECTED_PARTS"}


def test_sliver_attachment_blocks(plan, bone):
    found, report = codes(with_cutting(plan, placement=translation(-6.3, -1.0, 28.0)), bone)
    assert found == {"WEAK_ATTACHMENT"}
    assert 0.0 < report.metrics["attachment_mm3"] < 30.0


def test_open_bone_mesh_blocks(plan):
    bone = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    bottom = np.argsort(bone.triangles_center[:, 2])[:6]   # far from the guide
    keep = np.ones(len(bone.faces), dtype=bool)
    keep[bottom] = False
    holed = trimesh.Trimesh(bone.vertices, bone.faces[keep], process=False)
    found, _ = codes(dataclasses.replace(plan, bone_sha256=mesh_sha256(holed)), holed)
    assert "BONE_NOT_SOLID" in found


# -- slot layout ------------------------------------------------------------

def _u_turn(plan, gap):
    return with_cutting(plan, angles_deg=(0.0, 90.0, 90.0), lengths_mm=(10.0, gap, 10.0),
                        start_ext_mm=(), end_ext_mm=())


def test_u_turn_with_thin_wall_between_passes_blocks(plan, bone):
    found, report = codes(_u_turn(plan, 2.0), bone)
    assert "SLOT_PASSES_TOO_CLOSE" in found
    assert report.metrics["thinnest_slot_gap_mm"] < 1.0


def test_u_turn_with_merged_channels_blocks(plan, bone):
    found, report = codes(_u_turn(plan, 1.0), bone)
    assert "SLOT_PASSES_TOO_CLOSE" in found
    assert report.metrics["thinnest_slot_gap_mm"] < 0.0


def test_u_turn_with_room_is_fine_for_layout(plan, bone):
    found, report = codes(_u_turn(plan, 6.0), bone)
    assert not {"SLOT_PASSES_TOO_CLOSE", "SLOT_PASSES_CLOSE"} & found


def test_fold_back_bend_blocks(plan, bone):
    found, _ = codes(with_cutting(plan, angles_deg=(0.0, 172.0), lengths_mm=(8.0, 8.0)), bone)
    assert "SLOT_PASSES_TOO_CLOSE" in found


def test_ordinary_bend_has_no_layout_finding(plan, bone):
    found, _ = codes(with_cutting(plan, angles_deg=(0.0, 60.0)), bone)
    assert not {"SLOT_PASSES_TOO_CLOSE", "SLOT_PASSES_CLOSE"} & found


# -- pins -------------------------------------------------------------------

def test_pin_through_slot_blocks(plan, bone):
    z = float(np.sqrt(20.0 ** 2 - 1.0))
    found, _ = codes(with_pins(plan, Pin(position=(0.0, -0.3, z),
                                         direction=(0.0, 0.0, -1.0))), bone)
    assert "PIN_INTERSECTS_SLOT" in found


def test_pin_crossing_the_cut_inside_the_bone_blocks(plan, bone):
    # Auto direction on a sphere aims at the centre, through the cut plane.
    found, _ = codes(with_pins(plan, Pin(position=sphere_point(20.0, np.pi / 2))), bone)
    assert found == {"PIN_INTERSECTS_SLOT"}


def test_pin_outside_guide_blocks(plan, bone):
    found, _ = codes(with_pins(plan, Pin(position=sphere_point(80.0, 0.0),
                                         direction=(-1.0, 0.0, 0.0))), bone)
    assert "PIN_OUTSIDE_GUIDE" in found


def test_pin_pointing_out_of_bone_blocks(plan, bone):
    found, _ = codes(with_pins(plan, Pin(position=sphere_point(20.0, np.pi / 2),
                                         direction=(0.0, 0.3, 1.0))), bone)
    assert "PIN_POINTS_OUT" in found


def test_pin_off_the_bone_surface_blocks(plan, bone):
    above = sphere_point(20.0, np.pi / 2, radius=22.5)
    found, _ = codes(with_pins(plan, Pin(position=above, direction=(0.0, 0.0, -1.0))), bone)
    assert "PIN_NOT_ON_BONE" in found


def test_pin_at_base_edge_blocks(plan, bone):
    # Centre just outside the 35-degree outline: the hole notches the edge.
    edge = sphere_point(36.5, np.pi / 2)
    found, _ = codes(with_pins(plan, Pin(position=edge, direction=(0.0, 0.0, -1.0))), bone)
    assert "PIN_AT_EDGE" in found


def test_pin_well_inside_base_is_fine(plan, bone):
    found, _ = codes(with_pins(plan, Pin(position=sphere_point(25.0, np.pi / 2),
                                         direction=(0.0, 0.0, -1.0))), bone)
    assert found == set()


# -- thickness and clearances ---------------------------------------------------

def test_wall_thickness_warn_then_block(plan, bone):
    assert {"WALL_THIN"} == codes(with_cutting(plan, wall_thickness_mm=1.5), bone)[0]
    assert {"WALL_TOO_THIN"} == codes(with_cutting(plan, wall_thickness_mm=0.8), bone)[0]


def test_base_thickness_warn(plan, bone):
    # Thinning the base lowers its top, and the walls stay where they were
    # placed, so they now barely touch it: the attachment check fires too.
    assert {"BASE_THIN", "WEAK_ATTACHMENT"} == codes(
        with_base(plan, thickness_mm=1.5), bone)[0]


def test_clearance_too_tight_warns(plan, bone):
    assert {"BLADE_CLEARANCE_MARGINAL"} == codes(
        with_cutting(plan, blade_clearance_mm=0.05), bone)[0]
    assert {"BONE_CLEARANCE_MARGINAL"} == codes(with_base(plan, clearance_mm=0.1), bone)[0]


def test_clearance_too_loose_warns(plan, bone):
    assert "BLADE_CLEARANCE_LOOSE" in codes(with_cutting(plan, blade_clearance_mm=1.0), bone)[0]
    assert "BONE_CLEARANCE_LOOSE" in codes(with_base(plan, clearance_mm=1.5), bone)[0]


def test_negative_bone_clearance_blocks(plan, bone):
    found, _ = codes(with_base(plan, clearance_mm=-0.2), bone)
    # The base itself now cuts into the bone, so both checks fire.
    assert found == {"BONE_INTERFERENCE", "GUIDE_INTERSECTS_BONE"}


def test_missing_base_warns(plan, bone):
    found, report = codes(dataclasses.replace(plan, base=None, pins=()), bone)
    assert found == {"NO_BASE"}
    assert not report.can_export() and report.can_export(["NO_BASE"])


def test_limits_are_configurable(plan, bone):
    strict = ValidationLimits(recommended_min_mm=3.0)
    assert "WALL_THIN" in codes(plan, bone, limits=strict)[0]


def test_pin_inside_outline_but_too_close_to_edge_blocks(plan, bone):
    # Centre 1.2 mm inside the outline: the hole fits, but the wall on the
    # outer side is thinner than the 1 mm printable minimum.
    near = sphere_point(31.5, np.pi / 2)
    found, _ = codes(with_pins(plan, Pin(position=near, direction=(0.0, 0.0, -1.0))), bone)
    assert found == {"PIN_AT_EDGE"}
