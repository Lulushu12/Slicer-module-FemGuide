"""FemGuide validation rules: one test per block and warning."""

import dataclasses

import numpy as np

from femguide.plan import Pin
from femguide.regenerate import regenerate
from femguide.validate import ValidationLimits, validate

from fg_scenario import sphere_point, translation, with_base, with_cutting


def codes(plan, bone, **kw):
    report = validate(plan, regenerate(plan, bone), bone, **kw)
    return {f.code for f in report.findings}, report


def test_scenario_is_clean(plan, bone):
    found, report = codes(plan, bone)
    assert found == set()
    assert report.can_export()
    assert report.metrics["bodies"] == 1.0


def test_walls_into_bone_block(plan, bone):
    found, _ = codes(with_cutting(plan, placement=translation(-6.3, -1.0, 22.0)), bone)
    assert found == {"GUIDE_INTERSECTS_BONE"}


def test_floating_cutting_structure_blocks(plan, bone):
    found, _ = codes(with_cutting(plan, placement=translation(-6.3, -1.0, 40.0)), bone)
    assert found == {"DISCONNECTED_PARTS"}


def test_pin_through_slot_blocks(plan, bone):
    z = float(np.sqrt(20.0 ** 2 - 1.0))
    bad = dataclasses.replace(plan, pins=(Pin(position=(0.0, -0.3, z),
                                              direction=(0.0, 0.0, -1.0)),))
    found, _ = codes(bad, bone)
    assert found == {"PIN_INTERSECTS_SLOT"}


def test_pin_outside_guide_blocks(plan, bone):
    far = dataclasses.replace(plan, pins=(Pin(position=sphere_point(80.0, 0.0)),))
    found, _ = codes(far, bone)
    assert found == {"PIN_OUTSIDE_GUIDE"}


def test_wall_thickness_warn_then_block(plan, bone):
    assert {"WALL_THIN"} == codes(with_cutting(plan, wall_thickness_mm=1.5), bone)[0]
    assert {"WALL_TOO_THIN"} == codes(with_cutting(plan, wall_thickness_mm=0.8), bone)[0]


def test_base_thickness_warn(plan, bone):
    assert {"BASE_THIN"} == codes(with_base(plan, thickness_mm=1.5), bone)[0]


def test_clearance_warnings(plan, bone):
    assert {"BLADE_CLEARANCE_MARGINAL"} == codes(
        with_cutting(plan, blade_clearance_mm=0.05), bone)[0]
    assert {"BONE_CLEARANCE_MARGINAL"} == codes(with_base(plan, clearance_mm=0.1), bone)[0]


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
    assert {"WALL_THIN"} == codes(plan, bone, limits=strict)[0]
