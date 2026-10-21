"""FemGuide plan model: structural checks and JSON round trip."""

import dataclasses

import numpy as np
import pytest

from femguide.plan import BaseParams, CuttingParams, FemGuidePlan, Pin

from fg_scenario import with_cutting


def test_json_round_trip_is_exact(plan, tmp_path):
    assert FemGuidePlan.from_json(plan.to_json()) == plan
    plan.save(tmp_path / "case.json")
    assert FemGuidePlan.load(tmp_path / "case.json") == plan


def test_plans_are_immutable_and_comparable(plan):
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.case_id = "other"
    edited = with_cutting(plan, wall_thickness_mm=3.0)
    assert edited != plan
    assert with_cutting(edited, wall_thickness_mm=2.5) == plan


def test_extensions_default_to_zero_per_segment():
    c = CuttingParams(angles_deg=(0, 10, 20), lengths_mm=(5, 5, 5))
    assert c.start_ext_mm == (0.0, 0.0, 0.0)
    assert c.end_ext_mm == (0.0, 0.0, 0.0)


def test_slot_width_is_blade_plus_clearance():
    c = CuttingParams(angles_deg=(0,), lengths_mm=(5,),
                      blade_thickness_mm=1.27, blade_clearance_mm=0.2)
    assert c.slot_width_mm == pytest.approx(1.47)


@pytest.mark.parametrize("kwargs", [
    dict(angles_deg=(0, 1), lengths_mm=(5,)),
    dict(angles_deg=(), lengths_mm=()),
    dict(angles_deg=(0,), lengths_mm=(0,)),
    dict(angles_deg=(0,), lengths_mm=(5,), start_ext_mm=(1, 2)),
    dict(angles_deg=(0,), lengths_mm=(5,), end_ext_mm=(-1,)),
    dict(angles_deg=(0,), lengths_mm=(5,), blade_thickness_mm=0),
    dict(angles_deg=(0,), lengths_mm=(5,), blade_clearance_mm=-0.1),
    dict(angles_deg=(0,), lengths_mm=(5,), wall_thickness_mm=0),
])
def test_meaningless_cutting_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        CuttingParams(**kwargs)


def test_non_rigid_placement_rejected():
    scaled = tuple((np.diag([2.0, 1.0, 1.0, 1.0])).ravel())
    with pytest.raises(ValueError, match="rigid"):
        CuttingParams(angles_deg=(0,), lengths_mm=(5,), placement=scaled)
    mirrored = tuple((np.diag([-1.0, 1.0, 1.0, 1.0])).ravel())
    with pytest.raises(ValueError, match="rigid"):
        CuttingParams(angles_deg=(0,), lengths_mm=(5,), placement=mirrored)


def test_base_needs_three_points():
    with pytest.raises(ValueError):
        BaseParams(outline_points=((0, 0, 0), (1, 0, 0)))


def test_pin_direction_normalised_and_blind_not_supported():
    assert Pin(position=(0, 0, 0), direction=(0, 0, 5)).direction == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Pin(position=(0, 0, 0), direction=(0, 0, 0))
    with pytest.raises(NotImplementedError):
        Pin(position=(0, 0, 0), through=False)


def test_unknown_schema_version_rejected(plan):
    data = plan.to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        FemGuidePlan.from_dict(data)
