"""FemGuide plan model: structural checks and JSON round trip."""

import dataclasses
import json

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


def test_pin_direction_normalised():
    assert Pin(position=(0, 0, 0), direction=(0, 0, 5)).direction == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Pin(position=(0, 0, 0), direction=(0, 0, 0))


def test_unit_pin_directions_round_trip_bit_exact():
    rng = np.random.default_rng(1)
    for _ in range(500):
        d = rng.normal(size=3)
        pin = Pin(position=(1.0, 2.0, 3.0), direction=tuple(d))
        again = Pin(**json.loads(json.dumps(dataclasses.asdict(pin))))
        assert again == pin


@pytest.mark.parametrize("bad", [
    dict(angles_deg=(float("nan"),), lengths_mm=(5,)),
    dict(angles_deg=(float("inf"),), lengths_mm=(5,)),
    dict(angles_deg=(True,), lengths_mm=(5,)),
    dict(angles_deg=("1",), lengths_mm=(5,)),
    dict(angles_deg=(0,), lengths_mm=(5,), placement=(1, 0, 0, float("nan"),
                                                       0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)),
])
def test_non_finite_and_non_numeric_inputs_rejected(bad):
    with pytest.raises(ValueError):
        CuttingParams(**bad)


@pytest.mark.parametrize("field,value", [
    ("case_id", 5), ("case_id", "  "), ("case_id", ["x"]),
    ("bone_sha256", "abc"), ("schema_version", True),
])
def test_plan_fields_are_strictly_typed(plan, field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(plan, **{field: value})


def test_unknown_schema_version_rejected(plan):
    data = plan.to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        FemGuidePlan.from_dict(data)
