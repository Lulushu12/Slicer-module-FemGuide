"""Regeneration from plan (D4): composition, determinism, speed."""

import dataclasses
import time

import numpy as np
import pytest
import trimesh

from femguide_core.geometry import is_manifold_solid, to_manifold
from femguide.plan import BaseParams, Pin
from femguide.regenerate import BoneMismatchError, regenerate

from fg_scenario import (blade_path_blocked, sphere_point, translation,
                         with_base, with_cutting)


def test_guide_is_one_closed_solid(plan, bone):
    g = regenerate(plan, bone)
    assert is_manifold_solid(g.guide) and g.guide.is_watertight
    assert len(to_manifold(g.guide).decompose()) == 1
    assert 0 < g.guide.volume < g.material.volume


def test_blade_path_is_clear_through_walls_and_base(plan, bone):
    assert blade_path_blocked(regenerate(plan, bone)) == []


def test_blade_path_clear_with_flipped_placement(plan, bone):
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    flip[:3, 3] = [-6.3, 1.0, 26.5]
    g = regenerate(with_cutting(plan, placement=tuple(flip.ravel())), bone)
    assert blade_path_blocked(g) == []


def test_blade_path_clear_when_base_is_thicker_than_walls(plan, bone):
    thick = with_base(with_cutting(plan, slot_depth_mm=4.0,
                                   placement=translation(-6.3, -1.0, 23.5)),
                      thickness_mm=8.0)
    assert blade_path_blocked(regenerate(thick, bone)) == []


def test_auto_pin_direction_is_inward_normal(plan, bone):
    auto = dataclasses.replace(plan, pins=(Pin(position=sphere_point(20.0, np.pi / 2)),))
    g = regenerate(auto, bone)
    inward = -np.array(auto.pins[0].position) / np.linalg.norm(auto.pins[0].position)
    assert np.dot(g.pin_directions[0], inward) > 0.99


def test_pins_survive_a_base_outline_edit(plan, bone):
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    smaller = dataclasses.replace(
        plan, base=BaseParams(outline_points=tuple(sphere_point(30.0, p) for p in t)))
    no_pins = dataclasses.replace(smaller, pins=())
    with_pins = regenerate(smaller, bone)
    assert len(with_pins.pin_holes) == 2
    assert with_pins.guide.volume < regenerate(no_pins, bone).guide.volume - 1.0


def test_regeneration_is_deterministic(plan, bone):
    a, b = regenerate(plan, bone), regenerate(plan, bone)
    assert np.array_equal(a.guide.vertices, b.guide.vertices)
    assert np.array_equal(a.guide.faces, b.guide.faces)


def test_regeneration_meets_the_two_second_target(plan, bone):
    # D4: regeneration must feel live, ~2 s worst case.
    t0 = time.perf_counter()
    regenerate(plan, bone)
    assert time.perf_counter() - t0 < 2.0


def test_plan_without_base(plan, bone):
    g = regenerate(dataclasses.replace(plan, base=None, pins=()), bone)
    assert g.base is None
    assert is_manifold_solid(g.guide)


def test_wrong_bone_is_refused(plan):
    other = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    with pytest.raises(BoneMismatchError):
        regenerate(plan, other)
