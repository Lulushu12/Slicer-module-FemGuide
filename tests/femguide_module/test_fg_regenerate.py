"""Regeneration from plan (D4): composition, determinism, speed."""

import dataclasses
import time

import numpy as np
import pytest
import trimesh

from femguide_core.geometry import intersection, is_manifold_solid, to_manifold
from femguide.plan import BaseParams
from femguide.regenerate import BoneMismatchError, regenerate

from fg_scenario import sphere_point


def test_guide_is_one_closed_solid(plan, bone):
    g = regenerate(plan, bone)
    assert is_manifold_solid(g.guide) and g.guide.is_watertight
    assert len(to_manifold(g.guide).decompose()) == 1
    assert 0 < g.guide.volume < g.material.volume


def test_slot_cuts_through_the_base_to_the_bone(plan, bone):
    g = regenerate(plan, bone)
    # The slot channel does pass through base material...
    assert intersection(g.base, g.slot).volume > 1.0
    # ...and none of the guide is left inside it, so the blade reaches bone.
    leftover = intersection(g.guide, g.slot)
    assert len(leftover.faces) == 0 or leftover.volume < 1e-6


def test_auto_pin_direction_is_inward_normal(plan, bone):
    g = regenerate(plan, bone)
    for pin, direction in zip(plan.pins, g.pin_directions):
        inward = -np.array(pin.position) / np.linalg.norm(pin.position)
        assert np.dot(direction, inward) > 0.99


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
