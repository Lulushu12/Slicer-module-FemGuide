"""Shared FemGuide scenario: a guide planned on a 20 mm sphere "bone".

The base is a cap outlined at 35 degrees from the top; the two-segment
cutting structure sits on top of the base with local -z toward the bone;
two fixation pins sit on the cap either side of the cut, parallel to the
cut plane so they stay clear of the blade's path both deep in the bone and
where they stick out of the guide (auto-direction pins on a sphere aim at
the centre, straight through the cut).
"""

import numpy as np
import pytest
import trimesh

from femguide_core.export import mesh_sha256
from femguide.plan import BaseParams, CuttingParams, FemGuidePlan, Pin

from fg_scenario import BONE_RADIUS, sphere_point, translation


@pytest.fixture(scope="session")
def bone():
    return trimesh.creation.icosphere(subdivisions=4, radius=BONE_RADIUS)


@pytest.fixture
def plan(bone):
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    outline = tuple(sphere_point(35.0, phi) for phi in t)
    return FemGuidePlan(
        case_id="TEST-001",
        bone_sha256=mesh_sha256(bone),
        cutting=CuttingParams(angles_deg=(0.0, 20.0), lengths_mm=(7.0, 6.0),
                              placement=translation(-6.3, -1.0, 26.5)),
        base=BaseParams(outline_points=outline),
        pins=(Pin(position=sphere_point(20.0, np.pi / 2), direction=(0.0, 0.0, -1.0)),
              Pin(position=sphere_point(20.0, 3 * np.pi / 2), direction=(0.0, 0.0, -1.0))),
    )


