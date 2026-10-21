"""Helpers for the FemGuide test scenario (see conftest.py)."""

import dataclasses

import numpy as np
import pytest
import trimesh

from femguide_core.geometry import intersection


BONE_RADIUS = 20.0


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return tuple(m.ravel())


def sphere_point(theta_deg, phi_rad, radius=BONE_RADIUS):
    th = np.radians(theta_deg)
    return (radius * np.sin(th) * np.cos(phi_rad),
            radius * np.sin(th) * np.sin(phi_rad),
            radius * np.cos(th))


def with_cutting(plan, **changes):
    return dataclasses.replace(plan, cutting=dataclasses.replace(plan.cutting, **changes))


def with_base(plan, **changes):
    return dataclasses.replace(plan, base=dataclasses.replace(plan.base, **changes))


def inside(solid, point, size=0.05):
    """True if a tiny cube at ``point`` lies entirely inside ``solid``."""
    probe = trimesh.creation.box(extents=(size, size, size))
    probe.apply_translation(point)
    overlap = intersection(probe, solid)
    return len(overlap.faces) > 0 and overlap.volume == pytest.approx(size ** 3, rel=1e-6)


def touches(solid, point, size=0.05):
    """True if a tiny cube at ``point`` overlaps ``solid`` at all."""
    probe = trimesh.creation.box(extents=(size, size, size))
    probe.apply_translation(point)
    overlap = intersection(probe, solid)
    return len(overlap.faces) > 0 and overlap.volume > 1e-12


def blade_path_blocked(geometry, samples=7, width=0.05):
    """Independent check that the blade's path is clear: drop a thin needle
    along the cutting structure's local z through each sample point of the
    cut line, long enough to cross the whole guide, and return the points
    whose needle finds guide material."""
    z_axis = geometry.placement[:3, 2]
    lo, hi = geometry.guide.bounds
    length = 2.0 * float(np.linalg.norm(hi - lo)) + 10.0
    pts = geometry.polyline
    blocked = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, samples):
            p = a + t * (b - a)
            if np.allclose(p, pts[0]) or np.allclose(p, pts[-1]):
                continue
            needle = trimesh.creation.box(extents=(width, width, length))
            frame = np.eye(4)
            frame[:3, :3] = geometry.placement[:3, :3]
            frame[:3, 3] = p
            needle.apply_transform(frame)
            overlap = intersection(needle, geometry.guide)
            if len(overlap.faces) and overlap.volume > 1e-9:
                blocked.append(p)
    return blocked
