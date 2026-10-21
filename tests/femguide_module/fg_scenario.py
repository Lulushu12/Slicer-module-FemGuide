"""Helpers for the FemGuide test scenario (see conftest.py)."""

import dataclasses

import numpy as np


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
