"""Regenerate FemGuide geometry from scratch from a plan (D4).

``regenerate(plan, bone)`` is the only way guide geometry comes into
existence: (walls of the cutting structure UNION patient-specific base)
MINUS (slot UNION pin holes). The slot is subtracted after the union so it
also cuts through the base and the blade reaches bone.

``bone`` must be the repaired bone mesh the plan was made on (repair runs
once at case import, not here); its hash must match ``plan.bone_sha256``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import trimesh

from femguide_core.export import mesh_sha256
from femguide_core.geometry import difference, molded_base, oriented_cylinder, union

from femguide.cutting import cutting_solids
from femguide.plan import FemGuidePlan, Pin

# Pin holes run from slightly inside the bone back out well past any guide
# material; only their intersection with the guide matters.
_PIN_INTO_BONE_MM = 2.0
_PIN_OUT_OF_BONE_MM = 60.0
_PIN_SECTIONS = 32


class BoneMismatchError(ValueError):
    """The bone mesh is not the one the plan was made on."""


@dataclass
class GuideGeometry:
    guide: trimesh.Trimesh
    walls: trimesh.Trimesh
    slot: trimesh.Trimesh
    base: Optional[trimesh.Trimesh]
    pin_holes: List[trimesh.Trimesh]
    pin_directions: List[np.ndarray]
    polyline: np.ndarray
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def material(self) -> trimesh.Trimesh:
        """Guide material before any subtraction (walls union base)."""
        parts = [self.walls] + ([self.base] if self.base is not None else [])
        return union(parts)


def resolve_pin_direction(pin: Pin, bone: trimesh.Trimesh) -> np.ndarray:
    """The pin's direction into the bone; for "auto", the inward normal of
    the bone face nearest to the pin position."""
    if pin.direction is not None:
        return np.array(pin.direction, dtype=float)
    position = np.array(pin.position, dtype=float)
    nearest = int(np.argmin(np.linalg.norm(bone.triangles_center - position, axis=1)))
    return -np.asarray(bone.face_normals[nearest], dtype=float)


def pin_hole(pin: Pin, direction: np.ndarray) -> trimesh.Trimesh:
    position = np.array(pin.position, dtype=float)
    length = _PIN_INTO_BONE_MM + _PIN_OUT_OF_BONE_MM
    center = position + direction * (_PIN_INTO_BONE_MM - _PIN_OUT_OF_BONE_MM) / 2.0
    return oriented_cylinder(center, direction, pin.diameter_mm / 2.0, length,
                             sections=_PIN_SECTIONS)


def regenerate(plan: FemGuidePlan, bone: trimesh.Trimesh) -> GuideGeometry:
    if mesh_sha256(bone) != plan.bone_sha256:
        raise BoneMismatchError(
            "bone mesh does not match the plan's bone_sha256; this plan was "
            "made on a different mesh (or the mesh was modified)")
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    cutting = cutting_solids(plan.cutting)
    timings["cutting_structure"] = time.perf_counter() - t0

    base = None
    if plan.base is not None:
        t0 = time.perf_counter()
        base = molded_base(bone, np.array(plan.base.outline_points),
                           thickness=plan.base.thickness_mm,
                           clearance=plan.base.clearance_mm)
        timings["base"] = time.perf_counter() - t0

    directions = [resolve_pin_direction(p, bone) for p in plan.pins]
    holes = [pin_hole(p, d) for p, d in zip(plan.pins, directions)]

    t0 = time.perf_counter()
    material = union([cutting.walls] + ([base] if base is not None else []))
    guide = difference(material, [cutting.slot] + holes)
    timings["booleans"] = time.perf_counter() - t0

    return GuideGeometry(
        guide=guide, walls=cutting.walls, slot=cutting.slot, base=base,
        pin_holes=holes, pin_directions=directions, polyline=cutting.polyline,
        timings=timings)
