"""Editing session: everything the clinician wizard does, minus the UI.

The Slicer wizard (Phase 2b) is a view: it turns clicks, drags and number
fields into calls on a :class:`PlanSession` and shows what comes back. All
editing rules live here so they are tested in CI without Slicer:

- every edit commits a new immutable plan to the undo history (D4);
- geometry is regenerated from the current plan, reusing the base when only
  the cutting structure or pins changed;
- the cutting structure can be dropped onto the base automatically and then
  nudged along / rotated about its own axes (the numeric half of the
  "direct drag + numeric panel" pair, docs/ARCHITECTURE.md 5.2);
- pin directions read out and set as tilt / azimuth angles.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import trimesh

from femguide_core.export import mesh_sha256
from femguide_core.geometry import (
    polyline_from_angles,
    safe_normalize,
    snap_points_to_surface,
    surface_normal,
)
from femguide_core.history import PlanHistory
from femguide_core.validation import ValidationReport

from femguide.export import export_guide
from femguide.plan import BaseParams, CuttingParams, FemGuidePlan, Pin
from femguide.regenerate import GuideGeometry, regenerate, resolve_pin_direction
from femguide.validate import ValidationLimits, validate

# The prototype's default cut (LineGenerator UI defaults).
PROTOTYPE_ANGLES_DEG = (8.0, 40.0, 45.0, 45.0, 42.0)
PROTOTYPE_LENGTHS_MM = (50.0, 20.5, 25.0, 11.0, 30.0)
# Height of the dropped-on block's bottom above the base's top.
_DROP_GAP_MM = 1.0


def orthonormalized(matrix) -> np.ndarray:
    """Nearest rigid transform to ``matrix`` (removes the round-off that
    interactive handles accumulate, which the plan would reject)."""
    m = np.array(matrix, dtype=float).reshape(4, 4)
    u, _, vt = np.linalg.svd(m[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = m[:3, 3]
    return out


def _rotation(axis_index: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis_index]
    r = np.eye(4)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


class PlanSession:
    """One case being planned: the bone, the undo history, caches."""

    def __init__(self, bone: trimesh.Trimesh, plan: FemGuidePlan,
                 limits: ValidationLimits = ValidationLimits()) -> None:
        if mesh_sha256(bone) != plan.bone_sha256:
            raise ValueError("bone mesh does not match the plan's bone_sha256")
        self.bone = bone
        self.limits = limits
        self.history: PlanHistory[FemGuidePlan] = PlanHistory(plan)
        self._base_cache: Dict = {}

    @classmethod
    def start(cls, bone: trimesh.Trimesh, case_id: str,
              limits: ValidationLimits = ValidationLimits()) -> "PlanSession":
        """New case on a (repaired) bone, starting from the prototype's cut."""
        plan = FemGuidePlan(
            case_id=case_id, bone_sha256=mesh_sha256(bone),
            cutting=CuttingParams(angles_deg=PROTOTYPE_ANGLES_DEG,
                                  lengths_mm=PROTOTYPE_LENGTHS_MM))
        return cls(bone, plan, limits)

    # -- history -------------------------------------------------------------

    @property
    def plan(self) -> FemGuidePlan:
        return self.history.current

    def commit(self, plan: FemGuidePlan) -> FemGuidePlan:
        if plan.bone_sha256 != self.plan.bone_sha256:
            raise ValueError("a session cannot switch to a different bone")
        return self.history.commit(plan)

    def undo(self) -> FemGuidePlan:
        return self.history.undo()

    def redo(self) -> FemGuidePlan:
        return self.history.redo()

    # -- cutting structure ---------------------------------------------------

    def set_cutting(self, **changes) -> FemGuidePlan:
        """Change cutting parameters. Changing the segment count without new
        extensions resets the extensions to zero."""
        c = self.plan.cutting
        if "angles_deg" in changes and len(changes["angles_deg"]) != len(c.angles_deg):
            changes.setdefault("start_ext_mm", ())
            changes.setdefault("end_ext_mm", ())
        return self.commit(dataclasses.replace(
            self.plan, cutting=dataclasses.replace(c, **changes)))

    def set_placement(self, matrix) -> FemGuidePlan:
        return self.set_cutting(placement=tuple(orthonormalized(matrix).ravel()))

    def polyline_center_local(self) -> np.ndarray:
        c = self.plan.cutting
        pts = polyline_from_angles(c.angles_deg, c.lengths_mm)
        return np.append((pts.min(axis=0) + pts.max(axis=0)) / 2.0, 1.0)

    def nudge(self, translate_mm: Sequence[float] = (0.0, 0.0, 0.0),
              rotate_deg: Sequence[float] = (0.0, 0.0, 0.0)) -> FemGuidePlan:
        """Move the cutting structure along its own x/y/z axes and rotate it
        about them (in x, y, z order) around the middle of the cut line."""
        m = self.plan.cutting.placement_matrix
        center = self.polyline_center_local()[:3]
        to_center, back = np.eye(4), np.eye(4)
        to_center[:3, 3], back[:3, 3] = center, -center
        local = np.eye(4)
        local[:3, 3] = np.asarray(translate_mm, dtype=float)
        for axis, deg in enumerate(rotate_deg):
            if deg:
                local = local @ to_center @ _rotation(axis, float(deg)) @ back
        return self.set_placement(m @ local)

    def drop_onto_base(self) -> FemGuidePlan:
        """Place the cutting structure over the middle of the base outline:
        local z along the bone's outward normal there, the cut line's long
        axis along the outline's long axis, the block just above the base.
        Auto-connect then joins it to the base."""
        base = self.plan.base
        if base is None:
            raise ValueError("draw the base outline first")
        outline = np.array(base.outline_points)
        center = snap_points_to_surface(self.bone, outline.mean(axis=0)[None, :])[0]
        z = surface_normal(self.bone, center)
        spread = outline - outline.mean(axis=0)
        _, _, vt = np.linalg.svd(spread - np.outer(spread @ z, z))
        x = safe_normalize(vt[0] - np.dot(vt[0], z) * z)
        y = np.cross(z, x)
        frame = np.eye(4)
        frame[:3, 0], frame[:3, 1], frame[:3, 2] = x, y, z
        height = base.clearance_mm + base.thickness_mm + _DROP_GAP_MM
        height += self.plan.cutting.slot_depth_mm / 2.0
        frame[:3, 3] = center + z * height
        frame = frame @ _translation(-self.polyline_center_local()[:3])
        return self.set_placement(frame)

    # -- base ----------------------------------------------------------------

    def set_outline(self, points: Iterable[Sequence[float]]) -> FemGuidePlan:
        pts = tuple(tuple(float(v) for v in p) for p in points)
        base = (dataclasses.replace(self.plan.base, outline_points=pts)
                if self.plan.base is not None else BaseParams(outline_points=pts))
        return self.commit(dataclasses.replace(self.plan, base=base))

    def set_base(self, **changes) -> FemGuidePlan:
        if self.plan.base is None:
            raise ValueError("draw the base outline first")
        return self.commit(dataclasses.replace(
            self.plan, base=dataclasses.replace(self.plan.base, **changes)))

    def clear_base(self) -> FemGuidePlan:
        return self.commit(dataclasses.replace(self.plan, base=None))

    # -- pins ----------------------------------------------------------------

    def _set_pins(self, pins) -> FemGuidePlan:
        return self.commit(dataclasses.replace(self.plan, pins=tuple(pins)))

    def add_pin(self, position, direction=None, diameter_mm: Optional[float] = None) -> FemGuidePlan:
        kwargs = {} if diameter_mm is None else {"diameter_mm": diameter_mm}
        return self._set_pins(self.plan.pins + (
            Pin(position=tuple(position), direction=direction, **kwargs),))

    def set_pin(self, index: int, **changes) -> FemGuidePlan:
        pins = list(self.plan.pins)
        pins[index] = dataclasses.replace(pins[index], **changes)
        return self._set_pins(pins)

    def remove_pin(self, index: int) -> FemGuidePlan:
        pins = list(self.plan.pins)
        del pins[index]
        return self._set_pins(pins)

    def set_pin_positions(self, positions) -> FemGuidePlan:
        """Match the pins to a list of positions (the point list in the 3D
        view): existing pins keep their settings, extra positions become new
        auto-direction pins, missing ones are removed."""
        positions = [tuple(float(v) for v in p) for p in positions]
        pins = []
        for i, position in enumerate(positions):
            if i < len(self.plan.pins):
                pins.append(dataclasses.replace(self.plan.pins[i], position=position))
            else:
                pins.append(Pin(position=position))
        return self._set_pins(pins)

    def align_pin_with_cut(self, index: int) -> FemGuidePlan:
        """Point a pin along the blade's direction of travel (local -z of the
        cutting structure). The cut planes contain that direction, so a pin
        beside the slot then runs parallel to the cut and cannot cross it,
        however deep it goes. Auto-direction pins on a rounded bone aim
        inward and often do cross it."""
        return self.align_pins_with_cut([index])

    def align_pins_with_cut(self, indices: Optional[Iterable[int]] = None) -> FemGuidePlan:
        """Align several pins (default: all) as one undoable edit."""
        direction = -self.plan.cutting.placement_matrix[:3, 2]
        pins = list(self.plan.pins)
        for index in (range(len(pins)) if indices is None else indices):
            inward = -surface_normal(self.bone, pins[index].position)
            if float(np.dot(direction, inward)) <= 0.0:
                raise ValueError("the cut does not face pin {}'s bone surface; move "
                                 "the cut or set the pin angles by hand".format(index + 1))
            pins[index] = dataclasses.replace(
                pins[index], direction=tuple(float(v) for v in direction))
        return self._set_pins(pins)

    def _pin_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inward normal at the pin, and two tangent axes: ``u`` along the
        cutting structure's local x projected onto the tangent plane."""
        pin = self.plan.pins[index]
        inward = -surface_normal(self.bone, pin.position)
        ref = self.plan.cutting.placement_matrix[:3, 0]
        u = ref - np.dot(ref, inward) * inward
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(inward, [1.0, 0.0, 0.0])
        u = safe_normalize(u)
        return inward, u, np.cross(inward, u)

    def pin_angles(self, index: int) -> Tuple[float, float]:
        """(tilt from the inward surface normal, azimuth from the cutting
        structure's x axis), in degrees: the readout for the pin panel."""
        direction = resolve_pin_direction(self.plan.pins[index], self.bone)
        inward, u, v = self._pin_frame(index)
        tilt = math.degrees(math.acos(float(np.clip(np.dot(direction, inward), -1.0, 1.0))))
        azimuth = math.degrees(math.atan2(float(np.dot(direction, v)), float(np.dot(direction, u))))
        return tilt, azimuth

    def set_pin_angles(self, index: int, tilt_deg: float, azimuth_deg: float) -> FemGuidePlan:
        inward, u, v = self._pin_frame(index)
        t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
        direction = (math.cos(t) * inward
                     + math.sin(t) * (math.cos(a) * u + math.sin(a) * v))
        return self.set_pin(index, direction=tuple(float(x) for x in direction))

    # -- results -------------------------------------------------------------

    def regenerate(self) -> GuideGeometry:
        return regenerate(self.plan, self.bone, base_cache=self._base_cache)

    def validate(self, geometry: Optional[GuideGeometry] = None) -> ValidationReport:
        if geometry is None:
            geometry = self.regenerate()
        return validate(self.plan, geometry, self.bone, self.limits)

    def export(self, stl_path, acknowledged: Iterable[str] = ()):
        return export_guide(self.plan, self.bone, stl_path, acknowledged, self.limits)


def _translation(offset) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(offset, dtype=float)
    return m
