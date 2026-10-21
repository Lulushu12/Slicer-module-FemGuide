"""FemGuide checks for the export validation gate (docs/ARCHITECTURE.md 5.3).

Hard blocks:
- broken geometry, or parts not connected into one body;
- the bone mesh not a closed solid (the guide cannot be checked against it);
- guide material intruding into the bone (it could never seat);
- guide material left in the blade's path (defence in depth: the slot is
  built as the blade's full path, so this should never fire);
- walls or base below the printable minimum, including the real material
  between slot passes that come close together and at sharp bends;
- the cutting structure attached to the base by too little material;
- pins: not on the bone surface, not pointing into the bone, crossing the
  blade's path (in the guide or in the bone), through no material, or with
  less than the printable minimum of material around the hole.

Warnings needing acknowledgement: thin walls, base or slot-pass gaps; blade
or bone-fit clearance too tight or too loose; no patient-specific base.

The numeric limits are printer-, material- and instrument-dependent. The
defaults below are placeholders to be replaced with values validated on the
actual printer, material and pin/saw sets; they are not clinically
established.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import trimesh
from manifold3d import Manifold, OpType

from femguide_core.geometry import (
    NonManifoldError,
    difference,
    is_manifold_solid,
    polyline_from_angles,
    safe_normalize,
    snap_points_to_surface,
    to_manifold,
)
from femguide_core.validation import Severity, ValidationReport

from femguide.plan import FemGuidePlan
from femguide.regenerate import (
    PIN_INTO_BONE_MM,
    PIN_OUT_OF_BONE_MM,
    GuideGeometry,
    pin_hole,
)

# Volumes below this (mm^3) count as "no overlap" (boolean round-off).
_OVERLAP_EPS_MM3 = 1e-3
# Pin wall check: needles parallel to the pin at the required wall distance
# must each pass through at least this fraction of the material the pin's
# own axis passes through (curvature makes lengths slightly uneven).
_PIN_WALL_FRACTION = 0.75
_PIN_WALL_NEEDLES = 12
_NEEDLE_WIDTH_MM = 0.05


@dataclass(frozen=True)
class ValidationLimits:
    """Thresholds. PLACEHOLDER DEFAULTS: set per printer, material and
    instrument set before clinical use."""

    min_printable_mm: float = 1.0
    recommended_min_mm: float = 2.0
    min_blade_clearance_mm: float = 0.1
    max_blade_clearance_mm: float = 0.5
    min_bone_clearance_mm: float = 0.2
    max_bone_clearance_mm: float = 1.0
    min_attachment_mm3: float = 30.0
    # How deep a fixation pin may go into the bone; the pin must not cross
    # the blade's path anywhere along that depth.
    pin_depth_in_bone_mm: float = 30.0
    # How far a pin's entry point may sit from the bone surface.
    pin_surface_tolerance_mm: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Volume of ``a`` intersected with ``b``, computed by manifold3d itself
    (exact for the zero-volume slivers that coincident faces produce, where
    trimesh's mass properties divide by zero)."""
    common = Manifold.batch_boolean([to_manifold(a, "a"), to_manifold(b, "b")],
                                    OpType.Intersect)
    return max(0.0, float(common.volume()))


def _segment_distance_2d(a0, a1, b0, b1, samples: int = 64) -> float:
    """Minimum distance between two 2D segments (dense sampling of one
    against exact point-to-segment distance to the other, both ways)."""
    def point_seg(p, s0, s1):
        d = s1 - s0
        t = np.clip(((p - s0) @ d) / max(float(d @ d), 1e-18), 0.0, 1.0)
        return np.linalg.norm(p - (s0 + np.outer(t, d)), axis=1)

    t = np.linspace(0.0, 1.0, samples)[:, None]
    pa = a0 + t * (a1 - a0)
    pb = b0 + t * (b1 - b0)
    return float(min(point_seg(pa, b0, b1).min(), point_seg(pb, a0, a1).min()))


def _check_slot_layout(plan: FemGuidePlan, limits: ValidationLimits,
                       report: ValidationReport) -> None:
    """Real material between slot passes, measured on the plan's polyline.

    Non-adjacent segments (a U-turn): the gap between the two channels is
    their centreline distance minus the slot width. Adjacent segments always
    taper to nothing at the inner corner of a bend; that corner is supported
    material, so what matters is the fin's thickness at the end of the
    shorter segment: a fold-back bend leaves a fin that is thin all along.
    """
    c = plan.cutting
    pts = polyline_from_angles(c.angles_deg, c.lengths_mm)[:, :2]
    segs = []
    for i in range(len(pts) - 1):
        u = safe_normalize(np.append(pts[i + 1] - pts[i], 0.0))[:2]
        segs.append((pts[i] - u * c.start_ext_mm[i], pts[i + 1] + u * c.end_ext_mm[i]))
    thinnest = np.inf
    for i, j in itertools.combinations(range(len(segs)), 2):
        if j == i + 1:
            len_i = np.linalg.norm(pts[i + 1] - pts[i])
            len_j = np.linalg.norm(pts[j + 1] - pts[j])
            far, other = ((pts[i], (pts[j], pts[j + 1])) if len_i <= len_j
                          else (pts[j + 1], (pts[i], pts[i + 1])))
            gap = _segment_distance_2d(far, far, *other) - c.slot_width_mm
        else:
            gap = _segment_distance_2d(*segs[i], *segs[j]) - c.slot_width_mm
        thinnest = min(thinnest, gap)
    if not np.isfinite(thinnest):
        return
    report.metrics["thinnest_slot_gap_mm"] = float(thinnest)
    if thinnest < limits.min_printable_mm:
        report.add("SLOT_PASSES_TOO_CLOSE", Severity.BLOCK,
                   "Parts of the slot come within {:.2f} mm of each other (printable "
                   "minimum {:.2f} mm); the wall between them would be too thin or "
                   "missing, and the blade may not be captured.".format(
                       thinnest, limits.min_printable_mm))
    elif thinnest < limits.recommended_min_mm:
        report.add("SLOT_PASSES_CLOSE", Severity.WARN,
                   "Parts of the slot come within {:.2f} mm of each other "
                   "(recommended {:.2f} mm).".format(thinnest, limits.recommended_min_mm))


def _check_pins(plan: FemGuidePlan, geometry: GuideGeometry, bone: trimesh.Trimesh,
                limits: ValidationLimits, report: ValidationReport) -> None:
    holed = difference(geometry.material, geometry.slot)  # material minus slot
    positions = np.array([p.position for p in plan.pins], dtype=float)
    snapped = snap_points_to_surface(bone, positions)
    for i, (pin, direction) in enumerate(zip(plan.pins, geometry.pin_directions)):
        label = "Pin {}".format(i + 1)
        position = positions[i]
        if np.linalg.norm(snapped[i] - position) > limits.pin_surface_tolerance_mm:
            report.add("PIN_NOT_ON_BONE", Severity.BLOCK,
                       "{} entry point is {:.2f} mm from the bone surface.".format(
                           label, np.linalg.norm(snapped[i] - position)))
        nearest = int(np.argmin(np.linalg.norm(bone.triangles_center - snapped[i], axis=1)))
        inward = -np.asarray(bone.face_normals[nearest], dtype=float)
        if float(np.dot(direction, inward)) <= 0.0:
            report.add("PIN_POINTS_OUT", Severity.BLOCK,
                       "{} points out of the bone instead of into it.".format(label))

        # The pin, as deep as it may go into the bone, must stay clear of the
        # blade's whole path.
        reach = pin_hole(pin, direction, into_bone_mm=limits.pin_depth_in_bone_mm)
        if _overlap(reach, geometry.slot) > _OVERLAP_EPS_MM3:
            report.add("PIN_INTERSECTS_SLOT", Severity.BLOCK,
                       "{} crosses the blade's path; the saw would hit it.".format(label))

        r = pin.diameter_mm / 2.0
        m = limits.min_printable_mm
        axis_length = _material_along(holed, position, direction)
        if axis_length <= 1e-6:
            report.add("PIN_OUTSIDE_GUIDE", Severity.BLOCK,
                       "{} does not pass through any guide material.".format(label))
            continue
        report.metrics["pin{}_material_mm".format(i + 1)] = axis_length
        u = safe_normalize(np.cross(direction, [1.0, 0.0, 0.0]))
        if np.linalg.norm(np.cross(direction, [1.0, 0.0, 0.0])) < 0.1:
            u = safe_normalize(np.cross(direction, [0.0, 1.0, 0.0]))
        v = np.cross(direction, u)
        for k in range(_PIN_WALL_NEEDLES):
            a = 2.0 * np.pi * k / _PIN_WALL_NEEDLES
            offset = (r + m) * (np.cos(a) * u + np.sin(a) * v)
            if (_material_along(holed, position + offset, direction)
                    < _PIN_WALL_FRACTION * axis_length):
                report.add("PIN_AT_EDGE", Severity.BLOCK,
                           "{} has less than {:.2f} mm of material around it on at "
                           "least one side (too close to an edge or the slot, or "
                           "breaking out).".format(label, m))
                break


def _material_along(solid: trimesh.Trimesh, point, direction) -> float:
    """Length of material a thin needle through ``point`` along ``direction``
    passes through (the needle spans the same range as a pin hole)."""
    length = PIN_INTO_BONE_MM + PIN_OUT_OF_BONE_MM
    needle = trimesh.creation.box(extents=(_NEEDLE_WIDTH_MM, _NEEDLE_WIDTH_MM, length))
    frame = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    frame[:3, 3] = (np.asarray(point, dtype=float)
                    + direction * (PIN_INTO_BONE_MM - PIN_OUT_OF_BONE_MM) / 2.0)
    needle.apply_transform(frame)
    return _overlap(needle, solid) / (_NEEDLE_WIDTH_MM ** 2)


def validate(plan: FemGuidePlan, geometry: GuideGeometry, bone: trimesh.Trimesh,
             limits: ValidationLimits = ValidationLimits()) -> ValidationReport:
    report = ValidationReport()
    guide = geometry.guide
    c = plan.cutting

    # -- geometry integrity -------------------------------------------------
    guide_ok = (len(guide.faces) > 0 and guide.is_watertight
                and is_manifold_solid(guide))
    if not guide_ok:
        report.add("GEOMETRY_INVALID", Severity.BLOCK,
                   "The guide geometry is broken (not a closed, manifold solid) "
                   "and cannot be printed.")
    else:
        bodies = len(to_manifold(guide).decompose())
        report.metrics["bodies"] = float(bodies)
        report.metrics["volume_mm3"] = float(guide.volume)
        if bodies != 1:
            report.add("DISCONNECTED_PARTS", Severity.BLOCK,
                       "The guide falls apart into {} separate pieces; the cutting "
                       "structure must overlap the base.".format(bodies))

    # -- seating ------------------------------------------------------------
    try:
        to_manifold(bone, "bone")
    except NonManifoldError:
        report.add("BONE_NOT_SOLID", Severity.BLOCK,
                   "The bone mesh is not a closed solid, so the guide cannot be "
                   "checked against it. Repair the bone mesh first.")
    else:
        intrusion = _overlap(geometry.material, bone)
        report.metrics["bone_intrusion_mm3"] = intrusion
        if intrusion > _OVERLAP_EPS_MM3:
            report.add("GUIDE_INTERSECTS_BONE", Severity.BLOCK,
                       "Guide material intrudes {:.1f} mm^3 into the bone; the "
                       "guide could not seat. Move the cutting structure away "
                       "from the bone.".format(intrusion))

    if guide_ok:
        obstruction = _overlap(guide, geometry.slot)
        if obstruction > _OVERLAP_EPS_MM3:
            report.add("SLOT_OBSTRUCTED", Severity.BLOCK,
                       "{:.2f} mm^3 of guide material is in the blade's path.".format(
                           obstruction))

    # -- thickness ----------------------------------------------------------
    for code, label, value in (("WALL", "Slot wall", c.wall_thickness_mm),) + (
            (("BASE", "Base", plan.base.thickness_mm),) if plan.base else ()):
        if value < limits.min_printable_mm:
            report.add(code + "_TOO_THIN", Severity.BLOCK,
                       "{} thickness {:.2f} mm is below the printable minimum "
                       "{:.2f} mm.".format(label, value, limits.min_printable_mm))
        elif value < limits.recommended_min_mm:
            report.add(code + "_THIN", Severity.WARN,
                       "{} thickness {:.2f} mm is below the recommended "
                       "{:.2f} mm.".format(label, value, limits.recommended_min_mm))
    _check_slot_layout(plan, limits, report)

    # -- attachment ---------------------------------------------------------
    if geometry.base is not None:
        attachment = _overlap(geometry.walls, geometry.base)
        report.metrics["attachment_mm3"] = attachment
        if attachment < limits.min_attachment_mm3 and not report.blocks_with(
                "DISCONNECTED_PARTS"):
            report.add("WEAK_ATTACHMENT", Severity.BLOCK,
                       "The cutting structure overlaps the base by only {:.1f} mm^3 "
                       "(minimum {:.1f}); it could break off.".format(
                           attachment, limits.min_attachment_mm3))

    # -- clearances ---------------------------------------------------------
    if c.blade_clearance_mm < limits.min_blade_clearance_mm:
        report.add("BLADE_CLEARANCE_MARGINAL", Severity.WARN,
                   "Blade clearance {:.2f} mm is below {:.2f} mm; the blade may "
                   "bind in the slot.".format(c.blade_clearance_mm,
                                              limits.min_blade_clearance_mm))
    elif c.blade_clearance_mm > limits.max_blade_clearance_mm:
        report.add("BLADE_CLEARANCE_LOOSE", Severity.WARN,
                   "Blade clearance {:.2f} mm is above {:.2f} mm; the blade can "
                   "wobble and the cut will be less accurate.".format(
                       c.blade_clearance_mm, limits.max_blade_clearance_mm))
    if plan.base is None:
        report.add("NO_BASE", Severity.WARN,
                   "No patient-specific base: the guide will not seat in a "
                   "unique position on the bone.")
    elif plan.base.clearance_mm < 0.0:
        report.add("BONE_INTERFERENCE", Severity.BLOCK,
                   "Bone-fit clearance {:.2f} mm is negative: the base would "
                   "cut into the bone.".format(plan.base.clearance_mm))
    elif plan.base.clearance_mm < limits.min_bone_clearance_mm:
        report.add("BONE_CLEARANCE_MARGINAL", Severity.WARN,
                   "Bone-fit clearance {:.2f} mm is below {:.2f} mm; the guide "
                   "may not seat.".format(plan.base.clearance_mm,
                                          limits.min_bone_clearance_mm))
    elif plan.base.clearance_mm > limits.max_bone_clearance_mm:
        report.add("BONE_CLEARANCE_LOOSE", Severity.WARN,
                   "Bone-fit clearance {:.2f} mm is above {:.2f} mm; the guide "
                   "may rock on the bone.".format(plan.base.clearance_mm,
                                                  limits.max_bone_clearance_mm))

    # -- pins ---------------------------------------------------------------
    if plan.pins:
        _check_pins(plan, geometry, bone, limits, report)
    return report
