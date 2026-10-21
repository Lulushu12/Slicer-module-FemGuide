"""FemGuide checks for the export validation gate (docs/ARCHITECTURE.md 5.3).

Hard blocks: broken geometry; walls or base below the printable minimum;
guide material intruding into the bone (it could never seat); a pin hole
intersecting the slot; features outside the guide (a pin hole through no
material, or parts that are not one connected body).

Warnings needing acknowledgement: thin walls or base; marginal blade or
bone-fit clearance; no patient-specific base.

The numeric limits are printer- and material-dependent. The defaults below
are placeholders to be replaced with values validated on the actual printer
and material; they are not clinically established.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

from femguide_core.geometry import (
    NonManifoldError,
    intersection,
    is_manifold_solid,
    to_manifold,
)
from femguide_core.validation import Severity, ValidationReport

from femguide.plan import FemGuidePlan
from femguide.regenerate import GuideGeometry

# Volumes below this (mm^3) count as "no overlap" (boolean round-off).
_OVERLAP_EPS_MM3 = 1e-3


@dataclass(frozen=True)
class ValidationLimits:
    """Thresholds in mm. PLACEHOLDER DEFAULTS: set per printer/material."""

    min_printable_mm: float = 1.0
    recommended_min_mm: float = 2.0
    min_blade_clearance_mm: float = 0.1
    min_bone_clearance_mm: float = 0.2


def _overlap_volume(a, b) -> float:
    result = intersection(a, b)
    return float(result.volume) if len(result.faces) else 0.0


def validate(plan: FemGuidePlan, geometry: GuideGeometry, bone: trimesh.Trimesh,
             limits: ValidationLimits = ValidationLimits()) -> ValidationReport:
    report = ValidationReport()
    guide = geometry.guide
    c = plan.cutting

    # -- geometry integrity -------------------------------------------------
    if len(guide.faces) == 0 or not guide.is_watertight or not is_manifold_solid(guide):
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
        intrusion = _overlap_volume(geometry.material, bone)
    except NonManifoldError:
        report.add("BONE_CHECK_UNAVAILABLE", Severity.WARN,
                   "The bone mesh is not a closed solid, so the guide could not "
                   "be checked for intruding into the bone.")
    else:
        report.metrics["bone_intrusion_mm3"] = intrusion
        if intrusion > _OVERLAP_EPS_MM3:
            report.add("GUIDE_INTERSECTS_BONE", Severity.BLOCK,
                       "Guide material intrudes {:.1f} mm^3 into the bone; the "
                       "guide could not seat. Move the cutting structure away "
                       "from the bone.".format(intrusion))

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

    # -- clearances ---------------------------------------------------------
    if c.blade_clearance_mm < limits.min_blade_clearance_mm:
        report.add("BLADE_CLEARANCE_MARGINAL", Severity.WARN,
                   "Blade clearance {:.2f} mm is below {:.2f} mm; the blade may "
                   "bind in the slot.".format(c.blade_clearance_mm,
                                              limits.min_blade_clearance_mm))
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

    # -- pins ---------------------------------------------------------------
    if geometry.pin_holes:
        material = geometry.material
        for i, hole in enumerate(geometry.pin_holes):
            if _overlap_volume(hole, geometry.slot) > _OVERLAP_EPS_MM3:
                report.add("PIN_INTERSECTS_SLOT", Severity.BLOCK,
                           "Pin {} passes through the cutting slot.".format(i + 1))
            if _overlap_volume(hole, material) <= _OVERLAP_EPS_MM3:
                report.add("PIN_OUTSIDE_GUIDE", Severity.BLOCK,
                           "Pin {} does not pass through any guide "
                           "material.".format(i + 1))
    return report
