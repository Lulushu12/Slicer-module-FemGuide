"""FemGuide export: validated, traceable STL (docs/ARCHITECTURE.md 4.3, 5.1).

``export_guide`` takes the plan and the bone, not ready-made geometry or a
report: it regenerates and validates itself, so what is exported, what the
gate judged, and what the record claims can never come from different plans.

``verify_reproducible`` checks D4's promise that any export can be rebuilt
exactly from its traceability record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Union

import trimesh

from femguide_core.export import (
    ExportBlockedError,
    export_traceable,
    file_sha256,
    load_trace,
)

from femguide import __version__
from femguide.plan import FemGuidePlan
from femguide.regenerate import regenerate
from femguide.validate import ValidationLimits, validate

MODULE_NAME = "femguide"


def export_guide(plan: FemGuidePlan, bone: trimesh.Trimesh,
                 stl_path: Union[str, Path], acknowledged: Iterable[str] = (),
                 limits: ValidationLimits = ValidationLimits()) -> Path:
    """Regenerate, validate and export the guide STL plus its traceability
    record (returns the record path). Raises ``ExportBlockedError`` if the
    validation gate says no; the error's ``report`` attribute has details."""
    geometry = regenerate(plan, bone)
    report = validate(plan, geometry, bone, limits)
    try:
        return export_traceable(
            geometry.guide, stl_path, plan=plan.to_dict(), module=MODULE_NAME,
            module_version=__version__, case_id=plan.case_id,
            bone_sha256=plan.bone_sha256, report=report,
            acknowledged=acknowledged,
            extra={"validation_limits": limits.to_dict()})
    except ExportBlockedError as exc:
        exc.report = report
        raise


@dataclass
class Reproduction:
    ok: bool
    problems: List[str] = field(default_factory=list)


def verify_reproducible(record_path: Union[str, Path], bone: trimesh.Trimesh,
                        scratch_stl: Union[str, Path]) -> Reproduction:
    """Rebuild an export from its record and compare.

    Checks that the exported STL on disk is the one the record describes,
    that regenerating from the recorded plan gives a byte-identical STL, and
    that re-validating with the recorded limits gives the recorded findings.
    ``scratch_stl`` is where the regenerated STL is written. Only meaningful
    with the software and dependency versions the record lists.
    """
    record_path = Path(record_path)
    record = load_trace(record_path)
    problems = []

    on_disk = record_path.parent / record["stl_file"]
    if not on_disk.exists() or file_sha256(on_disk) != record["stl_sha256"]:
        problems.append("the exported STL on disk does not match the record")

    plan = FemGuidePlan.from_dict(record["plan"])
    geometry = regenerate(plan, bone)
    Path(scratch_stl).write_bytes(geometry.guide.export(file_type="stl"))
    if file_sha256(scratch_stl) != record["stl_sha256"]:
        problems.append("regenerating the recorded plan gives a different STL")

    limits = ValidationLimits(**record["validation_limits"])
    report = validate(plan, geometry, bone, limits)
    findings = sorted((f["code"], f["severity"]) for f in record["validation"]["findings"])
    if sorted((f.code, f.severity.value) for f in report.findings) != findings:
        problems.append("re-validating gives different findings than recorded")
    return Reproduction(ok=not problems, problems=problems)
