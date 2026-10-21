"""FemGuide export: validated, traceable STL (docs/ARCHITECTURE.md 4.3, 5.1).

Wraps the core traceable export with FemGuide's plan, and provides the
reproducibility check D4 promises: any export can be regenerated exactly
from its traceability record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import trimesh

from femguide_core.export import export_traceable, file_sha256, load_trace
from femguide_core.validation import ValidationReport

from femguide import __version__
from femguide.plan import FemGuidePlan
from femguide.regenerate import GuideGeometry, regenerate

MODULE_NAME = "femguide"


def export_guide(plan: FemGuidePlan, geometry: GuideGeometry,
                 report: ValidationReport, stl_path: Union[str, Path],
                 acknowledged: Iterable[str] = ()) -> Path:
    """Export the guide STL and its traceability record (returns the record
    path). Raises ``ExportBlockedError`` if the validation gate says no."""
    return export_traceable(
        geometry.guide, stl_path, plan=plan.to_dict(), module=MODULE_NAME,
        module_version=__version__, case_id=plan.case_id,
        bone_sha256=plan.bone_sha256, report=report, acknowledged=acknowledged)


def verify_reproducible(record_path: Union[str, Path], bone: trimesh.Trimesh,
                        scratch_stl: Union[str, Path]) -> bool:
    """Regenerate the guide from a traceability record and check the STL is
    byte-identical to the one that was exported.

    ``scratch_stl`` is where the regenerated STL is written for comparison.
    Only meaningful with the same software and dependency versions the record
    lists (geometry kernels are not guaranteed bit-stable across versions).
    """
    record = load_trace(record_path)
    plan = FemGuidePlan.from_dict(record["plan"])
    geometry = regenerate(plan, bone)
    geometry.guide.export(str(scratch_stl), file_type="stl")
    return file_sha256(scratch_stl) == record["stl_sha256"]
