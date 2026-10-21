"""Traceable export (docs/ARCHITECTURE.md section 4.3).

Every printed guide must be traceable to the exact software state and inputs
that produced it. An export is the STL plus a JSON traceability record
written next to it (``<name>.stl`` + ``<name>.trace.json``) holding the
complete plan snapshot, the software and dependency versions, hashes of the
bone mesh, plan and STL, the case ID, a UTC timestamp, and the validation
report with the warnings the user acknowledged.

STL cannot carry metadata, so the record is a sidecar file. Embedding it in
3MF is a planned follow-up; the record format does not change.

Export refuses to run unless the validation gate passes (section 5.3).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

import numpy as np
import trimesh

from femguide_core import __version__ as core_version
from femguide_core.validation import ValidationReport

TRACE_FORMAT = "femguide-trace/1"


class ExportBlockedError(RuntimeError):
    """The validation gate does not allow this export."""


def mesh_sha256(mesh: trimesh.Trimesh) -> str:
    """Content hash of a mesh's geometry (vertex and face arrays).

    Used to tie a plan to the exact bone mesh it was made on: a plan
    regenerated against a different mesh would silently produce a different
    guide.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def json_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dependency_versions() -> Dict[str, str]:
    import manifold3d

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "trimesh": trimesh.__version__,
        "manifold3d": getattr(manifold3d, "__version__", "unknown"),
    }


def export_traceable(
    mesh: trimesh.Trimesh,
    stl_path: Union[str, Path],
    *,
    plan: Mapping[str, Any],
    module: str,
    module_version: str,
    case_id: str,
    bone_sha256: str,
    report: ValidationReport,
    acknowledged: Iterable[str] = (),
) -> Path:
    """Write ``mesh`` as binary STL plus its traceability record.

    Raises :class:`ExportBlockedError` (writing nothing) if the report has
    blocking findings or unacknowledged warnings. Only acknowledgements that
    match an actual warning are recorded. Returns the record's path.
    """
    acknowledged = sorted(set(acknowledged))
    if report.blocks:
        raise ExportBlockedError(
            "export blocked by validation: "
            + "; ".join("{}: {}".format(f.code, f.message) for f in report.blocks)
        )
    missing = report.unacknowledged(acknowledged)
    if missing:
        raise ExportBlockedError(
            "export requires acknowledging these warnings first: "
            + "; ".join("{}: {}".format(f.code, f.message) for f in missing)
        )

    stl_path = Path(stl_path)
    if stl_path.suffix.lower() != ".stl":
        raise ValueError("stl_path must end in .stl")
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(stl_path), file_type="stl")

    warning_codes = {f.code for f in report.warnings}
    record = {
        "format": TRACE_FORMAT,
        "exported_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "case_id": case_id,
        "module": module,
        "module_version": module_version,
        "core_version": core_version,
        "dependencies": _dependency_versions(),
        "stl_file": stl_path.name,
        "stl_sha256": file_sha256(stl_path),
        "bone_sha256": bone_sha256,
        "plan_sha256": json_sha256(dict(plan)),
        "plan": dict(plan),
        "validation": report.to_dict(),
        "acknowledged_warnings": [c for c in acknowledged if c in warning_codes],
    }
    record_path = stl_path.with_suffix(".trace.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record_path


def load_trace(record_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a traceability record, checking its format tag."""
    record = json.loads(Path(record_path).read_text())
    if record.get("format") != TRACE_FORMAT:
        raise ValueError(
            "{} is not a {} record".format(record_path, TRACE_FORMAT))
    return record
