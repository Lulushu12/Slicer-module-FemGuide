"""Traceable export (core): gating, record contents, hashes."""

import json

import pytest
import trimesh

from femguide_core import __version__ as core_version
from femguide_core.export import (
    TRACE_FORMAT,
    ExportBlockedError,
    export_traceable,
    file_sha256,
    json_sha256,
    load_trace,
    mesh_sha256,
)
from femguide_core.validation import Severity, ValidationReport

PLAN = {"case_id": "C1", "value": 3.0}


def _export(tmp_path, report, acknowledged=()):
    return export_traceable(
        trimesh.creation.box(extents=(1, 2, 3)), tmp_path / "guide.stl",
        plan=PLAN, module="test", module_version="9.9", case_id="C1",
        bone_sha256="abc", report=report, acknowledged=acknowledged)


def test_blocked_export_writes_nothing(tmp_path):
    report = ValidationReport()
    report.add("BAD", Severity.BLOCK, "broken")
    with pytest.raises(ExportBlockedError, match="BAD"):
        _export(tmp_path, report, acknowledged=["BAD"])
    assert list(tmp_path.iterdir()) == []


def test_unacknowledged_warning_blocks(tmp_path):
    report = ValidationReport()
    report.add("THIN", Severity.WARN, "thin")
    with pytest.raises(ExportBlockedError, match="THIN"):
        _export(tmp_path, report)
    assert list(tmp_path.iterdir()) == []


def test_record_contents(tmp_path):
    report = ValidationReport()
    report.add("THIN", Severity.WARN, "thin")
    record_path = _export(tmp_path, report, acknowledged=["THIN", "NOT_A_WARNING"])
    assert record_path.name == "guide.trace.json"
    record = load_trace(record_path)
    assert record["format"] == TRACE_FORMAT
    assert record["case_id"] == "C1"
    assert record["module"] == "test" and record["module_version"] == "9.9"
    assert record["core_version"] == core_version
    assert record["plan"] == PLAN
    assert record["plan_sha256"] == json_sha256(PLAN)
    assert record["stl_sha256"] == file_sha256(tmp_path / "guide.stl")
    assert record["bone_sha256"] == "abc"
    assert record["acknowledged_warnings"] == ["THIN"]
    assert {"python", "numpy", "trimesh", "manifold3d"} <= set(record["dependencies"])
    assert record["validation"]["findings"][0]["code"] == "THIN"


def test_load_trace_rejects_foreign_json(tmp_path):
    path = tmp_path / "x.trace.json"
    path.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        load_trace(path)


def test_mesh_sha256_detects_any_change():
    a = trimesh.creation.icosphere(subdivisions=2)
    b = a.copy()
    assert mesh_sha256(a) == mesh_sha256(b)
    b.vertices[0] += 1e-9
    assert mesh_sha256(a) != mesh_sha256(b)


def test_existing_files_are_never_overwritten(tmp_path):
    _export(tmp_path, ValidationReport())
    with pytest.raises(FileExistsError):
        _export(tmp_path, ValidationReport())


def test_extra_fields_recorded_but_cannot_override(tmp_path):
    def run(extra, name):
        return export_traceable(
            trimesh.creation.box(), tmp_path / name, plan=PLAN, module="t",
            module_version="1", case_id="C1", bone_sha256="abc",
            report=ValidationReport(), extra=extra)
    record = load_trace(run({"validation_limits": {"min": 1.0}}, "a.stl"))
    assert record["validation_limits"] == {"min": 1.0}
    with pytest.raises(ValueError):
        run({"case_id": "forged"}, "b.stl")
    assert not (tmp_path / "b.stl").exists()
