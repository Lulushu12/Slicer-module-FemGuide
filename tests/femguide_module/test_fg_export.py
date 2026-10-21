"""FemGuide export: gating, traceability record, exact reproducibility."""

import json

import pytest

from femguide_core.export import ExportBlockedError, load_trace
from femguide.export import export_guide, verify_reproducible
from femguide.plan import FemGuidePlan
from femguide.regenerate import regenerate
from femguide.validate import validate

from fg_scenario import with_cutting


def _run(plan, bone):
    g = regenerate(plan, bone)
    return g, validate(plan, g, bone)


def test_export_record_reproduces_the_exact_stl(plan, bone, tmp_path):
    g, report = _run(plan, bone)
    record_path = export_guide(plan, g, report, tmp_path / "TEST-001.stl")
    record = load_trace(record_path)
    assert FemGuidePlan.from_dict(record["plan"]) == plan
    assert record["module"] == "femguide"
    assert record["bone_sha256"] == plan.bone_sha256
    assert verify_reproducible(record_path, bone, tmp_path / "again.stl")


def test_tampered_record_does_not_reproduce(plan, bone, tmp_path):
    g, report = _run(plan, bone)
    record_path = export_guide(plan, g, report, tmp_path / "TEST-001.stl")
    record = json.loads(record_path.read_text())
    record["plan"]["cutting"]["wall_thickness_mm"] = 3.0
    record_path.write_text(json.dumps(record))
    assert not verify_reproducible(record_path, bone, tmp_path / "again.stl")


def test_blocked_guide_cannot_export(plan, bone, tmp_path):
    bad = with_cutting(plan, wall_thickness_mm=0.8)
    g, report = _run(bad, bone)
    with pytest.raises(ExportBlockedError, match="WALL_TOO_THIN"):
        export_guide(bad, g, report, tmp_path / "x.stl", acknowledged=["WALL_TOO_THIN"])


def test_warning_needs_acknowledgement(plan, bone, tmp_path):
    thin = with_cutting(plan, wall_thickness_mm=1.5)
    g, report = _run(thin, bone)
    with pytest.raises(ExportBlockedError):
        export_guide(thin, g, report, tmp_path / "x.stl")
    record = load_trace(export_guide(thin, g, report, tmp_path / "x.stl",
                                     acknowledged=["WALL_THIN"]))
    assert record["acknowledged_warnings"] == ["WALL_THIN"]
