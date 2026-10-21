"""FemGuide export: gating, traceability record, exact reproducibility."""

import json

import pytest

from femguide_core.export import ExportBlockedError, load_trace
from femguide.export import export_guide, verify_reproducible
from femguide.plan import FemGuidePlan
from femguide.validate import ValidationLimits

from fg_scenario import with_cutting


def test_export_record_reproduces_the_exact_stl(plan, bone, tmp_path):
    record_path = export_guide(plan, bone, tmp_path / "TEST-001.stl")
    record = load_trace(record_path)
    assert FemGuidePlan.from_dict(record["plan"]) == plan
    assert record["module"] == "femguide"
    assert record["bone_sha256"] == plan.bone_sha256
    assert record["validation_limits"] == ValidationLimits().to_dict()
    result = verify_reproducible(record_path, bone, tmp_path / "again.stl")
    assert result.ok, result.problems


def test_tampered_plan_in_record_is_detected(plan, bone, tmp_path):
    record_path = export_guide(plan, bone, tmp_path / "TEST-001.stl")
    record = json.loads(record_path.read_text())
    record["plan"]["cutting"]["wall_thickness_mm"] = 3.0
    record_path.write_text(json.dumps(record))
    result = verify_reproducible(record_path, bone, tmp_path / "again.stl")
    assert not result.ok


def test_swapped_stl_on_disk_is_detected(plan, bone, tmp_path):
    record_path = export_guide(plan, bone, tmp_path / "TEST-001.stl")
    other = export_guide(with_cutting(plan, wall_thickness_mm=3.0), bone,
                         tmp_path / "other.stl")
    (tmp_path / "TEST-001.stl").write_bytes((tmp_path / "other.stl").read_bytes())
    result = verify_reproducible(record_path, bone, tmp_path / "again.stl")
    assert not result.ok
    assert any("on disk" in p for p in result.problems)
    assert other.exists()


def test_blocked_guide_cannot_export(plan, bone, tmp_path):
    bad = with_cutting(plan, wall_thickness_mm=0.8)
    with pytest.raises(ExportBlockedError, match="WALL_TOO_THIN") as info:
        export_guide(bad, bone, tmp_path / "x.stl", acknowledged=["WALL_TOO_THIN"])
    assert info.value.report.blocks
    assert list(tmp_path.iterdir()) == []


def test_warning_needs_acknowledgement(plan, bone, tmp_path):
    thin = with_cutting(plan, wall_thickness_mm=1.5)
    with pytest.raises(ExportBlockedError):
        export_guide(thin, bone, tmp_path / "x.stl")
    record = load_trace(export_guide(thin, bone, tmp_path / "x.stl",
                                     acknowledged=["WALL_THIN"]))
    assert record["acknowledged_warnings"] == ["WALL_THIN"]


def test_existing_export_is_never_overwritten(plan, bone, tmp_path):
    export_guide(plan, bone, tmp_path / "TEST-001.stl")
    with pytest.raises(FileExistsError):
        export_guide(plan, bone, tmp_path / "TEST-001.stl")
