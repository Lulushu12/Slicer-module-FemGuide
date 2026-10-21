"""Export gating rule of the shared validation report."""

from femguide_core.validation import Severity, ValidationReport


def test_empty_report_allows_export():
    assert ValidationReport().can_export()


def test_block_always_prevents_export():
    r = ValidationReport()
    r.add("X", Severity.BLOCK, "broken")
    assert not r.can_export(["X"])


def test_warnings_need_acknowledgement_by_code():
    r = ValidationReport()
    r.add("THIN", Severity.WARN, "thin")
    r.add("TIGHT", Severity.WARN, "tight")
    assert not r.can_export()
    assert not r.can_export(["THIN"])
    assert [f.code for f in r.unacknowledged(["THIN"])] == ["TIGHT"]
    assert r.can_export(["THIN", "TIGHT"])


def test_to_dict_round_trips_codes():
    r = ValidationReport()
    r.add("THIN", Severity.WARN, "thin")
    r.metrics["volume_mm3"] = 1.5
    d = r.to_dict()
    assert d["findings"] == [{"code": "THIN", "severity": "warn", "message": "thin"}]
    assert d["metrics"] == {"volume_mm3": 1.5}
