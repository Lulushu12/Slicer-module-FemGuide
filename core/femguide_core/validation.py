"""Export validation gate (docs/ARCHITECTURE.md section 5.3).

Free adjustment can produce invalid guides, so every export is gated by a
validation pass. Each module contributes its own geometric checks; this file
holds the shared report type and the gating rule:

- BLOCK findings stop the export outright.
- WARN findings stop it until the user explicitly acknowledges each one
  (by code), and the acknowledgement is recorded in the export's
  traceability record.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


class Severity(str, enum.Enum):
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class Finding:
    """One validation result. ``code`` is a stable machine-readable ID (used
    for acknowledgement), ``message`` is for the clinician."""

    code: str
    severity: Severity
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add(self, code: str, severity: Severity, message: str) -> None:
        self.findings.append(Finding(code, severity, message))

    @property
    def blocks(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCK]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    def blocks_with(self, code: str) -> bool:
        return any(f.code == code for f in self.blocks)

    def unacknowledged(self, acknowledged: Iterable[str] = ()) -> List[Finding]:
        """Warnings not covered by the acknowledged codes."""
        acked = set(acknowledged)
        return [f for f in self.warnings if f.code not in acked]

    def can_export(self, acknowledged: Iterable[str] = ()) -> bool:
        return not self.blocks and not self.unacknowledged(acknowledged)

    def to_dict(self) -> Dict[str, object]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "metrics": dict(self.metrics),
        }

    def __str__(self) -> str:
        if not self.findings:
            return "ValidationReport: no findings"
        lines = ["ValidationReport: {} block(s), {} warning(s)".format(
            len(self.blocks), len(self.warnings))]
        for f in self.findings:
            lines.append("  [{}] {}: {}".format(
                f.severity.value.upper(), f.code, f.message))
        return "\n".join(lines)
