"""FemGuide plan: every user input, and nothing else (D4).

"The plan is the product; the mesh is disposable." A plan holds every input
the surgeon sets in the FemGuide wizard (docs/ARCHITECTURE.md 6.1); guide
geometry is always regenerated from it (:mod:`femguide.regenerate`). Plans
are immutable, so undo/redo is plan history and edits are
``dataclasses.replace``. Plans serialize to plain JSON for case files and
for the export traceability record.

All lengths are millimetres; all coordinates are in the bone mesh's frame.

Constructor checks only reject inputs that are structurally meaningless
(mismatched list lengths, non-positive sizes, non-rigid transforms).
Merely risky values (thin walls, tight clearances) are allowed here and
reported by the validation gate (:mod:`femguide.validate`) instead, because
free adjustment must never be blocked mid-edit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

SCHEMA_VERSION = 1

Vec3 = Tuple[float, float, float]
IDENTITY: Tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _floats(values, name: str = "value") -> Tuple[float, ...]:
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("{} must contain numbers (got {!r})".format(name, v))
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("{} must contain finite numbers (got {})".format(name, f))
        out.append(f)
    return tuple(out)


def _vec3(value, name: str) -> Vec3:
    out = _floats(value, name)
    if len(out) != 3:
        raise ValueError("{} must be three numbers".format(name))
    return out  # type: ignore[return-value]


def _number(value, name: str) -> float:
    return _floats((value,), name)[0]


def _positive(value, name: str) -> float:
    value = _number(value, name)
    if value <= 0.0:
        raise ValueError("{} must be > 0 (got {})".format(name, value))
    return value


@dataclass(frozen=True)
class CuttingParams:
    """The osteotomy slot and the walls that capture the saw blade (6.1 item 1).

    The polyline (``angles_deg`` / ``lengths_mm``, prototype math) is the
    planned cut line, drawn in its local z=0 plane; each segment defines one
    cut plane (the segment swept along local z, the blade's direction of
    travel). Local -z points toward the bone: the walls auto-connect in that
    direction. The **slot** is the empty channel the blade passes through:
    ``blade_thickness_mm + blade_clearance_mm`` wide, centred on the cut
    line, open along the blade's entire path. Walls of
    ``wall_thickness_mm`` surround the slot on both sides and close its two
    outer ends; ``slot_depth_mm`` is the height of this placed wall block,
    which is then automatically extended down to the bone and trimmed to the
    bone-fit clearance (auto-connect). ``placement`` (row-major 4x4, rigid)
    maps the local frame into bone coordinates; it is the whole-guide
    placement transform, moving the cutting structure relative to the base
    fit.
    """

    angles_deg: Tuple[float, ...]
    lengths_mm: Tuple[float, ...]
    start_ext_mm: Tuple[float, ...] = ()
    end_ext_mm: Tuple[float, ...] = ()
    blade_thickness_mm: float = 1.27
    blade_clearance_mm: float = 0.15
    slot_depth_mm: float = 10.0
    wall_thickness_mm: float = 2.5
    placement: Tuple[float, ...] = IDENTITY

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "angles_deg", _floats(self.angles_deg, "angles_deg"))
        set_(self, "lengths_mm", _floats(self.lengths_mm, "lengths_mm"))
        n = len(self.angles_deg)
        if n == 0 or n != len(self.lengths_mm):
            raise ValueError(
                "angles_deg and lengths_mm must be non-empty and the same length")
        for i, length in enumerate(self.lengths_mm):
            _positive(length, "lengths_mm[{}]".format(i))
        for name in ("start_ext_mm", "end_ext_mm"):
            ext = _floats(getattr(self, name), name) or (0.0,) * n
            if len(ext) != n:
                raise ValueError("{} needs one entry per segment ({})".format(name, n))
            if any(v < 0.0 for v in ext):
                raise ValueError("{} entries must be >= 0".format(name))
            set_(self, name, ext)
        for name in ("blade_thickness_mm", "slot_depth_mm", "wall_thickness_mm"):
            set_(self, name, _positive(getattr(self, name), name))
        clearance = _number(self.blade_clearance_mm, "blade_clearance_mm")
        if clearance < 0.0:
            raise ValueError("blade_clearance_mm must be >= 0")
        set_(self, "blade_clearance_mm", clearance)
        placement = _floats(self.placement, "placement")
        if len(placement) != 16:
            raise ValueError("placement must be 16 numbers (row-major 4x4)")
        matrix = np.array(placement).reshape(4, 4)
        rotation = matrix[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
                or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])):
            raise ValueError(
                "placement must be a rigid transform (rotation + translation): "
                "scaling would change the slot width the blade relies on")
        set_(self, "placement", placement)

    @property
    def slot_width_mm(self) -> float:
        return self.blade_thickness_mm + self.blade_clearance_mm

    @property
    def placement_matrix(self) -> np.ndarray:
        return np.array(self.placement, dtype=float).reshape(4, 4)


@dataclass(frozen=True)
class BaseParams:
    """Patient-specific base (6.1 item 2): an outline drawn on the bone, offset
    by ``clearance_mm`` (bone-fit gap) into a ``thickness_mm`` solid."""

    outline_points: Tuple[Vec3, ...]
    thickness_mm: float = 3.0
    clearance_mm: float = 0.4

    def __post_init__(self) -> None:
        pts = tuple(_vec3(p, "outline_points[{}]".format(i))
                    for i, p in enumerate(self.outline_points))
        if len(pts) < 3:
            raise ValueError("outline_points needs at least 3 points")
        object.__setattr__(self, "outline_points", pts)
        object.__setattr__(self, "thickness_mm",
                           _positive(self.thickness_mm, "thickness_mm"))
        object.__setattr__(self, "clearance_mm",
                           _number(self.clearance_mm, "clearance_mm"))


@dataclass(frozen=True)
class Pin:
    """One fixation pin hole (6.1 item 3): holds the guide still on the bone
    during the cuts, so it always runs through the guide into the bone.

    ``position`` is where the pin enters the bone (a point on the bone
    surface). ``direction`` points into the bone; ``None`` means "auto":
    the inward surface normal at ``position``, recomputed on every
    regeneration so it follows edits. An explicit direction is stored as a
    unit vector; one that already is unit length is kept bit-for-bit so the
    plan round-trips through JSON exactly.
    """

    position: Vec3
    direction: Optional[Vec3] = None
    diameter_mm: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        if self.direction is not None:
            direction = _vec3(self.direction, "direction")
            norm = math.sqrt(sum(v * v for v in direction))
            if norm < 1e-9:
                raise ValueError("direction must be non-zero (or None for auto)")
            if abs(norm - 1.0) > 1e-12:
                direction = tuple(v / norm for v in direction)
            object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "diameter_mm",
                           _positive(self.diameter_mm, "diameter_mm"))


@dataclass(frozen=True)
class FemGuidePlan:
    """The complete FemGuide plan for one case.

    ``bone_sha256`` ties the plan to the exact (repaired) bone mesh it was
    made on; regeneration refuses a different mesh.
    """

    case_id: str
    bone_sha256: str
    cutting: CuttingParams
    base: Optional[BaseParams] = None
    pins: Tuple[Pin, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-blank string")
        if (not isinstance(self.bone_sha256, str) or len(self.bone_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.bone_sha256)):
            raise ValueError("bone_sha256 must be a 64-character lowercase hex digest")
        if not isinstance(self.cutting, CuttingParams):
            raise ValueError("cutting must be CuttingParams")
        if self.base is not None and not isinstance(self.base, BaseParams):
            raise ValueError("base must be BaseParams or None")
        if (type(self.schema_version) is not int
                or self.schema_version != SCHEMA_VERSION):
            raise ValueError("unsupported plan schema_version {} (expected {})"
                             .format(self.schema_version, SCHEMA_VERSION))
        pins = tuple(self.pins)
        if not all(isinstance(p, Pin) for p in pins):
            raise ValueError("pins must all be Pin")
        object.__setattr__(self, "pins", pins)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FemGuidePlan":
        data = dict(data)
        cutting = CuttingParams(**data.pop("cutting"))
        base_data = data.pop("base", None)
        base = BaseParams(**base_data) if base_data is not None else None
        pins = tuple(Pin(**p) for p in data.pop("pins", ()))
        return cls(cutting=cutting, base=base, pins=pins, **data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "FemGuidePlan":
        return cls.from_dict(json.loads(text))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FemGuidePlan":
        return cls.from_json(Path(path).read_text())
