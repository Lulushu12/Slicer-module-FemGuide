"""Three-layer geometry engine (docs/ARCHITECTURE.md section 4.1).

Layer 1 (primitives): parametric polylines, prisms, offset flanges, oriented
cylinders. Extracted from the LineGenerator prototype, reimplemented on numpy +
trimesh with no VTK.

Layer 2 (robust mesh operations): mesh repair, manifold3d booleans, surface
offset, plane splitting. Segmentation output is assumed noisy and non-manifold;
every boolean goes through the repair pass first.

Layer 3 (anatomy operations): closed curve snapped to a bone surface, enclosed
patch extraction, patch offset into a conforming solid ("molded base"),
landmark-derived coordinate frames.

The interchange mesh type throughout is trimesh.Trimesh.
"""

from femguide_core.geometry.primitives import (
    safe_normalize,
    polyline_from_angles,
    polyline_plane_normal,
    segment_prism,
    polyline_prism_solids,
    offset_prism,
    oriented_cylinder,
    paired_cylinders,
)
from femguide_core.geometry.repair import (
    RepairReport,
    is_manifold_solid,
    repair,
)
from femguide_core.geometry.booleans import (
    NonManifoldError,
    to_manifold,
    from_manifold,
    union,
    difference,
    intersection,
)
from femguide_core.geometry.offset import offset_solid
from femguide_core.geometry.split import split_by_plane
from femguide_core.geometry.surface import (
    snap_points_to_surface,
    project_curve_to_surface,
    extract_patch,
    conforming_solid_from_patch,
    molded_base,
)
from femguide_core.geometry.frames import (
    frame_from_three_points,
    frame_from_origin_and_axes,
)

__all__ = [
    # layer 1
    "safe_normalize",
    "polyline_from_angles",
    "polyline_plane_normal",
    "segment_prism",
    "polyline_prism_solids",
    "offset_prism",
    "oriented_cylinder",
    "paired_cylinders",
    # layer 2
    "RepairReport",
    "is_manifold_solid",
    "repair",
    "NonManifoldError",
    "to_manifold",
    "from_manifold",
    "union",
    "difference",
    "intersection",
    "offset_solid",
    "split_by_plane",
    # layer 3
    "snap_points_to_surface",
    "project_curve_to_surface",
    "extract_patch",
    "conforming_solid_from_patch",
    "molded_base",
    "frame_from_three_points",
    "frame_from_origin_and_axes",
]
