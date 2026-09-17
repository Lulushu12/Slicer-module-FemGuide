#!/usr/bin/env python3
"""Phase 1 smoke test: exercise the geometry engine and dump STLs to inspect.

Two modes:

  demo      Generate STLs from the prototype's default parameters plus a
            molded-base example on a synthetic femur-like shape. No inputs
            needed. Open the resulting STLs in Slicer (drag and drop) and
            compare the cutting structure against the old LineGenerator
            module with the same numbers.

  real      Take a real segmented bone STL through the commercial pipeline:
            repair() -> footprint curve -> molded_base() -> subtract pins.
            Prints the repair report and per-stage timings against the
            architecture's ~2 s regeneration target (D4). This is the test
            that closes the "prove layer 2 on real segmented meshes" item.

Examples (run inside the venv, from the repo root):

  python scripts/smoke_test.py demo
  python scripts/smoke_test.py real ~/segmentations/femur.stl
  python scripts/smoke_test.py real femur.stl --center 12 40 -655 --radius 14

By default the footprint for `real` is drawn around the surface point nearest
the mesh centroid, which is arbitrary; pass --center (a point near where you
would place the guide, in the mesh's own coordinates, e.g. read off Slicer's
data probe) for a meaningful location.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from femguide_core.geometry import (
    conforming_solid_from_patch,
    difference,
    extract_patch,
    molded_base,
    offset_prism,
    oriented_cylinder,
    paired_cylinders,
    polyline_from_angles,
    polyline_plane_normal,
    polyline_prism_solids,
    project_curve_to_surface,
    repair,
    safe_normalize,
    snap_points_to_surface,
    union,
)


def _describe(name: str, mesh: trimesh.Trimesh) -> None:
    print(
        "  {:<28} {:>7} faces  watertight={}  volume={:.1f} mm^3".format(
            name, len(mesh.faces), mesh.is_watertight, mesh.volume
        )
    )


def _export(mesh: trimesh.Trimesh, out_dir: Path, name: str) -> None:
    path = out_dir / name
    mesh.export(str(path))
    _describe(name, mesh)


def _tangent_basis(normal: np.ndarray) -> tuple:
    """Two unit vectors spanning the plane perpendicular to ``normal``."""
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = safe_normalize(np.cross(normal, helper))
    v = np.cross(normal, u)
    return u, v


def _circle(center: np.ndarray, u: np.ndarray, v: np.ndarray, radius: float,
            n: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return center + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


# ---------------------------------------------------------------------------
# demo mode
# ---------------------------------------------------------------------------

def _femur_like() -> trimesh.Trimesh:
    """Same synthetic femur stand-in as the test fixtures (clean version)."""
    shaft = trimesh.creation.capsule(radius=14.0, height=120.0, count=[48, 48])
    head = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    head.apply_translation([12.0, 0.0, 70.0])
    combined = trimesh.util.concatenate([shaft, head])
    combined.merge_vertices()
    return combined


def run_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Writing demo STLs to {}/".format(out_dir))

    # Prototype default parameters (LineGenerator UI defaults).
    angles = [8.0, 40.0, 45.0, 45.0, 42.0]
    lengths = [50.0, 20.5, 25.0, 11.0, 30.0]
    pts = polyline_from_angles(angles, lengths)
    normal = polyline_plane_normal(pts)

    print("\nLayer 1: prototype defaults "
          "(angles {}, lengths {})".format(angles, lengths))
    cutting = union(polyline_prism_solids(pts, normal, thickness=1.0,
                                          extrusion=5.0))
    _export(cutting, out_dir, "01_cutting_structure.stl")

    flange = offset_prism(pts, normal, seg_index=1, length=20.0,
                          width_inplane=1.0, extrusion=2.0, offset=0.0)
    _export(flange, out_dir, "02_secondary_flange.stl")

    cylinders = paired_cylinders(pts, normal, seg_index=0, frac_along=0.5,
                                 radius=2.0, height=10.0, offset_from_plane=5.0)
    _export(union(cylinders), out_dir, "03_paired_cylinders.stl")

    # The prototype default offset (5) keeps the cylinders clear of the
    # +-2.5 mm prism, so for a visible subtraction demo bring them closer.
    cutting_cyls = paired_cylinders(pts, normal, seg_index=0, frac_along=0.5,
                                    radius=2.0, height=10.0,
                                    offset_from_plane=2.0)
    guide = difference(union([cutting, flange]), cutting_cyls)
    _export(guide, out_dir, "04_guide_union_minus_cylinders.stl")

    # Layer 3: molded base on the synthetic femur head, minus two pins.
    print("\nLayer 3: molded base on the synthetic femur head")
    bone = _femur_like()
    _export(bone, out_dir, "05_synthetic_femur.stl")

    head_center = np.array([12.0, 0.0, 70.0])
    outward = np.array([1.0, 0.0, 0.0])
    u, v = _tangent_basis(outward)
    curve = _circle(head_center + outward * 26.0, u, v, radius=10.0)
    base = molded_base(bone, curve, thickness=3.0, clearance=0.4)
    _export(base, out_dir, "06_molded_base.stl")

    surface_point = snap_points_to_surface(bone, (head_center + outward * 26.0)[None, :])[0]
    pins = [
        oriented_cylinder(surface_point + u * 4.0, outward, radius=1.6, height=24.0),
        oriented_cylinder(surface_point - u * 4.0, outward, radius=1.6, height=24.0),
    ]
    base_with_pins = difference(base, pins)
    _export(base_with_pins, out_dir, "07_molded_base_with_pin_holes.stl")

    print("\nDone. Open the STLs in Slicer and compare 01/04 against the old "
          "LineGenerator module\nwith the same parameters (they should "
          "coincide).")
    return 0


# ---------------------------------------------------------------------------
# real mode
# ---------------------------------------------------------------------------

def run_real(args: argparse.Namespace) -> int:
    stl_path = Path(args.stl)
    if not stl_path.exists():
        print("error: {} does not exist".format(stl_path), file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stl_path.stem

    print("Loading {} ...".format(stl_path))
    raw = trimesh.load(str(stl_path), force="mesh")
    print("  {} vertices, {} faces, watertight={}".format(
        len(raw.vertices), len(raw.faces), raw.is_watertight))

    # Stage 1: repair (one-time cost at case import, not part of the 2 s
    # regeneration budget).
    t0 = time.perf_counter()
    bone, report = repair(raw)
    t_repair = time.perf_counter() - t0
    print("\nrepair() took {:.2f} s".format(t_repair))
    print(report)
    _export(bone, out_dir, stem + "_repaired.stl")
    if not report.is_manifold_solid:
        print("\nFAIL: repair() could not produce a manifold solid from this "
              "mesh.\nThis is exactly the finding Phase 1 exists to surface - "
              "please share this mesh so the\nrepair pipeline can be extended.",
              file=sys.stderr)
        return 1

    # Footprint location and local frame.
    if args.center is not None:
        seed = np.array(args.center, dtype=float)
    else:
        seed = bone.centroid
        print("\nNo --center given; drawing the footprint around the surface "
              "point nearest the\nmesh centroid. That spot is arbitrary - pass "
              "--center X Y Z for a real location.")
    center = snap_points_to_surface(bone, seed[None, :])[0]
    # Nearest FACE normal, not vertex_normals: without scipy installed,
    # trimesh's vertex-normal fallback is a very slow pure-Python loop.
    nearest_face = int(
        np.argmin(np.linalg.norm(bone.triangles_center - center, axis=1))
    )
    normal = safe_normalize(np.asarray(bone.face_normals)[nearest_face])
    radius = args.radius
    if radius is None:
        radius = 0.12 * float(np.linalg.norm(bone.extents))
    u, v = _tangent_basis(normal)
    curve = _circle(center + normal * (radius * 0.25), u, v, radius)
    print("\nFootprint: circle of radius {:.1f} mm at ({:.1f}, {:.1f}, {:.1f})"
          .format(radius, *center))

    # Stage 2: the regeneration pipeline (this is what the D4 ~2 s budget
    # covers: everything from plan inputs to final guide geometry).
    timings = {}
    try:
        t0 = time.perf_counter()
        on_surface = project_curve_to_surface(bone, curve)
        timings["project_curve_to_surface"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        patch = extract_patch(bone, on_surface)
        timings["extract_patch"] = time.perf_counter() - t0
        print("  patch: {} faces, area {:.0f} mm^2".format(
            len(patch.faces), patch.area))

        t0 = time.perf_counter()
        base = conforming_solid_from_patch(patch, thickness=args.thickness,
                                           clearance=args.clearance)
        timings["conforming_solid_from_patch"] = time.perf_counter() - t0
    except ValueError as exc:
        print("\nFAIL during base generation: {}\nTry a different --center "
              "(flatter region) or a larger/smaller --radius.".format(exc),
              file=sys.stderr)
        return 1
    _export(base, out_dir, stem + "_base.stl")

    pin_offset = radius * 0.5
    pins = [
        oriented_cylinder(center + u * pin_offset, normal,
                          radius=args.pin_radius, height=args.thickness * 8.0),
        oriented_cylinder(center - u * pin_offset, normal,
                          radius=args.pin_radius, height=args.thickness * 8.0),
    ]
    t0 = time.perf_counter()
    guide = difference(base, pins)
    timings["boolean difference (pins)"] = time.perf_counter() - t0
    _export(guide, out_dir, stem + "_guide.stl")

    print("\nRegeneration timings (D4 target: ~2 s worst case, whole pipeline):")
    total = 0.0
    for name, seconds in timings.items():
        print("  {:<32} {:6.2f} s".format(name, seconds))
        total += seconds
    verdict = "OK" if total <= 2.0 else "OVER BUDGET"
    print("  {:<32} {:6.2f} s   [{}]".format("total", total, verdict))

    print("\nDone. Inspect {}_base.stl in Slicer overlaid on the original: "
          "the underside must\nhug the bone with a uniform {:.1f} mm gap "
          "(the clearance).".format(stem, args.clearance))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_demo = sub.add_parser("demo", help="dump STLs from prototype defaults")
    p_demo.add_argument("-o", "--out", default="smoke_out",
                        help="output directory (default: smoke_out)")
    p_demo.set_defaults(func=run_demo)

    p_real = sub.add_parser("real",
                            help="run a segmented bone STL through the pipeline")
    p_real.add_argument("stl", help="path to the segmented bone STL")
    p_real.add_argument("-o", "--out", default="smoke_out",
                        help="output directory (default: smoke_out)")
    p_real.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="footprint center in mesh coordinates "
                             "(default: near the centroid, arbitrary)")
    p_real.add_argument("--radius", type=float,
                        help="footprint circle radius in mm "
                             "(default: 12%% of the bounding-box diagonal)")
    p_real.add_argument("--thickness", type=float, default=3.0,
                        help="base thickness in mm (default: 3)")
    p_real.add_argument("--clearance", type=float, default=0.4,
                        help="bone-fit clearance in mm (default: 0.4)")
    p_real.add_argument("--pin-radius", type=float, default=1.6,
                        help="pin hole radius in mm (default: 1.6)")
    p_real.set_defaults(func=run_real)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
