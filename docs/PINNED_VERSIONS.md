# Pinned versions

Per D1/D7 (docs/ARCHITECTURE.md): the Slicer version is pinned here first;
the SlicerCustomAppTemplate superbuild in Phase 2 must pin the exact same
release tag and record its git revision below when the superbuild is set up.

| Component | Pinned version | Notes |
|-----------|----------------|-------|
| 3D Slicer | 5.12.4 | Newest release tag at re-pin time (2026-10-21). Ships Python 3.12.10 and defaults to Qt 5.15 (Qt 6.8 optional). Exact source revision to be recorded at superbuild setup. |
| Python (in-app) | 3.12 | Dictated by Slicer 5.12.4 (`SuperBuild/External_python.cmake`: 3.12.10). Core library requires >=3.12. |
| Qt | 5.15 | Slicer 5.12 default. Qt 5 is past end of standard support; a move to Qt 6.8 (LTS) is a deliberate later decision once Slicer's Qt 6 build is proven for us. |
| numpy | >=1.24 | Range, resolved and frozen per release at packaging. |
| trimesh | >=4.0 | Mesh interchange type + surface queries. MIT license. |
| manifold3d | >=2.5 | All booleans (D3). Apache-2.0 license. |

History:

- 2026-09-16: pinned 5.8.1 (Python 3.9). That was a stale pick, not the
  newest release at the time; CI then had to support Python 3.9, which
  dragged in trimesh 4.x and caused version-drift test failures.
- 2026-10-21: re-pinned to 5.12.4 (Python 3.12) before any superbuild work.

Rules:

- Nothing in `core/` may import `slicer`, `vtk`, `qt`, or `ctk`.
- Bumping the Slicer pin is a deliberate decision recorded in this file,
  never a side effect of a rebuild. Verify the bundled Python version from
  the tag's `SuperBuild/External_python.cmake` when bumping.
