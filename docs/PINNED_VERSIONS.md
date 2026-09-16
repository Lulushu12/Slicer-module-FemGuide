# Pinned versions

Per D1/D7 (docs/ARCHITECTURE.md): the Slicer version is pinned here first;
the SlicerCustomAppTemplate superbuild in Phase 2 must pin the exact same
release tag and record its git revision below when the superbuild is set up.

| Component | Pinned version | Notes |
|-----------|----------------|-------|
| 3D Slicer | 5.8.1 | Latest stable release at time of pinning. Ships Python 3.9, which is why CI tests on 3.9. Exact source revision to be recorded at Phase 2 superbuild setup. |
| Python (in-app) | 3.9 | Dictated by Slicer 5.8.1. Core library must stay 3.9-compatible. |
| numpy | >=1.24 | Range, resolved and frozen per release at Phase 2 packaging. |
| trimesh | >=4.0 | Mesh interchange type + surface queries. MIT license. |
| manifold3d | >=2.5 | All booleans (D3). Apache-2.0 license. |

Rules:

- Nothing in `core/` may import `slicer`, `vtk`, `qt`, or `ctk`.
- Bumping the Slicer pin is a deliberate decision recorded in this file,
  never a side effect of a rebuild.
