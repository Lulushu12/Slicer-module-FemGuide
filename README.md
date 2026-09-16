# FemGuide Suite

Patient-specific, 3D-printable surgical guide planning. See
`docs/ARCHITECTURE.md` for the frozen commercial architecture plan.

## Repository layout

- `core/` - pure-Python geometry engine (`femguide_core`), no Slicer/VTK/Qt
  imports, installable and testable on its own: `pip install -e "./core[dev]"`.
- `modules/femguide/` - the FemGuide Slicer module (currently the original
  prototype; rebuilt as a clinician wizard in Phase 2).
- `tests/` - pytest suite for the geometry engine. Run with `pytest tests/`.
- `docs/` - architecture plan and pinned versions.

CI (GitHub Actions) runs the geometry tests on Python 3.9 (the pinned Slicer's
Python) and 3.12.
