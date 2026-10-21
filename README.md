# FemGuide Suite

Patient-specific, 3D-printable surgical guide planning. See
`docs/ARCHITECTURE.md` for the frozen commercial architecture plan.

## Repository layout

- `core/` - pure-Python geometry engine (`femguide_core`), no Slicer/VTK/Qt
  imports, installable and testable on its own: `pip install -e "./core[dev]"`.
- `modules/femguide/` - the FemGuide module. `femguide/` is its pure-Python
  logic (plan model, regeneration, validation gate, traceable export),
  installable with `pip install -e ./modules/femguide`; `LineGenerator.py` is
  the original Slicer prototype, kept until the clinician wizard (Phase 2b)
  replaces it.
- `tests/` - pytest suite for the core library and module logic. Run with
  `pytest tests/`.
- `docs/` - architecture plan and pinned versions.

CI (GitHub Actions) runs the geometry tests on Python 3.12 (the pinned Slicer's
Python) and 3.13.
