# FemGuide module

- `femguide/` - the module's logic in pure Python (no Slicer imports):
  `plan.py` holds every user input as an immutable plan (D4), `cutting.py`
  builds the slot and the walls that capture the blade, `regenerate.py` turns
  a plan into guide geometry, `validate.py` is FemGuide's part of the export
  validation gate, `export.py` writes the STL plus its traceability record.
  The Slicer clinician wizard (Phase 2b) will be a view on top of this.
- `LineGenerator.py` - the original single-file Slicer prototype, unchanged.
  Its geometry math lives on in `core/femguide_core/geometry/` (layer 1).
