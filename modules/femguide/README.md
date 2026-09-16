# FemGuide module

`LineGenerator.py` is the original single-file Slicer scripted-module prototype,
moved here unchanged during the Phase 1 repo restructure. It is the seed of the
commercial FemGuide module (docs/ARCHITECTURE.md section 6.1) and still runs as a
stock Slicer scripted module.

Its geometry math has been extracted into the pure library at
`core/femguide_core/geometry/` (layer 1). The full clinician-wizard rebuild of
this module on top of that library is Phase 2 work; until then this file stays
as the working prototype and is intentionally not refactored.
