# FemGuide module

- `FemGuide.py` - the clinician wizard, a Slicer scripted module (Phase 2b):
  Case -> Base outline -> Cut -> Pins -> Review -> Export. It is a thin view:
  every action goes to `femguide/session.py`.
- `femguide/` - the module's logic in pure Python (no Slicer imports), tested
  in CI: `plan.py` holds every user input as an immutable plan (D4),
  `cutting.py` builds the slot and the walls (auto-connected to the base),
  `regenerate.py` turns a plan into guide geometry, `validate.py` is FemGuide's
  part of the export validation gate, `export.py` writes the STL plus its
  traceability record, `session.py` is everything the wizard does (edits,
  undo/redo, drop-onto-base, nudges, pin angles).
- `LineGenerator.py` - the original single-file prototype, unchanged.

## Trying the wizard in Slicer

1. Install 3D Slicer **5.12.4** (the pinned version, see
   `docs/PINNED_VERSIONS.md`).
2. Edit > Application Settings > Modules > Additional module paths: add this
   folder (`modules/femguide`). Restart Slicer.
3. Open the **FemGuide** module (category "FemGuide Suite"). The first time,
   click **Install required packages** (installs trimesh and manifold3d
   into Slicer's Python), then restart Slicer once more.
4. Load a femur surface: a model (STL) or a segmentation. STL files are read
   as LPS by default; if the bone appears mirrored, reload it with the
   coordinate system set to RAS in the Add Data dialog.
5. Work through the steps. Useful checks:
   - Case: Prepare bone shows the repair report.
   - Base outline: click points around the seat area; they snap to the bone
     and the outline follows the surface. Drag points to adjust.
   - Cut: edit segments and blade/wall numbers, "Drop cut onto base", then
     drag with the 3D handles or use the nudge buttons. The guide rebuilds
     after every change and stays connected to the base.
   - Pins: click to place, "Align pins with the cut direction" if Review
     reports pins crossing the blade path.
   - Review: blocks in red, warnings in orange with a tick box each.
   - Export: enabled only when nothing blocks and every warning is ticked.
   - Undo/Redo (Ctrl+Z / Ctrl+Y) at any point.
   - Save the scene (Ctrl+S) and reopen it: the plan reopens fully editable.

## Self-test

Runs the whole wizard on a synthetic bone, headless, and fails on any error
(including errors inside Qt callbacks, which Qt would otherwise only print):

    Slicer --no-splash --no-main-window \
        --additional-module-paths modules/femguide \
        --python-script scripts/slicer_selftest.py
