# FemGuide Suite: Commercial Architecture Plan

Status: FROZEN (planning baseline, 2026-09-16)
Owner: Radu Josanu
Scope: technical architecture only. Regulatory and legal matters are handled separately and are out of scope for this document.

This document is the agreed baseline for commercialization of FemGuide and its sibling
products. Execution work happens in a separate thread/sessions; this file is the source
of truth for what was decided and why. Changes to frozen decisions should update this
document first.

---

## 1. Product vision

A single branded desktop application ("FemGuide Suite", working name) for on-site use by
surgeons, producing patient-specific, 3D-printable surgical guides and molds. Sold as a
subscription (in practice: annual site licenses via purchase orders), with per-module
entitlements.

Product modules (patent applications submitted for the underlying devices where noted):

1. **FemGuide** - femoral cutting guide (existing prototype).
2. **Glenoid Guide** - TSA guidance-wire guide (patented).
3. **Hip Spacer Mold** - mold for casting custom hip spacers (patented).
4. **TKA Planning** - full total knee arthroplasty planning with licensed manufacturer
   implants. Gated on manufacturer agreement for implant geometry access.

Team: solo developer plus Claude. Every decision below is shaped by that constraint:
one thing ships at a time, boring technology off the critical path, no ops burden.

---

## 2. Frozen decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Host platform | Custom Slicer-based application (SlicerCustomAppTemplate / "SlicerCAT" route). Fallback held in reserve: own installer bundling a pinned stock Slicer (see 3.2). |
| D2 | Module architecture | Shared core + entitlement-gated plugin modules. Plugin API kept minimal; generalized only after the second module port. |
| D3 | Geometry kernel | Three-layer engine (see 4.1). Mesh booleans via **manifold3d**, never vtkBooleanOperationPolyDataFilter on segmented anatomy. |
| D4 | Parametric model | "The plan is the product; the mesh is disposable." Fully non-destructive: all inputs stored as plan data, geometry always regenerated from inputs. See section 5. |
| D5 | Licensing | Signed offline license files with per-module entitlements, expiry + grace period (30-90 days), named-machine seats first. No always-online checks. |
| D6 | Updates | Signed versioned packages, release channels (internal/beta/stable), manual apply only (never silent auto-update), offline USB path via same signature verification. |
| D7 | Module language | All modules, licensing client, and updater in Python. C++ touched only for the app superbuild, set up once and frozen. |
| D8 | Backend | Minimal FastAPI license server on a VPS; update packages as signed static files on object storage. Stripe or invoice-based billing behind the license server. No Kubernetes, no microservices. |
| D9 | Module sequencing | Foundation -> App skeleton + FemGuide -> Licensing/updates -> Glenoid -> Hip Spacer Mold -> TKA. |
| D10 | Open-source licensing posture | Slicer core license (BSD-style) permits proprietary commercialization. Qt used under LGPLv3: dynamic linking only, Qt libs remain replaceable in the installer. Any future community extension gets a license check before inclusion. Attribution/third-party-notices screen required in installer. |

---

## 3. Platform

### 3.1 Custom Slicer application (chosen)

- Built from Kitware's SlicerCustomAppTemplate: own branding, own installer, pinned
  Slicer version, all unneeded Slicer modules stripped, full UI ownership.
- Ships as one installer per platform (Windows first; the on-site planning workstation
  is the target deployment).
- CI (GitHub Actions) produces the installer; aggressive build caching because the
  superbuild is multi-hour. The C++/CMake layer is configured once, then frozen (D7).

### 3.2 Fallback (held in reserve, not the plan)

If the superbuild costs more than ~2 weeks of fight: own installer bundling a pinned
stock Slicer with modules preloaded and unwanted modules hidden via application
settings. ~80% of the product experience for ~20% of the infrastructure; upgradeable
to the true custom app later without rewriting modules. Do not lead with this;
escalate to it consciously if needed.

### 3.3 Deployment model

Fat client on a designated on-site planning workstation. No on-site server component,
nothing for hospital IT to veto beyond one installer. Hospital networks are assumed
hostile: locked-down machines, blocked outbound connections. Nothing in the product
may hard-require network access at runtime.

---

## 4. Core platform

Shared, always-installed components.

### 4.1 Geometry engine (three layers)

Pure Python library, **no Slicer imports**, testable outside Slicer, versioned
independently. The existing `LineGenerator - Backup.py` geometry math is the seed for
layer 1 and must be extracted from its UI code.

- **Layer 1 - Parametric primitives.** Polyline profiles, extruded prisms, offset
  flanges, oriented cylinders (current LineGenerator capability). Also generates pin /
  K-wire cylinders and the parametric spacer body.
- **Layer 2 - Robust mesh operations.** Boolean union/subtract, surface offset /
  shelling, solid splitting along a parting plane. Implemented on manifold3d with a
  mandatory mesh-repair pass (decimation, hole closing, manifoldness enforcement)
  before any boolean, because segmentation output is noisy and non-manifold.
  This layer is the highest technical risk in the project and is de-risked first
  (Phase 1) with tests against real segmented meshes.
- **Layer 3 - Anatomy operations.** Closed-curve drawing snapped to a bone model
  surface; enclosed surface-patch extraction; patch offset into a conforming solid
  ("molded base"); landmark-derived local coordinate frames. The
  "draw footprint on bone -> conforming base -> subtract pin cylinders" pipeline is
  shared by FemGuide and Glenoid Guide.

### 4.2 Case management

Load DICOM or mesh, name the case, save and reopen planning sessions. A saved case is
the full plan data (section 5), reopenable and fully re-editable.

### 4.3 Export pipeline

STL/3MF export with an embedded traceability record: module + core versions, complete
parameter set / plan snapshot, case ID, timestamp. Every printed guide must be
traceable to the exact software state and inputs that produced it.

Export is gated by a **validation pass** (section 5.3).

### 4.4 Implant & instrument library

Versioned packages containing meshes plus metadata (sizes, reference frames,
manufacturer, catalog IDs), distributed through the same signed-package channel as
module updates. Defined in Phase 3. First consumers: K-wire/pin specs and saw blade
specs (Glenoid, FemGuide). Primary future consumer: TKA implant geometries.

### 4.5 Licensing client, update client, telemetry

See sections 7 and 8. Telemetry and crash reporting are opt-in.

---

## 5. Parametric model and adjustability

### 5.1 The rule (D4)

Everything the user does - landmarks, drawn outlines, pin positions, angles,
thicknesses, placement transforms - is stored as plan inputs. Final geometry is always
regenerated from scratch from those inputs. The user never edits a mesh directly;
nothing is baked in until export. Consequences:

- Any input editable at any time in any order (change the base outline after placing
  pins; pins survive).
- Undo = restore previous plan state.
- Cases reopen fully editable.
- Any export is exactly reproducible from its plan record.
- Regeneration must feel live: target under ~2 seconds worst case.

### 5.2 Adjustability, two synchronized forms everywhere

1. **Direct 3D manipulation**: drag handles, curve control points, planes, axes.
2. **Numeric panel**: spinboxes with safe ranges, mirroring every spatial adjustment
   live. Surgeons think in numbers; both forms always in sync.

Advanced/engineer cockpit (full raw parameters) exists behind a toggle; the clinician
wizard is the default face of every module.

### 5.3 Validation gate at export

Free adjustment can produce invalid guides. Export runs geometric checks:

- Hard blocks: non-manifold/broken geometry, base thinner than printable minimum,
  pin hole intersecting the cut slot, features outside the base.
- Warnings requiring explicit acknowledgement: thin regions, marginal clearances,
  base overlapping suspicious surface regions.

### 5.4 Clinician UX baseline (all modules)

Wizard flow per module: load scan -> landmarks/outline -> generate -> adjust -> review
-> validate -> export. Mouse-only operation, large hit targets, readable at arm's
length, undo everywhere.

---

## 6. Module specifications

### 6.1 FemGuide (full commercial version)

Union of three parts, minus holes:

1. **Cutting structure**: polyline-driven osteotomy slot geometry (existing prototype
   math). Slot width = **saw blade thickness + clearance**, an explicit named
   parameter (eventually per-blade from the instrument library), plus slot depth and
   per-segment start/end extensions.
2. **Patient-specific base**: surgeon draws a closed outline directly on the 3D femur
   surface (Slicer closed-curve markups with surface snapping); enclosed patch is
   extracted and offset outward into a solid; underside is a negative of the patient's
   femur so the guide seats uniquely.
3. **Pin holes**: cylinders subtracted through the base.

Adjustable: segment angles/lengths (drag control points or type); extensions; slot
width/depth; base outline (drag/add/delete control points or redraw); base thickness;
bone-fit clearance (printer-dependent snug vs slip fit); pin add/remove, position,
direction (auto surface-normal or free with angle readout), diameter, through/blind;
whole-guide placement transform (drag + numeric nudge).

### 6.2 Glenoid Guide (TSA)

- **Base**: drawn footprint on the glenoid, conforming molded solid (same layer-3
  pipeline as FemGuide base). Adjustable: outline, thickness, fit clearance.
- **K-wire hole**: larger central hole for the guidance K-wire. Entry point draggable
  on the glenoid face; direction adjustable by dragging the wire axis with live
  version/inclination numeric readout (numbers lead here); wire diameter from
  instrument library; sleeve/boss height around the hole.
- **2 pin holes** for intraoperative fixation: position, diameter, direction
  adjustable; option for non-parallel (convergent/divergent) pins so the guide cannot
  lift off along a single axis (exposed to the user, not hidden automatic).

### 6.3 Hip Spacer Mold

- Parametric spacer body (Radu's patented parameterization; the concrete parameter
  set is supplied when this module starts - OPEN ITEM).
- Mold = block/shell minus spacer cavity, **split into two halves** along a parting
  plane, with **registration keys** so the halves align. **No pour channel** (handled
  in the OR if needed).
- Adjustable: spacer shape parameters; mold outer dimensions or auto-fit with margin;
  parting plane position/orientation (draggable plane + numeric); cavity clearance
  (cement shrinkage/fit compensation); registration key count, size, auto placement
  with manual override.

### 6.4 TKA Planning (gated)

Landmarking on femur/tibia, axis construction, resection plane planning, implant
sizing and placement from the manufacturer implant library, varus/valgus / slope /
rotation adjustment, review UI. Comparable in effort to the other three modules
combined. **Gated on the manufacturer conversation producing at least a verbal yes on
implant geometry access before any development effort.** Built on the implant library
from Phase 3.

---

## 7. Licensing and subscription (D5)

- **License server** (FastAPI, VPS) issues cryptographically signed entitlement
  files: customer, licensed modules, seat count, expiry. Billing (Stripe or invoices)
  sits behind it; sales onboarding is human, so no self-serve portal investment.
- **App validates the signature locally.** No network required at launch.
- **Renewal**: silent refresh when online; works until expiry + grace window (30-90
  days) when offline, with visible warnings; renewal also possible by emailing a
  license file for permanently offline workstations.
- **Entitlements are per-module**: "FemGuide only" vs "full suite" is a license file
  difference, not a build difference. Module plugins declare an entitlement ID in
  their manifest; core activates only entitled modules at startup.
- **Seats**: named-machine first. Floating/concurrent licensing only if a customer
  demands it.

---

## 8. Updates and distribution (D6)

- Update service = signed, versioned packages for core, each module, and
  implant/instrument library packages, on static object storage. Channels:
  internal, beta, stable.
- Client checks for updates but **never auto-applies**; operator confirms every
  update. The app records exactly which versions produced every plan/export.
- Module manifests declare `minimum core version`; core enforces compatibility.
  New product modules ship through the same channel, gated by entitlement.
- **Offline path**: downloadable signed bundle applied from USB, verified by the same
  signature check as online updates. Free by construction if everything is
  signed-package-based from day one.

---

## 9. Plugin module contract

Each module is a Python package with a manifest:

- name, version, minimum core version, entitlement ID
- workflow UI entry point (wizard), advanced panel
- parametric templates / plan schema contributions

Core: discovers installed modules at startup, checks entitlements, activates.
The contract stays minimal until the Glenoid port (module #2 on the new core)
corrects it with real-world requirements (D2).

---

## 10. Open-source licensing posture (D10)

Verified against the current Slicer License.txt (BSD-style, Brigham and Women's
Hospital):

- Commercial, proprietary use and redistribution explicitly permitted; no copyleft.
- Requirements: include license text + attribution preface in third-party notices;
  no use of Brigham/contributor names or logos for endorsement; modified versions
  identified as modified (satisfied by own branding).
- Clinical-use clause is a disclaimer, not a prohibition; clinical-fitness burden is
  the vendor's (handled on the regulatory track, out of scope here).
- **Qt (LGPLv3)**: dynamic linking only; Qt libraries ship as separate replaceable
  DLLs; no packaging scheme that prevents user relinking. Commercial Qt license
  purchasable if ever needed; not expected for a standard dynamically-linked app.
- Dependencies verified permissive: VTK (BSD), ITK (Apache-2.0), CTK (Apache-2.0),
  DCMTK (BSD-style), numpy (BSD), manifold3d (Apache-2.0).
- Standing rule: any community Slicer extension gets a license check before
  inclusion (some are research-only or GPL). Currently none are used.
- Counsel to confirm this reading (cheap insurance; no landmines expected).

---

## 11. Phasing

1. **Foundation.** Repo restructure (`core/`, `modules/femguide/`, `tests/`);
   extract LineGenerator math into the pure geometry library with tests; build
   layer 2 (manifold3d + mesh repair) and prove it on real segmented meshes;
   build layer 3 curve-on-surface -> patch -> conforming solid pipeline;
   pin Slicer version; CI running geometry tests.
2. **App skeleton + full FemGuide.** SlicerCustomAppTemplate branded shell, stripped
   module set, installer out of CI. FemGuide rebuilt as clinician wizard per 6.1
   (note: now includes layers 2+3 work, larger than a naive port).
3. **Licensing + updater.** Signed license files, entitlements, grace period;
   signed update packages, manual apply, offline USB path; implant/instrument
   library format defined.
4. **Glenoid Guide.** Exercises the anatomy layer fully; plugin API corrections
   happen here (D2).
5. **Hip Spacer Mold.** Mostly layers 1+2 reuse plus parting-plane split and
   registration keys. Needs spacer parameterization from Radu.
6. **TKA Planning.** Gated on manufacturer agreement (6.4).
7. **Continuous.** Beta channel, first on-site pilot, feedback into wizard UX.

Sequencing rationale: licensing before module #2 (retrofitting entitlements onto a
shipped multi-module app is miserable); module #2 before generalizing the plugin API;
Glenoid before Spacer (fully exercises the anatomy layer, clearest patent-backed
differentiator, well-bounded scope); TKA last (largest, externally gated).

Honest solo-dev horizon to first external FemGuide/Glenoid beta: **12-18 months** at
side-project pace.

---

## 12. Open items

- [ ] Spacer parameterization: concrete parameter list from the patented design
      (needed at Phase 5 start).
- [ ] Manufacturer conversation for TKA implant geometry access (gates Phase 6;
      costs nothing to start early).
- [ ] Product/brand name for the suite.
- [ ] First real segmented datasets (femur, scapula) for layer 2/3 test fixtures.

## 13. Current state of this repository

Single-file Slicer scripted-module prototype (`LineGenerator - Backup.py`):
polyline-driven slot geometry, offset flanges, paired cylinders. It is the seed of
geometry layer 1 and nothing else yet. Phase 1 begins by restructuring this repo.
