"""FemGuide: clinician wizard for the femoral cutting guide (Phase 2b).

A thin Slicer view on the pure-Python module logic in ``femguide/``: every
click, drag and number goes to a :class:`femguide.session.PlanSession`,
which owns the plan, undo history, regeneration, validation and export.
This file only translates between MRML (markups, transforms, models) and
that session, and lays out the wizard (docs/ARCHITECTURE.md 5.4):

    Case -> Base outline -> Cut -> Pins -> Review -> Export

The plan is stored as JSON in the module's parameter node, so saving the
Slicer scene saves the case, and reopening it reopens the plan fully
editable (4.2).
"""

import json
import logging
import os
import sys

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "..", "core"))):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Packages the module logic needs that Slicer does not bundle.
REQUIRED_PACKAGES = {"trimesh": "trimesh>=4.0", "manifold3d": "manifold3d>=2.5"}

STEPS = ("Case", "Base outline", "Cut", "Pins", "Review", "Export")
BIG = 40  # minimum height of buttons and fields, for mouse-only use at arm's length
FONT_PT = 12  # wizard text size, readable at arm's length


def missingPackages():
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def installPackages():
    slicer.util.pip_install(" ".join(REQUIRED_PACKAGES.values()))


#
# Module
#

class FemGuide(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "FemGuide"
        parent.categories = ["FemGuide Suite"]
        parent.contributors = ["Radu Josanu"]
        parent.helpText = (
            "Plan a patient-specific femoral cutting guide: prepare the bone, "
            "draw the base outline, place the cut and the fixation pins, "
            "review the validation, and export a traceable STL.")
        parent.acknowledgementText = ""


#
# Mesh conversion
#

def trimeshFromPolyData(polyData):
    import trimesh
    from vtk.util.numpy_support import vtk_to_numpy

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polyData)
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    out = triangles.GetOutput()
    vertices = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)


def polyDataFromTrimesh(mesh):
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(cells)
    return polyData


def worldPolyData(node, segmentId=None):
    """A node's surface in world (RAS) coordinates, hardening any transform."""
    if node.IsA("vtkMRMLSegmentationNode"):
        node.CreateClosedSurfaceRepresentation()
        if segmentId is None:
            segmentId = node.GetSegmentation().GetNthSegmentID(0)
        polyData = vtk.vtkPolyData()
        node.GetClosedSurfaceRepresentation(segmentId, polyData)
    else:
        polyData = node.GetPolyData()
    if polyData is None or polyData.GetNumberOfPoints() == 0:
        raise ValueError("the selected bone has no surface")
    if node.GetParentTransformNode() is None:
        return polyData
    toWorld = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(node.GetParentTransformNode(), None, toWorld)
    transformed = vtk.vtkTransformPolyDataFilter()
    transformed.SetTransform(toWorld)
    transformed.SetInputData(polyData)
    transformed.Update()
    return transformed.GetOutput()


def matrixFromNumpy(array):
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, float(array[i][j]))
    return matrix


def numpyFromMatrix(matrix):
    return np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])


def placementMatrix(transformNode):
    matrix = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToParent(matrix)
    return numpyFromMatrix(matrix)


#
# Logic: MRML <-> session
#

class FemGuideLogic(ScriptedLoadableModuleLogic):
    """Owns the planning session and the MRML nodes that show it."""

    BONE = "FemGuideBone"
    OUTLINE = "FemGuideOutline"
    PINS = "FemGuidePins"
    PLACEMENT = "FemGuidePlacement"
    GUIDE = "FemGuideGuide"
    CUTS = "FemGuideCuts"

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.session = None
        self.geometry = None
        self.report = None
        self.repairReport = None

    # -- nodes ---------------------------------------------------------------

    def node(self, role):
        return self.getParameterNode().GetNodeReference(role)

    def _ensureNode(self, role, className, name):
        node = self.node(role)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(className, name)
            node.CreateDefaultDisplayNodes()
            self.getParameterNode().SetNodeReferenceID(role, node.GetID())
        return node

    def _setModel(self, role, name, mesh, color, opacity=1.0):
        node = self._ensureNode(role, "vtkMRMLModelNode", name)
        node.SetAndObservePolyData(polyDataFromTrimesh(mesh))
        display = node.GetDisplayNode()
        display.SetColor(*color)
        display.SetOpacity(opacity)
        display.SetVisibility(True)
        return node

    # -- case ----------------------------------------------------------------

    def prepareBone(self, sourceNode, caseId, segmentId=None):
        """Repair the bone surface (once, at case import) and start a session."""
        from femguide.session import PlanSession
        from femguide_core.geometry import repair

        raw = trimeshFromPolyData(worldPolyData(sourceNode, segmentId))
        bone, report = repair(raw)
        self.repairReport = report
        if not report.is_manifold_solid:
            raise ValueError("the bone surface could not be repaired into a closed "
                             "solid:\n{}".format(report))
        boneNode = self._setModel(self.BONE, "FemGuide bone", bone, (0.95, 0.91, 0.82))
        if sourceNode.GetDisplayNode():
            sourceNode.GetDisplayNode().SetVisibility(False)
        self.session = PlanSession.start(bone, caseId)
        self._createEditingNodes(boneNode)
        self.saveToParameterNode()
        return report

    def _createEditingNodes(self, boneNode):
        outline = self._ensureNode(self.OUTLINE, "vtkMRMLMarkupsClosedCurveNode", "Base outline")
        outline.SetAndObserveSurfaceConstraintNode(boneNode)
        outline.SetCurveTypeToShortestDistanceOnSurface(boneNode)
        outline.GetDisplayNode().SetSelectedColor(0.1, 0.6, 1.0)
        pins = self._ensureNode(self.PINS, "vtkMRMLMarkupsFiducialNode", "Pins")
        pins.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)
        pins.SetMarkupLabelFormat("Pin %d")
        placement = self._ensureNode(self.PLACEMENT, "vtkMRMLTransformNode", "Cut placement")
        display = placement.GetDisplayNode()
        display.SetEditorScalingEnabled(False)  # scaling would change the slot width

    def restoreFromParameterNode(self):
        """Rebuild the session from a saved scene. Returns False if there is
        nothing to restore; raises if the saved bone no longer matches."""
        from femguide.plan import FemGuidePlan
        from femguide.session import PlanSession

        planJson = self.getParameterNode().GetParameter("PlanJSON")
        boneNode = self.node(self.BONE)
        if not planJson or boneNode is None:
            return False
        bone = trimeshFromPolyData(boneNode.GetPolyData())
        self.session = PlanSession(bone, FemGuidePlan.from_json(planJson))
        self._createEditingNodes(boneNode)
        return True

    def saveToParameterNode(self):
        if self.session is not None:
            self.getParameterNode().SetParameter("PlanJSON", self.session.plan.to_json())

    # -- plan <-> markups/transform ------------------------------------------

    @staticmethod
    def controlPoints(markupsNode):
        points = []
        for i in range(markupsNode.GetNumberOfControlPoints()):
            if markupsNode.GetNthControlPointPositionStatus(i) != markupsNode.PositionDefined:
                continue
            p = [0.0, 0.0, 0.0]
            markupsNode.GetNthControlPointPositionWorld(i, p)
            points.append(tuple(p))
        return points

    @staticmethod
    def setControlPoints(markupsNode, points):
        wasModifying = markupsNode.StartModify()
        markupsNode.RemoveAllControlPoints()
        for p in points:
            markupsNode.AddControlPointWorld(vtk.vtkVector3d(*p))
        markupsNode.EndModify(wasModifying)

    @staticmethod
    def samePoints(a, b):
        a = np.array(a, dtype=float).reshape(-1, 3)
        b = np.array(b, dtype=float).reshape(-1, 3)
        return a.shape == b.shape and np.allclose(a, b)

    def applyPlanToNodes(self):
        """Show the current plan in the markups and the placement transform."""
        plan = self.session.plan
        outline = self.node(self.OUTLINE)
        current = self.controlPoints(outline)
        wanted = list(plan.base.outline_points) if plan.base else []
        # Fewer than 3 points with no base in the plan is an outline being
        # drawn: leave the user's clicks alone.
        drawing = plan.base is None and len(current) < 3
        if not drawing and not self.samePoints(current, wanted):
            self.setControlPoints(outline, wanted)
        pins = self.node(self.PINS)
        wantedPins = [p.position for p in plan.pins]
        if not self.samePoints(self.controlPoints(pins), wantedPins):
            self.setControlPoints(pins, wantedPins)
        placement = self.node(self.PLACEMENT)
        if not np.allclose(placementMatrix(placement),
                           plan.cutting.placement_matrix):
            placement.SetMatrixTransformToParent(matrixFromNumpy(plan.cutting.placement_matrix))

    def outlineChanged(self):
        points = self.controlPoints(self.node(self.OUTLINE))
        if len(points) >= 3:
            self.session.set_outline(points)
        elif self.session.plan.base is not None:
            self.session.clear_base()

    def pinsChanged(self):
        self.session.set_pin_positions(self.controlPoints(self.node(self.PINS)))

    def placementChanged(self):
        matrix = placementMatrix(self.node(self.PLACEMENT))
        self.session.set_placement(matrix)

    # -- results -------------------------------------------------------------

    def regenerate(self):
        """Rebuild and validate the guide from the current plan and show it."""
        from femguide_core.geometry import intersection

        self.geometry = self.session.regenerate()
        self.report = self.session.validate(self.geometry)
        self._setModel(self.GUIDE, "FemGuide guide", self.geometry.guide, (0.35, 0.75, 0.45))
        try:
            cuts = intersection(self.geometry.slot, self.session.bone)
            if len(cuts.faces):
                self._setModel(self.CUTS, "Planned cuts", cuts, (0.9, 0.2, 0.2), 0.6)
        except Exception:  # the cut preview is a convenience, never a blocker
            logging.exception("FemGuide: could not build the cut preview")
        self.saveToParameterNode()
        return self.report

    def export(self, directory, acknowledged):
        path = os.path.join(directory, "{}.stl".format(self.session.plan.case_id))
        return self.session.export(path, acknowledged)


#
# Widget: the wizard
#

class FemGuideWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._syncing = False
        self._regenTimer = qt.QTimer()
        self._regenTimer.setSingleShot(True)
        self._regenTimer.setInterval(250)
        self._regenTimer.connect("timeout()", self.regenerate)
        self._placementTimer = qt.QTimer()
        self._placementTimer.setSingleShot(True)
        self._placementTimer.setInterval(400)
        self._placementTimer.connect("timeout()", self._commitPlacement)

    # -- layout --------------------------------------------------------------

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = FemGuideLogic()
        panel = qt.QWidget()
        panel.setStyleSheet("QWidget {{ font-size: {}pt; }}".format(FONT_PT))
        self.layout.addWidget(panel)
        layout = qt.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stepButtons = []
        header = qt.QHBoxLayout()
        for index, name in enumerate(STEPS):
            button = qt.QPushButton("{}. {}".format(index + 1, name))
            button.setCheckable(True)
            button.setMinimumHeight(BIG)
            button.connect("clicked()", lambda i=index: self.goToStep(i))
            header.addWidget(button)
            self.stepButtons.append(button)
        layout.addLayout(header)

        # Navigation and status sit above the page so they are always visible.
        navigation = qt.QHBoxLayout()
        self.undoButton = self._button("Undo", self.undo)
        self.redoButton = self._button("Redo", self.redo)
        self.backButton = self._button("< Back", lambda: self.goToStep(self.pages.currentIndex - 1))
        self.nextButton = self._button("Next >", lambda: self.goToStep(self.pages.currentIndex + 1))
        for b in (self.undoButton, self.redoButton, self.backButton, self.nextButton):
            navigation.addWidget(b)
        layout.addLayout(navigation)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo)):
            shortcut = qt.QShortcut(qt.QKeySequence(keys), self.parent)
            shortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)
            shortcut.connect("activated()", slot)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        self.statusLabel.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.statusLabel)

        self.pages = qt.QStackedWidget()
        layout.addWidget(self.pages)
        for build in (self._buildCasePage, self._buildBasePage, self._buildCutPage,
                      self._buildPinsPage, self._buildReviewPage, self._buildExportPage):
            page = qt.QWidget()
            build(qt.QFormLayout(page))
            self.pages.addWidget(page)

        self.advancedCheck = qt.QCheckBox("Advanced (engineer view)")
        layout.addWidget(self.advancedCheck)
        self.advancedText = qt.QPlainTextEdit()
        self.advancedText.readOnly = True
        self.advancedText.visible = False
        self.advancedCheck.connect("toggled(bool)", self._toggleAdvanced)
        layout.addWidget(self.advancedText)
        self.layout.addStretch(1)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneImported)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartSaveEvent, self.onSceneAboutToBeSaved)
        self._tryRestore()
        self.goToStep(0)

    def _button(self, text, slot):
        button = qt.QPushButton(text)
        button.setMinimumHeight(BIG)
        button.connect("clicked()", slot)
        return button

    def _spin(self, minimum, maximum, step, decimals, suffix, slot):
        spin = qt.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setMinimumHeight(BIG)
        spin.connect("valueChanged(double)", lambda value: self._edit(slot, value))
        return spin

    def _buildCasePage(self, form):
        self.packagesButton = self._button("Install required packages", self.onInstallPackages)
        form.addRow(self.packagesButton)
        self.caseIdEdit = qt.QLineEdit("CASE-001")
        self.caseIdEdit.setMinimumHeight(BIG)
        form.addRow("Case ID:", self.caseIdEdit)
        self.boneSelector = slicer.qMRMLNodeComboBox()
        self.boneSelector.nodeTypes = ["vtkMRMLModelNode", "vtkMRMLSegmentationNode"]
        self.boneSelector.addEnabled = False
        self.boneSelector.removeEnabled = False
        self.boneSelector.noneEnabled = True
        self.boneSelector.setMRMLScene(slicer.mrmlScene)
        self.boneSelector.setMinimumHeight(BIG)
        form.addRow("Femur (model or segmentation):", self.boneSelector)
        self.prepareButton = self._button("Prepare bone", self.onPrepareBone)
        form.addRow(self.prepareButton)
        self.repairLabel = qt.QLabel("")
        self.repairLabel.wordWrap = True
        form.addRow(self.repairLabel)

    def _buildBasePage(self, form):
        form.addRow(qt.QLabel("Click around the area where the guide should sit on the "
                              "bone. Drag points to adjust; the outline follows the surface."))
        form.addRow(self._button("Draw outline on bone", self.onDrawOutline))
        form.addRow(self._button("Clear outline", self.onClearOutline))
        self.baseThickness = self._spin(0.5, 10.0, 0.1, 2, " mm", self._setBaseThickness)
        form.addRow("Base thickness:", self.baseThickness)
        self.boneClearance = self._spin(-1.0, 3.0, 0.05, 2, " mm", self._setBoneClearance)
        form.addRow("Bone-fit clearance:", self.boneClearance)

    def _buildCutPage(self, form):
        self.segmentTable = qt.QTableWidget(0, 4)
        self.segmentTable.setHorizontalHeaderLabels(
            ["Angle (deg)", "Length (mm)", "Start ext (mm)", "End ext (mm)"])
        self.segmentTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.segmentTable.setMinimumHeight(4 * BIG)
        self.segmentTable.connect("cellChanged(int,int)", self.onSegmentCellChanged)
        form.addRow("Cut line segments:", self.segmentTable)
        rows = qt.QHBoxLayout()
        rows.addWidget(self._button("Add segment", self.onAddSegment))
        rows.addWidget(self._button("Remove last segment", self.onRemoveSegment))
        form.addRow(rows)
        self.bladeThickness = self._spin(0.1, 5.0, 0.01, 2, " mm", lambda v: self.logic.session.set_cutting(blade_thickness_mm=v))
        form.addRow("Saw blade thickness:", self.bladeThickness)
        self.bladeClearance = self._spin(0.0, 2.0, 0.01, 2, " mm", lambda v: self.logic.session.set_cutting(blade_clearance_mm=v))
        form.addRow("Blade clearance:", self.bladeClearance)
        self.wallThickness = self._spin(0.2, 10.0, 0.1, 2, " mm", lambda v: self.logic.session.set_cutting(wall_thickness_mm=v))
        form.addRow("Wall thickness:", self.wallThickness)
        self.slotDepth = self._spin(1.0, 40.0, 0.5, 1, " mm", lambda v: self.logic.session.set_cutting(slot_depth_mm=v))
        form.addRow("Slot depth (wall height):", self.slotDepth)
        form.addRow(self._button("Drop cut onto base", lambda: self._edit(self.logic.session.drop_onto_base)))
        self.handlesCheck = qt.QCheckBox("Show drag handles in 3D")
        self.handlesCheck.setMinimumHeight(BIG)
        self.handlesCheck.connect("toggled(bool)", self.onToggleHandles)
        form.addRow(self.handlesCheck)
        self.nudgeStep = self._spin(0.1, 10.0, 0.1, 1, " mm / deg", lambda v: None)
        self.nudgeStep.value = 1.0
        form.addRow("Nudge step:", self.nudgeStep)
        grid = qt.QGridLayout()
        for row, (label, kind) in enumerate((("Move", "translate"), ("Rotate", "rotate"))):
            grid.addWidget(qt.QLabel(label), row, 0)
            for axis, name in enumerate("XYZ"):
                for column, sign in ((1 + 2 * axis, -1), (2 + 2 * axis, 1)):
                    button = self._button("{}{}".format(name, "-" if sign < 0 else "+"),
                                          lambda k=kind, a=axis, s=sign: self.onNudge(k, a, s))
                    grid.addWidget(button, row, column)
        form.addRow(grid)
        self.placementLabel = qt.QLabel("")
        form.addRow("Placement:", self.placementLabel)

    def _buildPinsPage(self, form):
        form.addRow(qt.QLabel("Click on the base area to add fixation pins. Pins point "
                              "along the bone surface normal unless you set angles."))
        form.addRow(self._button("Place pins", self.onPlacePins))
        self.pinTable = qt.QTableWidget(0, 4)
        self.pinTable.setHorizontalHeaderLabels(["Diameter (mm)", "Auto direction", "Tilt (deg)", "Azimuth (deg)"])
        self.pinTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.pinTable.setMinimumHeight(3 * BIG)
        self.pinTable.connect("cellChanged(int,int)", self.onPinCellChanged)
        form.addRow(self.pinTable)
        form.addRow(self._button("Align pins with the cut direction", self.onAlignPins))
        form.addRow(self._button("Remove last pin", self.onRemovePin))

    def _buildReviewPage(self, form):
        self.findingsList = qt.QListWidget()
        self.findingsList.setMinimumHeight(4 * BIG)
        self.findingsList.connect("itemChanged(QListWidgetItem*)", lambda item: self._updateExportState())
        form.addRow("Validation (tick each warning to acknowledge it):", self.findingsList)
        self.metricsLabel = qt.QLabel("")
        self.metricsLabel.wordWrap = True
        form.addRow(self.metricsLabel)

    def _buildExportPage(self, form):
        self.exportDir = ctk.ctkPathLineEdit()
        self.exportDir.filters = ctk.ctkPathLineEdit.Dirs
        self.exportDir.currentPath = os.path.expanduser("~")
        form.addRow("Export folder:", self.exportDir)
        self.exportButton = self._button("Export guide (STL + trace record)", self.onExport)
        form.addRow(self.exportButton)
        self.exportLabel = qt.QLabel("")
        self.exportLabel.wordWrap = True
        form.addRow(self.exportLabel)

    # -- navigation ----------------------------------------------------------

    def goToStep(self, index):
        index = max(0, min(len(STEPS) - 1, index))
        if index > 0 and self.logic.session is None:
            index = 0
            self.statusLabel.text = "Prepare the bone first."
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.stepButtons):
            button.setChecked(i == index)
        self.backButton.enabled = index > 0
        self.nextButton.enabled = index < len(STEPS) - 1
        self._refreshPage()

    def _toggleAdvanced(self, on):
        self.advancedText.visible = on
        self._refreshPage()

    # -- editing -------------------------------------------------------------

    def _edit(self, action, *args):
        """Run a session edit triggered from the GUI, then show the result."""
        if self._syncing or self.logic.session is None:
            return
        try:
            action(*args)
        except (ValueError, NotImplementedError) as exc:
            self.statusLabel.text = "Not applied: {}".format(exc)
            self._showPlan()
            return
        # Store the plan in the scene right away, so a scene saved before
        # the guide finishes rebuilding still has the latest plan.
        self.logic.saveToParameterNode()
        self._showPlan()
        self._regenTimer.start()

    def _setBaseThickness(self, value):
        if self.logic.session.plan.base is not None:
            self.logic.session.set_base(thickness_mm=value)

    def _setBoneClearance(self, value):
        if self.logic.session.plan.base is not None:
            self.logic.session.set_base(clearance_mm=value)

    def undo(self):
        if self.logic.session is not None and self.logic.session.history.can_undo:
            self._edit(self.logic.session.undo)

    def redo(self):
        if self.logic.session is not None and self.logic.session.history.can_redo:
            self._edit(self.logic.session.redo)

    def _showPlan(self):
        """Push the current plan into the markups, transform and fields."""
        if self.logic.session is None:
            return
        self._syncing = True
        try:
            self.logic.applyPlanToNodes()
            self._refreshPage()
        finally:
            self._syncing = False
        self.undoButton.enabled = self.logic.session.history.can_undo
        self.redoButton.enabled = self.logic.session.history.can_redo

    def _refreshPage(self):
        session = self.logic.session
        self.packagesButton.visible = bool(missingPackages())
        if session is None:
            return
        wasSyncing, self._syncing = self._syncing, True
        try:
            plan = session.plan
            c = plan.cutting
            if plan.base is not None:
                self.baseThickness.value = plan.base.thickness_mm
                self.boneClearance.value = plan.base.clearance_mm
            self.bladeThickness.value = c.blade_thickness_mm
            self.bladeClearance.value = c.blade_clearance_mm
            self.wallThickness.value = c.wall_thickness_mm
            self.slotDepth.value = c.slot_depth_mm
            self.segmentTable.setRowCount(len(c.angles_deg))
            for row, values in enumerate(zip(c.angles_deg, c.lengths_mm, c.start_ext_mm, c.end_ext_mm)):
                for column, value in enumerate(values):
                    self.segmentTable.setItem(row, column, qt.QTableWidgetItem("{:g}".format(value)))
            m = c.placement_matrix
            self.placementLabel.text = "origin ({:.1f}, {:.1f}, {:.1f}) mm".format(*m[:3, 3])
            self.pinTable.setRowCount(len(plan.pins))
            for row, pin in enumerate(plan.pins):
                tilt, azimuth = session.pin_angles(row)
                self.pinTable.setItem(row, 0, qt.QTableWidgetItem("{:g}".format(pin.diameter_mm)))
                auto = qt.QTableWidgetItem()
                auto.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                auto.setCheckState(qt.Qt.Checked if pin.direction is None else qt.Qt.Unchecked)
                self.pinTable.setItem(row, 1, auto)
                self.pinTable.setItem(row, 2, qt.QTableWidgetItem("{:.1f}".format(tilt)))
                self.pinTable.setItem(row, 3, qt.QTableWidgetItem("{:.1f}".format(azimuth)))
            if self.advancedCheck.checked:
                self.advancedText.setPlainText(plan.to_json())
        finally:
            self._syncing = wasSyncing

    # -- MRML observers ------------------------------------------------------

    def _observeEditingNodes(self):
        self.removeObservers(self._onMarkups)
        self.removeObservers(self._onPlacement)
        markupEvents = (slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                        slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                        slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                        slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent)
        for role in (FemGuideLogic.OUTLINE, FemGuideLogic.PINS):
            for event in markupEvents:
                self.addObserver(self.logic.node(role), event, self._onMarkups)
        self.addObserver(self.logic.node(FemGuideLogic.PLACEMENT),
                         slicer.vtkMRMLTransformNode.TransformModifiedEvent, self._onPlacement)

    def _onMarkups(self, caller, event):
        if self._syncing or self.logic.session is None:
            return
        if caller is self.logic.node(FemGuideLogic.OUTLINE):
            self._edit(self.logic.outlineChanged)
        else:
            self._edit(self.logic.pinsChanged)

    def _onPlacement(self, caller, event):
        if not self._syncing and self.logic.session is not None:
            self._placementTimer.start()  # commit once the drag pauses

    def _commitPlacement(self):
        self._edit(self.logic.placementChanged)

    # -- actions ---------------------------------------------------------------

    def onInstallPackages(self):
        with slicer.util.tryWithErrorDisplay("Installing packages failed.", waitCursor=True):
            installPackages()
        self._refreshPage()

    def onPrepareBone(self):
        source = self.boneSelector.currentNode()
        if source is None:
            self.statusLabel.text = "Select the femur model or segmentation first."
            return
        if missingPackages():
            self.statusLabel.text = "Install the required packages first."
            return
        with slicer.util.tryWithErrorDisplay("Preparing the bone failed.", waitCursor=True):
            report = self.logic.prepareBone(source, self.caseIdEdit.text.strip() or "CASE")
            self.repairLabel.text = str(report)
            self._observeEditingNodes()
            self._showPlan()
            self.goToStep(1)

    def _placeMode(self, role):
        node = self.logic.node(role)
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetReferenceActivePlaceNodeClassName(node.GetClassName())
        selection.SetReferenceActivePlaceNodeID(node.GetID())
        slicer.app.applicationLogic().GetInteractionNode().SetPlaceModePersistence(1)
        slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.Place)

    def onDrawOutline(self):
        if self.logic.session is not None:
            self._placeMode(FemGuideLogic.OUTLINE)

    def onClearOutline(self):
        self._edit(self.logic.session.clear_base)

    def onPlacePins(self):
        if self.logic.session is not None:
            self._placeMode(FemGuideLogic.PINS)

    def onRemovePin(self):
        if self.logic.session is not None and self.logic.session.plan.pins:
            self._edit(self.logic.session.remove_pin, len(self.logic.session.plan.pins) - 1)

    def onAlignPins(self):
        session = self.logic.session
        if session is not None and session.plan.pins:
            self._edit(session.align_pins_with_cut)

    def onToggleHandles(self, on):
        node = self.logic.node(FemGuideLogic.PLACEMENT)
        if node is not None:
            node.GetDisplayNode().SetEditorVisibility(on)

    def onNudge(self, kind, axis, sign):
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * self.nudgeStep.value
        if kind == "translate":
            self._edit(self.logic.session.nudge, delta, (0.0, 0.0, 0.0))
        else:
            self._edit(self.logic.session.nudge, (0.0, 0.0, 0.0), delta)

    def _segmentValues(self):
        columns = [[], [], [], []]
        for row in range(self.segmentTable.rowCount):
            for column in range(4):
                item = self.segmentTable.item(row, column)
                columns[column].append(float(item.text()) if item and item.text() else 0.0)
        return columns

    def onSegmentCellChanged(self, row, column):
        if self._syncing:
            return
        try:
            angles, lengths, starts, ends = self._segmentValues()
        except ValueError:
            self.statusLabel.text = "Enter numbers in the segment table."
            self._showPlan()
            return
        self._edit(lambda: self.logic.session.set_cutting(
            angles_deg=angles, lengths_mm=lengths, start_ext_mm=starts, end_ext_mm=ends))

    def onAddSegment(self):
        c = self.logic.session.plan.cutting
        self._edit(lambda: self.logic.session.set_cutting(
            angles_deg=c.angles_deg + (0.0,), lengths_mm=c.lengths_mm + (10.0,),
            start_ext_mm=c.start_ext_mm + (0.0,), end_ext_mm=c.end_ext_mm + (0.0,)))

    def onRemoveSegment(self):
        c = self.logic.session.plan.cutting
        if len(c.angles_deg) > 1:
            self._edit(lambda: self.logic.session.set_cutting(
                angles_deg=c.angles_deg[:-1], lengths_mm=c.lengths_mm[:-1],
                start_ext_mm=c.start_ext_mm[:-1], end_ext_mm=c.end_ext_mm[:-1]))

    def onPinCellChanged(self, row, column):
        if self._syncing or row >= len(self.logic.session.plan.pins):
            return
        try:
            if column == 0:
                self._edit(self.logic.session.set_pin, row,
                           diameter_mm=float(self.pinTable.item(row, 0).text()))
            elif column == 1:
                if self.pinTable.item(row, 1).checkState() == qt.Qt.Checked:
                    self._edit(self.logic.session.set_pin, row, direction=None)
                else:
                    tilt, azimuth = self.logic.session.pin_angles(row)
                    self._edit(self.logic.session.set_pin_angles, row, tilt, azimuth)
            else:
                tilt = float(self.pinTable.item(row, 2).text())
                azimuth = float(self.pinTable.item(row, 3).text())
                self._edit(self.logic.session.set_pin_angles, row, tilt, azimuth)
        except ValueError:
            self.statusLabel.text = "Enter numbers in the pin table."
            self._showPlan()

    def regenerate(self):
        if self.logic.session is None:
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            report = self.logic.regenerate()
        except Exception as exc:
            logging.exception("FemGuide: regeneration failed")
            self.statusLabel.text = "Could not build the guide: {}".format(exc)
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._showReport(report)

    def _showReport(self, report):
        acknowledged = self.acknowledged()
        self.findingsList.clear()
        for finding in report.findings:
            item = qt.QListWidgetItem("[{}] {}".format(
                "BLOCK" if finding.severity.value == "block" else "WARNING", finding.message))
            item.setData(qt.Qt.UserRole, finding.code)
            if finding.severity.value == "warn":
                item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                item.setCheckState(qt.Qt.Checked if finding.code in acknowledged else qt.Qt.Unchecked)
                item.setForeground(qt.QBrush(qt.QColor(180, 110, 0)))
            else:
                item.setForeground(qt.QBrush(qt.QColor(200, 30, 30)))
            self.findingsList.addItem(item)
        timings = sum(self.logic.geometry.timings.values())
        self.metricsLabel.text = "  ".join(
            "{}: {:.1f}".format(k, v) for k, v in sorted(report.metrics.items()))
        self.statusLabel.text = "Guide rebuilt in {:.2f} s: {} block(s), {} warning(s).".format(
            timings, len(report.blocks), len(report.warnings))
        self._updateExportState()

    def acknowledged(self):
        codes = []
        for i in range(self.findingsList.count):
            item = self.findingsList.item(i)
            if item.flags() & qt.Qt.ItemIsUserCheckable and item.checkState() == qt.Qt.Checked:
                codes.append(item.data(qt.Qt.UserRole))
        return codes

    def _updateExportState(self):
        report = self.logic.report
        ok = report is not None and report.can_export(self.acknowledged())
        self.exportButton.enabled = ok
        if report is not None and not ok:
            self.exportLabel.text = ("Export is blocked until every block is resolved and "
                                     "every warning is acknowledged on the Review step.")

    def onExport(self):
        with slicer.util.tryWithErrorDisplay("Export failed.", waitCursor=True):
            record = self.logic.export(self.exportDir.currentPath, self.acknowledged())
            self.exportLabel.text = "Exported. Traceability record: {}".format(record)

    # -- scene lifecycle -----------------------------------------------------

    def _tryRestore(self):
        if missingPackages():
            return
        try:
            restored = self.logic.restoreFromParameterNode()
        except ValueError as exc:
            self.statusLabel.text = "Could not reopen the saved plan: {}".format(exc)
            return
        if restored:
            self._observeEditingNodes()
            self._showPlan()
            self._regenTimer.start()

    def onSceneClosed(self, caller, event):
        self.logic.session = None
        self.logic.report = None
        self.goToStep(0)

    def onSceneImported(self, caller, event):
        self._tryRestore()

    def onSceneAboutToBeSaved(self, caller, event):
        self.logic.saveToParameterNode()

    def cleanup(self):
        self.removeObservers()
        self._regenTimer.stop()
        self._placementTimer.stop()


#
# Self-test
#

class FemGuideTest(ScriptedLoadableModuleTest):
    """End-to-end wizard test on a synthetic bone (runs headless)."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_wizard_end_to_end()

    def test_wizard_end_to_end(self):
        import tempfile

        import trimesh

        bone = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        source = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Femur")
        source.SetAndObservePolyData(polyDataFromTrimesh(bone))
        source.CreateDefaultDisplayNodes()

        widget = slicer.modules.femguide.widgetRepresentation().self()
        logic = widget.logic
        widget.boneSelector.setCurrentNode(source)
        widget.caseIdEdit.text = "SELFTEST-1"
        widget.onPrepareBone()
        self.assertIsNotNone(logic.session)

        # Draw the base outline by placing control points, as clicks would.
        theta = np.radians(35.0)
        outline = logic.node(FemGuideLogic.OUTLINE)
        for k, phi in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False)):
            outline.AddControlPointWorld(vtk.vtkVector3d(
                20 * np.sin(theta) * np.cos(phi), 20 * np.sin(theta) * np.sin(phi),
                20 * np.cos(theta)))
            # The first clicks of an outline must survive until it closes.
            self.assertEqual(outline.GetNumberOfControlPoints(), k + 1)
        self.assertIsNotNone(logic.session.plan.base)

        # A short cut, dropped onto the base, then nudged.
        widget._edit(lambda: logic.session.set_cutting(angles_deg=(0.0, 20.0), lengths_mm=(7.0, 6.0)))
        widget._edit(logic.session.drop_onto_base)
        widget.onNudge("translate", 2, 1)
        report = logic.regenerate()
        self.assertFalse(report.blocks, str(report))
        self.assertGreater(logic.node(FemGuideLogic.GUIDE).GetPolyData().GetNumberOfPoints(), 0)

        # Placement transform and plan stay in sync.
        matrix = placementMatrix(logic.node(FemGuideLogic.PLACEMENT))
        np.testing.assert_allclose(matrix, logic.session.plan.cutting.placement_matrix, atol=1e-9)

        # Undo/redo through the widget.
        before = logic.session.plan
        widget.onNudge("rotate", 2, 1)
        widget.undo()
        self.assertEqual(logic.session.plan, before)
        widget.redo()
        self.assertNotEqual(logic.session.plan, before)
        widget.undo()

        # Save the scene to a file right after an edit (before the guide has
        # rebuilt), close it, and reopen it: the plan must come back exactly.
        widget.onNudge("translate", 0, 1)
        saved = logic.session.plan
        scenePath = os.path.join(tempfile.mkdtemp(), "case.mrb")
        self.assertTrue(slicer.util.saveScene(scenePath))
        slicer.mrmlScene.Clear(0)
        self.assertIsNone(logic.session)
        slicer.util.loadScene(scenePath)
        self.assertIsNotNone(logic.session)
        self.assertEqual(logic.session.plan, saved)

        # Export.
        report = logic.regenerate()
        directory = tempfile.mkdtemp()
        record = logic.export(directory, [f.code for f in report.warnings])
        self.assertTrue(os.path.exists(record))
        self.delayDisplay("FemGuide wizard test passed")
