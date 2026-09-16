import math
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import ctk
import numpy as np
import os

# ---------------- Polyline Generator ----------------
def generatePolyline(angle_increments, lengths):
    if len(angle_increments) != len(lengths):
        raise ValueError("Lengths list must match angle increments list in size.")
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    current_angle = 0.0
    x, y = 0.0, 0.0
    points.InsertNextPoint(x, y, 0.0)
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(len(angle_increments) + 1)
    polyLine.GetPointIds().SetId(0, 0)
    for i, (inc, length) in enumerate(zip(angle_increments, lengths)):
        current_angle += math.radians(inc)
        x += length * math.cos(current_angle)
        y += length * math.sin(current_angle)
        points.InsertNextPoint(x, y, 0.0)
        polyLine.GetPointIds().SetId(i + 1, i + 1)
    lines.InsertNextCell(polyLine)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata

def computePolylinePlaneNormal(polydata):
    n = polydata.GetNumberOfPoints()
    if n < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    p0 = np.array(polydata.GetPoint(0), dtype=float)
    p1 = np.array(polydata.GetPoint(1), dtype=float)
    p2 = np.array(polydata.GetPoint(2), dtype=float)
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm

def safe_normalize(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array(fallback, dtype=float)
    return v / n

# ---------------- Module ----------------
class LineGenerator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Line Generator"
        parent.categories = ["Examples"]
        parent.contributors = ["Radu Josanu"]
        parent.helpText = "Generate a polyline with extrusion, secondary and tertiary lines."
        parent.acknowledgementText = ""

class LineGeneratorWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # --- Main UI ---
        self.mainCollapsibleButton = ctk.ctkCollapsibleButton()
        self.mainCollapsibleButton.text = "Line Generator"
        self.layout.addWidget(self.mainCollapsibleButton)
        self.formLayout = qt.QFormLayout(self.mainCollapsibleButton)
        # Angles and lengths
        self.angleEdit = qt.QLineEdit("8,40,45,45,42")
        self.lengthEdit = qt.QLineEdit("50,20.5,25,11,30")
        self.formLayout.addRow("Angle increments (deg):", self.angleEdit)
        self.formLayout.addRow("Lengths:", self.lengthEdit)
        self.startExtEdit = qt.QLineEdit()
        self.startExtEdit.setToolTip("Comma-separated start extension for each segment")
        self.layout.addWidget(qt.QLabel("Start Extensions"))
        self.layout.addWidget(self.startExtEdit)
        self.endExtEdit = qt.QLineEdit()
        self.endExtEdit.setToolTip("Comma-separated end extension for each segment")
        self.layout.addWidget(qt.QLabel("End Extensions"))
        self.layout.addWidget(self.endExtEdit)
        # Primary line thickness / extrusion
        self.lineThicknessSpin = qt.QDoubleSpinBox()
        self.lineThicknessSpin.setRange(0.1, 50)
        self.lineThicknessSpin.setValue(1.0)
        self.formLayout.addRow("Primary Thickness (one direction):", self.lineThicknessSpin)
        self.extrudeSpin = qt.QDoubleSpinBox()
        self.extrudeSpin.setRange(0, 1000)
        self.extrudeSpin.setValue(5.0)
        self.formLayout.addRow("Primary Extrusion Length:", self.extrudeSpin)
        # Secondary line parameters
        self.secondaryCheck = qt.QCheckBox()
        self.secondaryCheck.setChecked(False)
        self.formLayout.addRow("Generate Secondary Line:", self.secondaryCheck)
        self.secondaryIndexSpin = qt.QSpinBox(); self.secondaryIndexSpin.setRange(0, 100); self.secondaryIndexSpin.setValue(0)
        self.formLayout.addRow("Secondary Segment Index:", self.secondaryIndexSpin)
        self.secondaryLengthSpin = qt.QDoubleSpinBox(); self.secondaryLengthSpin.setRange(0.1, 500); self.secondaryLengthSpin.setValue(20.0)
        self.formLayout.addRow("Secondary Length:", self.secondaryLengthSpin)
        self.secondaryExtrudeSpin = qt.QDoubleSpinBox(); self.secondaryExtrudeSpin.setRange(0.1, 100); self.secondaryExtrudeSpin.setValue(2.0)
        self.formLayout.addRow("Secondary Extrusion:", self.secondaryExtrudeSpin)
        self.secondaryOffsetSpin = qt.QDoubleSpinBox(); self.secondaryOffsetSpin.setRange(-500, 500); self.secondaryOffsetSpin.setValue(0.0)
        self.formLayout.addRow("Secondary Offset:", self.secondaryOffsetSpin)
        self.secondaryThicknessSpin = qt.QDoubleSpinBox(); self.secondaryThicknessSpin.setRange(0.1, 50); self.secondaryThicknessSpin.setValue(1.0)
        self.formLayout.addRow("Secondary Width (in-plane):", self.secondaryThicknessSpin)
        # Tertiary line parameters
        self.tertiaryCheck = qt.QCheckBox()
        self.tertiaryCheck.setChecked(False)
        self.formLayout.addRow("Generate Tertiary Line:", self.tertiaryCheck)
        self.tertiaryIndexSpin = qt.QSpinBox(); self.tertiaryIndexSpin.setRange(0, 100); self.tertiaryIndexSpin.setValue(0)
        self.formLayout.addRow("Tertiary Segment Index:", self.tertiaryIndexSpin)
        self.tertiaryLengthSpin = qt.QDoubleSpinBox(); self.tertiaryLengthSpin.setRange(0.1, 500); self.tertiaryLengthSpin.setValue(20.0)
        self.formLayout.addRow("Tertiary Length:", self.tertiaryLengthSpin)
        self.tertiaryExtrudeSpin = qt.QDoubleSpinBox(); self.tertiaryExtrudeSpin.setRange(0.1, 100); self.tertiaryExtrudeSpin.setValue(2.0)
        self.formLayout.addRow("Tertiary Extrusion:", self.tertiaryExtrudeSpin)
        self.tertiaryOffsetSpin = qt.QDoubleSpinBox(); self.tertiaryOffsetSpin.setRange(-500, 500); self.tertiaryOffsetSpin.setValue(0.0)
        self.formLayout.addRow("Tertiary Offset:", self.tertiaryOffsetSpin)
        self.tertiaryThicknessSpin = qt.QDoubleSpinBox(); self.tertiaryThicknessSpin.setRange(0.1, 50); self.tertiaryThicknessSpin.setValue(1.0)
        self.formLayout.addRow("Tertiary Width (in-plane):", self.tertiaryThicknessSpin)
        # Cylinder parameters
        self.cylinderCheck = qt.QCheckBox()
        self.cylinderCheck.setChecked(False)
        self.formLayout.addRow("Generate Cylinders:", self.cylinderCheck)
        self.cylinderSegIndexSpin = qt.QSpinBox(); self.cylinderSegIndexSpin.setRange(0, 100); self.cylinderSegIndexSpin.setValue(0)
        self.formLayout.addRow("Cylinder Segment Index:", self.cylinderSegIndexSpin)
        self.cylinderOffsetAlongSpin = qt.QDoubleSpinBox()
        self.cylinderOffsetAlongSpin.setRange(0.0, 1.0)
        self.cylinderOffsetAlongSpin.setSingleStep(0.05)
        self.cylinderOffsetAlongSpin.setValue(0.5)
        self.cylinderOffsetAlongSpin.setToolTip("Fractional position along segment (0 = start, 1 = end)")
        self.formLayout.addRow("Cylinder Offset Along Segment [0-1]:", self.cylinderOffsetAlongSpin)
        self.cylinderOffsetFromPlaneSpin = qt.QDoubleSpinBox(); self.cylinderOffsetFromPlaneSpin.setRange(0.0, 500); self.cylinderOffsetFromPlaneSpin.setValue(5.0)
        self.formLayout.addRow("Cylinder Offset From Plane (±):", self.cylinderOffsetFromPlaneSpin)
        self.cylinderRadiusSpin = qt.QDoubleSpinBox(); self.cylinderRadiusSpin.setRange(0.1, 100); self.cylinderRadiusSpin.setValue(2.0)
        self.formLayout.addRow("Cylinder Radius:", self.cylinderRadiusSpin)
        self.cylinderHeightSpin = qt.QDoubleSpinBox(); self.cylinderHeightSpin.setRange(0.1, 500); self.cylinderHeightSpin.setValue(10.0)
        self.formLayout.addRow("Cylinder Height:", self.cylinderHeightSpin)
        self.applyButton = qt.QPushButton("Generate Polyline")
        self.applyButton.clicked.connect(self.apply)
        self.formLayout.addRow(self.applyButton)
        # Model names
        self.modelNodeName = "LineGeneratorModel"
        self.secondaryModelName = "LineGeneratorSecondary"
        self.tertiaryModelName = "LineGeneratorTertiary"
        self.cylinderModelName = "LineGeneratorCylinders"

    # ---------------- Primary prism ----------------
    def build_primary_prism(self, polydata, normal, thickness, extrusion, startExt=None, endExt=None):
        if startExt is None:
            startExt = [0.0] * (polydata.GetNumberOfPoints() - 1)
        if endExt is None:
            endExt = [0.0] * (polydata.GetNumberOfPoints() - 1)
        npts = polydata.GetNumberOfPoints()
        points = vtk.vtkPoints()
        quads = vtk.vtkCellArray()
        for i in range(npts - 1):
            p0 = np.array(polydata.GetPoint(i))
            p1 = np.array(polydata.GetPoint(i + 1))
            segVec = safe_normalize(p1 - p0)
            p0 = p0 - segVec * startExt[i]
            p1 = p1 + segVec * endExt[i]
            thickVec = np.cross(normal, segVec)
            thickVec = -safe_normalize(thickVec) * thickness
            pA = p0
            pB = p0 + thickVec
            pC = p1 + thickVec
            pD = p1
            baseId = points.GetNumberOfPoints()
            points.InsertNextPoint(pA)
            points.InsertNextPoint(pB)
            points.InsertNextPoint(pC)
            points.InsertNextPoint(pD)
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, baseId)
            quad.GetPointIds().SetId(1, baseId + 1)
            quad.GetPointIds().SetId(2, baseId + 2)
            quad.GetPointIds().SetId(3, baseId + 3)
            quads.InsertNextCell(quad)
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)
        pd.SetPolys(quads)
        extruder = vtk.vtkLinearExtrusionFilter()
        extruder.SetInputData(pd)
        extruder.SetVector(*normal)
        extruder.SetScaleFactor(extrusion)
        extruder.SetExtrusionTypeToVectorExtrusion()
        extruder.CappingOn()
        extruder.Update()
        tf = vtk.vtkTransform()
        tf.Translate(-normal * extrusion / 2.0)
        tfFilter = vtk.vtkTransformPolyDataFilter()
        tfFilter.SetInputData(extruder.GetOutput())
        tfFilter.SetTransform(tf)
        tfFilter.Update()
        return tfFilter.GetOutput()

    # ---------------- Offset prism (secondary/tertiary) ----------------
    def build_offset_prism(self, polydata, normal, segIndex, length, width_inplane, extrude_outofplane, offset, targetNodeName):
        if segIndex < 0 or segIndex >= polydata.GetNumberOfPoints() - 1:
            return None
        p0 = np.array(polydata.GetPoint(segIndex))
        p1 = np.array(polydata.GetPoint(segIndex + 1))
        segVec = safe_normalize(p1 - p0)
        planeVec = safe_normalize(np.cross(normal, segVec))
        centerPoint = (p0 + p1) / 2 + planeVec * offset
        halfLengthVec = segVec * (length / 2.0)
        halfWidthVec = planeVec * (width_inplane / 2.0)
        pA = centerPoint - halfLengthVec - halfWidthVec
        pB = centerPoint + halfLengthVec - halfWidthVec
        pC = centerPoint + halfLengthVec
        pD = centerPoint - halfLengthVec
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(pA)
        pts.InsertNextPoint(pB)
        pts.InsertNextPoint(pC)
        pts.InsertNextPoint(pD)
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, 0); quad.GetPointIds().SetId(1, 1)
        quad.GetPointIds().SetId(2, 2); quad.GetPointIds().SetId(3, 3)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)
        basePoly = vtk.vtkPolyData()
        basePoly.SetPoints(pts)
        basePoly.SetPolys(cells)
        extruder = vtk.vtkLinearExtrusionFilter()
        extruder.SetInputData(basePoly)
        extruder.SetVector(*normal)
        extruder.SetScaleFactor(extrude_outofplane)
        extruder.SetExtrusionTypeToVectorExtrusion()
        extruder.CappingOn()
        extruder.Update()
        tf = vtk.vtkTransform()
        tf.Translate(-normal * extrude_outofplane / 2.0)
        tfFilter = vtk.vtkTransformPolyDataFilter()
        tfFilter.SetInputData(extruder.GetOutput())
        tfFilter.SetTransform(tf)
        tfFilter.Update()
        result = tfFilter.GetOutput()
        oldNode = slicer.mrmlScene.GetFirstNodeByName(targetNodeName)
        if oldNode:
            slicer.mrmlScene.RemoveNode(oldNode)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", targetNodeName)
        node.CreateDefaultDisplayNodes()
        node.SetAndObservePolyData(result)
        node.GetDisplayNode().SetVisibility(True)
        return node

    # ---------------- Cylinder helpers ----------------
    def _make_oriented_cylinder(self, center, x_local, y_local, z_local, radius, height):
        """Create a cylinder oriented so its axis aligns with y_local, centered at center."""
        src = vtk.vtkCylinderSource()
        src.SetRadius(radius)
        src.SetHeight(height)
        src.SetResolution(20)
        src.SetCenter(0, 0, 0)
        src.Update()
        # Build 4x4 transform: columns are local X, Y, Z axes expressed in world coords,
        # translation column is the world-space center.
        mat = vtk.vtkMatrix4x4()
        mat.SetElement(0, 0, x_local[0]); mat.SetElement(0, 1, y_local[0]); mat.SetElement(0, 2, z_local[0]); mat.SetElement(0, 3, center[0])
        mat.SetElement(1, 0, x_local[1]); mat.SetElement(1, 1, y_local[1]); mat.SetElement(1, 2, z_local[1]); mat.SetElement(1, 3, center[1])
        mat.SetElement(2, 0, x_local[2]); mat.SetElement(2, 1, y_local[2]); mat.SetElement(2, 2, z_local[2]); mat.SetElement(2, 3, center[2])
        mat.SetElement(3, 0, 0);          mat.SetElement(3, 1, 0);          mat.SetElement(3, 2, 0);          mat.SetElement(3, 3, 1)
        tf = vtk.vtkTransform()
        tf.SetMatrix(mat)
        tfFilter = vtk.vtkTransformPolyDataFilter()
        tfFilter.SetInputConnection(src.GetOutputPort())
        tfFilter.SetTransform(tf)
        tfFilter.Update()
        return tfFilter.GetOutput()

    def build_cylinders(self, polydata, normal, segIndex, offset_along, radius, height, offset_from_plane, targetNodeName):
        """Build two cylinders symmetrically above/below the polyline plane on a chosen segment."""
        if segIndex < 0 or segIndex >= polydata.GetNumberOfPoints() - 1:
            return None
        p0 = np.array(polydata.GetPoint(segIndex))
        p1 = np.array(polydata.GetPoint(segIndex + 1))
        segVec = safe_normalize(p1 - p0)
        # Anchor point at fractional position along segment
        anchor = p0 + offset_along * (p1 - p0)
        # Local frame: cylinder axis is segVec (maps to VTK cylinder's Y axis)
        y_local = segVec
        z_local = safe_normalize(normal)
        x_local = safe_normalize(np.cross(y_local, z_local))
        z_local = safe_normalize(np.cross(x_local, y_local))  # re-orthogonalize
        # Two centers: symmetric above and below the polyline plane
        center1 = anchor + normal * offset_from_plane
        center2 = anchor - normal * offset_from_plane
        appender = vtk.vtkAppendPolyData()
        for center in (center1, center2):
            cyl = self._make_oriented_cylinder(center, x_local, y_local, z_local, radius, height)
            appender.AddInputData(cyl)
        appender.Update()
        oldNode = slicer.mrmlScene.GetFirstNodeByName(targetNodeName)
        if oldNode:
            slicer.mrmlScene.RemoveNode(oldNode)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", targetNodeName)
        node.CreateDefaultDisplayNodes()
        node.SetAndObservePolyData(appender.GetOutput())
        node.GetDisplayNode().SetVisibility(True)
        return node

    def apply(self):
        try:
            angles = [float(a.strip()) for a in self.angleEdit.text.split(",") if a.strip()]
            lengths = [float(l.strip()) for l in self.lengthEdit.text.split(",") if l.strip()]
        except Exception:
            return
        if len(angles) != len(lengths) or not angles:
            return
        numSegments = len(angles)
        try:
            startExt = [float(v.strip()) for v in self.startExtEdit.text.split(",") if v.strip()]
        except Exception:
            startExt = []
        if len(startExt) != numSegments:
            startExt = [0.0] * numSegments
        try:
            endExt = [float(v.strip()) for v in self.endExtEdit.text.split(",") if v.strip()]
        except Exception:
            endExt = []
        if len(endExt) != numSegments:
            endExt = [0.0] * numSegments
        poly = generatePolyline(angles, lengths)
        normal = computePolylinePlaneNormal(poly)
        primaryPoly = self.build_primary_prism(poly, normal, self.lineThicknessSpin.value, self.extrudeSpin.value, startExt, endExt)
        primaryNode = slicer.mrmlScene.GetFirstNodeByName(self.modelNodeName)
        if primaryNode is None:
            primaryNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.modelNodeName)
            primaryNode.CreateDefaultDisplayNodes()
        primaryNode.SetAndObservePolyData(primaryPoly)
        primaryNode.GetDisplayNode().SetVisibility(True)
        transformNodeName = self.modelNodeName + "_Transform"
        transformNode = slicer.mrmlScene.GetFirstNodeByName(transformNodeName)
        if transformNode is None:
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", transformNodeName)
        primaryNode.SetAndObserveTransformNodeID(transformNode.GetID())
        if self.secondaryCheck.isChecked():
            secondaryNode = self.build_offset_prism(
                poly, normal,
                int(self.secondaryIndexSpin.value),
                self.secondaryLengthSpin.value,
                self.secondaryThicknessSpin.value,
                self.secondaryExtrudeSpin.value,
                self.secondaryOffsetSpin.value,
                self.secondaryModelName
            )
            if secondaryNode:
                secondaryNode.SetAndObserveTransformNodeID(transformNode.GetID())
                secondaryNode.GetDisplayNode().SetVisibility(True)
        if self.tertiaryCheck.isChecked():
            tertiaryNode = self.build_offset_prism(
                poly, normal,
                int(self.tertiaryIndexSpin.value),
                self.tertiaryLengthSpin.value,
                self.tertiaryThicknessSpin.value,
                self.tertiaryExtrudeSpin.value,
                self.tertiaryOffsetSpin.value,
                self.tertiaryModelName
            )
            if tertiaryNode:
                tertiaryNode.SetAndObserveTransformNodeID(transformNode.GetID())
                tertiaryNode.GetDisplayNode().SetVisibility(True)
        if self.cylinderCheck.isChecked():
            cylNode = self.build_cylinders(
                poly, normal,
                int(self.cylinderSegIndexSpin.value),
                self.cylinderOffsetAlongSpin.value,
                self.cylinderRadiusSpin.value,
                self.cylinderHeightSpin.value,
                self.cylinderOffsetFromPlaneSpin.value,
                self.cylinderModelName
            )
            if cylNode:
                cylNode.SetAndObserveTransformNodeID(transformNode.GetID())
                cylNode.GetDisplayNode().SetVisibility(True)
