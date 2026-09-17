import adsk.core
import adsk.fusion
import math
import os
import traceback

_app = None
_ui = None
_handlers = []

ADDIN_NAME = 'FlushCheck'
CMD_ID = 'femguideFlushCheck'
CLEAR_CMD_ID = 'femguideFlushCheckClear'
PANEL_ID = 'InspectPanel'
OVERLAY_ID = 'FlushCheckOverlay'

DEFAULT_SAMPLES = 2500
DEFAULT_EPSILON_CM = 0.001  # 0.01 mm, Fusion internal units are cm


def _pointImage(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'points', name)


def _clearOverlay(design):
    for comp in design.allComponents:
        groups = comp.customGraphicsGroups
        for i in range(groups.count - 1, -1, -1):
            g = groups.item(i)
            if g.id == OVERLAY_ID:
                g.deleteMe()


def _sampleFace(face, count):
    ev = face.evaluator
    prange = ev.parametricRange()
    if not prange:
        return []
    uMin, vMin = prange.minPoint.x, prange.minPoint.y
    uMax, vMax = prange.maxPoint.x, prange.maxPoint.y
    if uMax <= uMin or vMax <= vMin:
        return []
    n = max(3, int(math.sqrt(count)))
    pts = []
    for i in range(n):
        u = uMin + (uMax - uMin) * (i + 0.5) / n
        for j in range(n):
            v = vMin + (vMax - vMin) * (j + 0.5) / n
            param = adsk.core.Point2D.create(u, v)
            if not ev.isParameterOnFace(param):
                continue
            ok, p3 = ev.getPointAtParameter(param)
            if ok:
                pts.append(p3)
    return pts


def _drawPointSet(group, pts, imageName):
    if not pts:
        return
    coords = []
    for p in pts:
        coords.extend([p.x, p.y, p.z])
    cgCoords = adsk.fusion.CustomGraphicsCoordinates.create(coords)
    group.addPointSet(
        cgCoords,
        list(range(len(pts))),
        adsk.fusion.CustomGraphicsPointTypes.UserDefinedCustomGraphicsPointType,
        _pointImage(imageName))


class FlushCheckCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            cmd.isRepeatable = False
            inputs = cmd.commandInputs

            faceSel = inputs.addSelectionInput(
                'baseFaces', 'Guide base faces',
                'Select the face(s) of the guide base that should sit flush on the bone')
            faceSel.addSelectionFilter('Faces')
            faceSel.setSelectionLimits(1, 0)

            bodySel = inputs.addSelectionInput(
                'boneBody', 'Bone body', 'Select the bone body')
            bodySel.addSelectionFilter('Bodies')
            bodySel.setSelectionLimits(1, 1)

            inputs.addIntegerSpinnerCommandInput(
                'samples', 'Sample points (total)', 100, 50000, 100, DEFAULT_SAMPLES)

            inputs.addValueInput(
                'epsilon', 'Noise epsilon', 'mm',
                adsk.core.ValueInput.createByReal(DEFAULT_EPSILON_CM))

            onExecute = FlushCheckExecuteHandler()
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)
        except Exception:
            _ui.messageBox('FlushCheck failed:\n{}'.format(traceback.format_exc()))


class FlushCheckExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        progress = None
        try:
            design = adsk.fusion.Design.cast(_app.activeProduct)
            if not design:
                _ui.messageBox('No active Fusion design.')
                return

            inputs = args.command.commandInputs
            faceSel = inputs.itemById('baseFaces')
            bodySel = inputs.itemById('boneBody')
            sampleBudget = inputs.itemById('samples').value
            epsilon = inputs.itemById('epsilon').value  # cm

            faces = [faceSel.selection(i).entity for i in range(faceSel.selectionCount)]
            bone = adsk.fusion.BRepBody.cast(bodySel.selection(0).entity)
            if not bone:
                _ui.messageBox('The bone selection is not a BRep body.')
                return

            totalArea = sum(f.area for f in faces)
            if totalArea <= 0:
                _ui.messageBox('Selected faces have no area.')
                return

            pts = []
            for f in faces:
                pts.extend(_sampleFace(f, max(9, int(sampleBudget * f.area / totalArea))))
            if not pts:
                _ui.messageBox('No sample points could be generated on the selected faces.')
                return

            progress = _ui.createProgressDialog()
            progress.isCancelButtonShown = True
            progress.show('Flush Check', 'Measuring point %v of %m', 0, len(pts), 1)

            measure = _app.measureManager
            inside = adsk.fusion.PointContainment.PointInsidePointContainment
            contact, gap, penetration = [], [], []
            maxGap = 0.0
            flaggedSum = 0.0

            for idx, p in enumerate(pts):
                if progress.wasCancelled:
                    progress.hide()
                    return
                progress.progressValue = idx + 1

                d = measure.measureMinimumDistance(p, bone).value
                if d <= epsilon:
                    contact.append(p)
                    continue
                if bone.pointContainment(p) == inside:
                    penetration.append(p)
                else:
                    gap.append(p)
                flaggedSum += d
                if d > maxGap:
                    maxGap = d

            progress.hide()
            progress = None

            _clearOverlay(design)
            group = design.rootComponent.customGraphicsGroups.add()
            group.id = OVERLAY_ID
            _drawPointSet(group, contact, 'green.png')
            _drawPointSet(group, gap, 'red.png')
            _drawPointSet(group, penetration, 'magenta.png')
            _app.activeViewport.refresh()

            total = len(pts)
            flagged = len(gap) + len(penetration)
            summary = (
                'Samples: {}\n'
                'Contact (green): {}  ({:.1f}%)\n'
                'Gap (red): {}\n'
                'Penetration (magenta): {}\n'
                'Max deviation: {:.4f} mm\n'
                'Mean deviation of flagged points: {:.4f} mm'
            ).format(
                total,
                len(contact), 100.0 * len(contact) / total,
                len(gap),
                len(penetration),
                maxGap * 10.0,
                (flaggedSum / flagged * 10.0) if flagged else 0.0)
            _ui.messageBox(summary, 'Flush Check Result')
        except Exception:
            if progress:
                progress.hide()
            _ui.messageBox('FlushCheck failed:\n{}'.format(traceback.format_exc()))


class ClearCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            onExecute = ClearExecuteHandler()
            args.command.execute.add(onExecute)
            _handlers.append(onExecute)
        except Exception:
            _ui.messageBox('FlushCheck failed:\n{}'.format(traceback.format_exc()))


class ClearExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            design = adsk.fusion.Design.cast(_app.activeProduct)
            if design:
                _clearOverlay(design)
                _app.activeViewport.refresh()
        except Exception:
            _ui.messageBox('FlushCheck failed:\n{}'.format(traceback.format_exc()))


def run(context):
    global _app, _ui
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface

        for cmdId in (CMD_ID, CLEAR_CMD_ID):
            existing = _ui.commandDefinitions.itemById(cmdId)
            if existing:
                existing.deleteMe()

        checkDef = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID, 'Flush Check',
            'Verify that the guide base is flush with the bone')
        clearDef = _ui.commandDefinitions.addButtonDefinition(
            CLEAR_CMD_ID, 'Clear Flush Check',
            'Remove the Flush Check overlay from the viewport')

        onCheckCreated = FlushCheckCreatedHandler()
        checkDef.commandCreated.add(onCheckCreated)
        _handlers.append(onCheckCreated)

        onClearCreated = ClearCreatedHandler()
        clearDef.commandCreated.add(onClearCreated)
        _handlers.append(onClearCreated)

        panel = _ui.allToolbarPanels.itemById(PANEL_ID)
        if panel:
            for cmdId, definition in ((CMD_ID, checkDef), (CLEAR_CMD_ID, clearDef)):
                if panel.controls.itemById(cmdId):
                    panel.controls.itemById(cmdId).deleteMe()
                panel.controls.addCommand(definition)
        else:
            _ui.messageBox('FlushCheck: Inspect panel not found; '
                           'run the commands from the shortcut search (S key).')
    except Exception:
        if _ui:
            _ui.messageBox('FlushCheck failed to start:\n{}'.format(traceback.format_exc()))


def stop(context):
    try:
        design = adsk.fusion.Design.cast(_app.activeProduct) if _app else None
        if design:
            _clearOverlay(design)

        if _ui:
            panel = _ui.allToolbarPanels.itemById(PANEL_ID)
            for cmdId in (CMD_ID, CLEAR_CMD_ID):
                if panel:
                    control = panel.controls.itemById(cmdId)
                    if control:
                        control.deleteMe()
                definition = _ui.commandDefinitions.itemById(cmdId)
                if definition:
                    definition.deleteMe()
    except Exception:
        if _ui:
            _ui.messageBox('FlushCheck failed to stop:\n{}'.format(traceback.format_exc()))
