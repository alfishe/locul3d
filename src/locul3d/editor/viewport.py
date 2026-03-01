"""Editor-specific viewport extending base with annotation capabilities."""

import numpy as np
from typing import List, Optional

from PySide6.QtCore import Qt, Signal

from ..core.constants import COLORS, TOOL_SELECT
from ..core.geometry import BBoxItem, PlaneItem
from ..core.layer import LayerManager
from ..rendering.gl.viewport import BaseGLViewport
from ..rendering.gizmos import GizmoSystem

try:
    from OpenGL.GL import *
    from OpenGL.GL import GL_LIGHTING
except ImportError:
    pass


class EditorViewport(BaseGLViewport):
    """Viewport with annotation editing capabilities."""

    # Signals
    bbox_selected = Signal(int)
    bbox_moved = Signal(int)
    point_picked = Signal(float, float, float)

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(layer_manager, parent)

        # Annotation data
        self.annotations: List[BBoxItem] = []
        self.planes: List[PlaneItem] = []
        self.selected_idx: int = -1

        # Tool state
        self.tool = TOOL_SELECT
        self.axis_constraint = None  # 0=X, 1=Y, 2=Z, None=free

        # Gizmo system
        self.gizmo = GizmoSystem()

        # Reference point
        self.ref_point: Optional[np.ndarray] = None

    # --- Rendering Overrides ---

    def _draw_global_overlays(self):
        """Save the pre-correction modelview matrix for later use."""
        from OpenGL.GL import glGetFloatv, GL_MODELVIEW_MATRIX
        self._pre_correction_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

    def _paintGL_inner(self):
        """Paint editor viewport with annotations overlay."""
        super()._paintGL_inner()
        
        # Draw annotations
        self._draw_annotations()
        
        # Draw gizmo for selected bbox
        if self.selected_idx >= 0 and self.selected_idx < len(self.annotations):
            bbox = self.annotations[self.selected_idx]
            self.gizmo.draw_move_gizmo(bbox.center_pos, self._gizmo_len(bbox), self.gizmo.hovered_gizmo)
            self.gizmo.draw_rotate_gizmo(bbox.center_pos, bbox.size, self.gizmo.hovered_gizmo)

        # Scene-coord planes: drawn inside correction (already active)
        scene_planes = [p for p in self.planes if p.visible and not p.global_coords]
        if scene_planes:
            self._draw_planes(scene_planes)

        # Global-coord planes: drawn after scene using saved pre-correction matrix
        global_planes = [p for p in self.planes if p.visible and p.global_coords]
        if global_planes and hasattr(self, '_pre_correction_matrix'):
            glPushMatrix()
            from OpenGL.GL import glLoadMatrixf
            glLoadMatrixf(self._pre_correction_matrix)
            self._draw_planes(global_planes)
            glPopMatrix()

        # Draw reference point
        if self.ref_point is not None:
            self._draw_ref_point()

    def _draw_annotations(self):
        """Draw bounding box annotations."""
        if not self.annotations:
            return

        from ..core.constants import AABB_EDGES
        
        try:
            from OpenGL.GL import glDisable, glDepthTest, glLineWidth, glBegin, glEnd
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDepthTest(GL_FALSE)

        for i, bbox in enumerate(self.annotations):
            if not bbox.visible:
                continue
            
            selected = (i == self.selected_idx)
            glLineWidth(3.0 if selected else 1.5)
            
            if selected:
                glColor4f(1.0, 1.0, 0.0, 1.0)
            else:
                r, g, b = bbox.color[:3]
                glColor4f(r, g, b, 0.85)
            
            corners = bbox.corners()
            glBegin(GL_LINES)
            for a, b in AABB_EDGES:
                glVertex3dv(corners[a])
                glVertex3dv(corners[b])
            glEnd()

        glDepthTest(GL_TRUE)
        glEnable(GL_LIGHTING)

    def _draw_planes(self, planes=None):
        """Draw reference planes."""
        if planes is None:
            planes = self.planes
        if not planes:
            return

        try:
            from OpenGL.GL import glDisable, glEnable, GL_BLEND, glBlendFunc
            from OpenGL.GL import glBegin, glEnd, GL_QUADS, GL_LINE_LOOP
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        from OpenGL.GL import glDepthMask, GL_DEPTH_TEST
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)

        for plane in planes:
            if not plane.visible:
                continue
            
            r, g, b = plane.color[:3]
            corners = plane.corners()

            # Filled quad
            glColor4f(r, g, b, plane.opacity)
            glBegin(GL_QUADS)
            for c in corners:
                glVertex3dv(c)
            glEnd()

            # Wireframe border
            glColor4f(r, g, b, min(1.0, plane.opacity + 0.4))
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            for c in corners:
                glVertex3dv(c)
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_ref_point(self):
        """Draw reference point marker."""
        if self.ref_point is None:
            return

        try:
            from OpenGL.GL import glDisable, glLineWidth, glBegin, glEnd
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDepthTest(GL_FALSE)
        glLineWidth(2.5)

        rp = self.ref_point
        sz = max(0.1, self.cam_distance * 0.01)

        from ..core.constants import AXIS_COLORS
        
        for axis in range(3):
            r, g, b = AXIS_COLORS[axis]
            glColor4f(r, g, b, 0.9)
            glBegin(GL_LINES)
            glVertex3dv(rp)
            glVertex3dv(rp + np.array([sz if axis == 0 else 0,
                                                 sz if axis == 1 else 0,
                                                 sz if axis == 2 else 0]))
            glEnd()

        glDepthTest(GL_TRUE)
        glEnable(GL_LIGHTING)

    def _gizmo_len(self, bbox):
        return max(0.3, float(np.max(bbox.size)) * 0.6)
