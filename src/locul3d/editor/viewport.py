"""Editor-specific viewport extending base with annotation capabilities."""

import math
import numpy as np
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent

from ..core.constants import COLORS, TOOL_SELECT, TOOL_MOVE, TOOL_ROTATE, TOOL_SCALE
from ..core.constants import AXIS_COLORS, AABB_EDGES, AABB_FACES, GIZMO_HIT_PX
from ..core.geometry import BBoxItem, GapItem, PlaneItem
from ..core.layer import LayerManager
from ..rendering.gl.viewport import BaseGLViewport
from ..utils.math import project_to_screen, project_points_to_screen, ray_from_mouse, ray_aabb_intersect

try:
    from OpenGL.GL import *
    from OpenGL.GL import GL_LIGHTING
    from OpenGL.GLU import gluProject
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


class EditorViewport(BaseGLViewport):
    """Viewport with annotation editing capabilities."""

    # Signals
    bbox_selected = Signal(int)
    bbox_moved = Signal(int)
    point_picked = Signal(float, float, float)
    transform_committed = Signal(int, dict)  # idx, {center, size, rotation_z}

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(layer_manager, parent)

        # Annotation data
        self.annotations: List[BBoxItem] = []
        self.planes: List[PlaneItem] = []
        self.gaps: List[GapItem] = []
        self.scene_bboxes: List[BBoxItem] = []  # pipeline bboxes in scene coords
        self.selected_idx: int = -1

        # Tool state
        self.tool = TOOL_SELECT
        self.axis_constraint = None  # 0=X, 1=Y, 2=Z, None=free

        # Reference point
        self.ref_point: Optional[np.ndarray] = None
        self._picking_ref_point = False

        # Gizmo hover state
        self._hovered_gizmo = None  # ('move',axis,sign)|('scale',axis,sign)|('rotate',2,0)|None

        # Drag state
        self._drag_mode: Optional[str] = None  # 'move'|'rotate'|'gizmo_move'|'gizmo_scale'|'gizmo_rotate'|None
        self._drag_start = None
        self._drag_orig_center = None
        self._drag_orig_size = None
        self._drag_orig_rot = 0.0
        self._drag_plane_z = 0.0
        self._drag_axis: int = 0
        self._drag_sign: int = 1

        # Saved GL matrices (updated each paint)
        self._gl_modelview = None
        self._gl_projection = None
        self._gl_viewport = None

        self.setMouseTracking(True)  # needed for hover detection
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # When True, scale handles move only one face (opposite corner is anchored)
        self.scale_from_corner = False

    # ------------------------------------------------------------------
    # State Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all editor viewport state.

        Propagates through the hierarchy: BaseGLViewport clears VBOs,
        scene correction, clip planes, and panorama; this layer clears
        annotations, planes, gaps, gizmos, and selection state.
        """
        super().reset()
        self.annotations.clear()
        self.planes.clear()
        self.gaps.clear()
        self.scene_bboxes.clear()
        self.selected_idx = -1
        self.ref_point = None
        self._picking_ref_point = False
        self._hovered_gizmo = None
        self._drag_mode = None
        self._drag_start = None
        self._drag_orig_center = None
        self._drag_orig_size = None
        self._drag_orig_rot = 0.0

    # ------------------------------------------------------------------
    # Rendering Overrides
    # ------------------------------------------------------------------

    def _draw_global_overlays(self):
        """Save the pre-correction modelview matrix for later use."""
        from OpenGL.GL import glGetFloatv, GL_MODELVIEW_MATRIX
        self._pre_correction_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

    def _paintGL_inner(self):
        """Paint editor viewport with annotations overlay."""
        # Reset GL state that QPainter (gap labels) may have corrupted
        # on the previous frame. This ensures the fixed-function VBO
        # pipeline works correctly for point cloud rendering.
        try:
            from OpenGL.GL import (glUseProgram, glBindBuffer, GL_ARRAY_BUFFER,
                                   GL_ELEMENT_ARRAY_BUFFER, glDisable, glEnable,
                                   GL_BLEND, GL_TEXTURE_2D, GL_DEPTH_TEST,
                                   glDisableClientState, GL_VERTEX_ARRAY,
                                   GL_COLOR_ARRAY, GL_NORMAL_ARRAY,
                                   GL_TEXTURE_COORD_ARRAY, glActiveTexture,
                                   GL_TEXTURE0, glBindTexture)
            # Unbind QPainter's VAO — it captures vertex attribute state
            # that breaks the fixed-function VBO pipeline.
            try:
                from OpenGL.GL import glBindVertexArray
                glBindVertexArray(0)
            except (ImportError, Exception):
                pass
            glUseProgram(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)
        except ImportError:
            pass
        super()._paintGL_inner()

        # In immersive panorama mode, skip all annotations, bboxes,
        # gizmos, planes, and reference points — only the point cloud
        # and panorama sphere are rendered.
        if self._panorama and self._panorama.is_active:
            return

        # Annotations and gizmos live in world (global) coordinates.
        # Render them using the pre-correction modelview matrix so the
        # GL scene correction does NOT apply a second time.
        from OpenGL.GL import (glGetFloatv, glGetIntegerv, glPushMatrix,
                               glPopMatrix, glLoadMatrixf,
                               GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT)

        pre_corr = getattr(self, '_pre_correction_matrix', None)

        # For hit testing we need the world-space (pre-correction) matrices
        # since bboxes are now in world coordinates.
        if pre_corr is not None:
            self._gl_modelview = np.array(pre_corr, dtype=np.float64)
        else:
            self._gl_modelview = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float64)
        self._gl_projection = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float64)
        self._gl_viewport = glGetIntegerv(GL_VIEWPORT)

        # Switch to world-space matrix for annotation rendering
        if pre_corr is not None:
            glPushMatrix()
            glLoadMatrixf(pre_corr)

        # Draw annotations (bboxes are in world coordinates)
        self._draw_annotations()

        # Draw gizmo for selected bbox (move arrows + scale handles + rotation ring)
        if self.selected_idx >= 0 and self.selected_idx < len(self.annotations):
            bbox = self.annotations[self.selected_idx]
            self._draw_gizmo(bbox)

        # Draw reference point (in world space)
        if self.ref_point is not None:
            self._draw_ref_point()

        if pre_corr is not None:
            glPopMatrix()

        # Pipeline bboxes + gap brackets live in scene (post-correction) coordinates
        self._draw_scene_bboxes()
        self._draw_gap_annotations()

        # Scene-coord planes: drawn inside correction (already active)
        scene_planes = [p for p in self.planes if p.visible and not p.global_coords]
        if scene_planes:
            self._draw_planes(scene_planes)

        # Global-coord planes: drawn after scene using saved pre-correction matrix
        global_planes = [p for p in self.planes if p.visible and p.global_coords]
        if global_planes and pre_corr is not None:
            glPushMatrix()
            glLoadMatrixf(pre_corr)
            self._draw_planes(global_planes)
            glPopMatrix()


    def _draw_annotations(self):
        """Draw bounding box annotations (wireframe + optional filled faces)."""
        if not self.annotations:
            return

        try:
            from OpenGL.GL import (glDisable, glEnable, glLineWidth, glBegin, glEnd,
                                   GL_DEPTH_TEST, GL_BLEND, glBlendFunc,
                                   GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                   glDepthMask, GL_FALSE, GL_TRUE)
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # --- Pass 1: Filled faces (back-to-front is approximate but acceptable) ---
        has_fills = any(b.visible and b.fill_opacity > 0 for b in self.annotations)
        if has_fills:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)

            for i, bbox in enumerate(self.annotations):
                if not bbox.visible or bbox.fill_opacity <= 0 or getattr(bbox, 'scene_coords', False):
                    continue
                r, g, b = bbox.color[:3]
                alpha = bbox.fill_opacity
                if i == self.selected_idx:
                    r, g, b = 1.0, 1.0, 0.0
                    alpha = min(alpha, 0.5)
                corners = bbox.corners()
                glColor4f(r, g, b, alpha)
                for face in AABB_FACES:
                    glBegin(GL_QUADS)
                    for vi in face:
                        glVertex3dv(corners[vi])
                    glEnd()

            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        # --- Pass 2: Wireframe edges ---
        for i, bbox in enumerate(self.annotations):
            if not bbox.visible or getattr(bbox, 'scene_coords', False):
                continue

            selected = (i == self.selected_idx)
            glLineWidth(4.0 if selected else 3.0)

            if selected:
                glColor4f(1.0, 1.0, 0.0, 1.0)
            else:
                r, g, b = bbox.color[:3]
                glColor4f(r, g, b, 1.0)

            corners = bbox.corners()
            glBegin(GL_LINES)
            for a, b in AABB_EDGES:
                glVertex3dv(corners[a])
                glVertex3dv(corners[b])
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_gizmo(self, bbox):
        """Draw UE-style gizmo: move arrows, scale cubes, rotation ring."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        c = bbox.center_pos
        gl = self._gizmo_len(bbox)
        hov = self._hovered_gizmo
        ah = gl * 0.12  # arrow head size
        sh = gl * 0.06  # scale handle half-size

        # --- Move arrows: lines from center with cone tips ---
        for axis in range(3):
            is_hov = (hov is not None and hov[0] == 'move' and hov[1] == axis)
            r, g, b = AXIS_COLORS[axis]
            if is_hov:
                r, g, b = 1.0, 1.0, 0.3
            glLineWidth(3.5 if is_hov else 2.0)
            glColor4f(r, g, b, 1.0)
            tip = c.copy()
            tip[axis] += gl
            glBegin(GL_LINES)
            glVertex3dv(c)
            glVertex3dv(tip)
            glEnd()
            # Arrow cone
            perp1 = (axis + 1) % 3
            perp2 = (axis + 2) % 3
            base = c.copy()
            base[axis] += gl - ah
            glBegin(GL_TRIANGLES)
            for sign1, sign2 in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                b1 = base.copy()
                b1[perp1] += sign1 * ah * 0.4
                b1[perp2] += sign2 * ah * 0.4
                next_s = [(0, 1), (-1, 0), (0, -1), (1, 0)]
                idx = [(1, 0), (0, 1), (-1, 0), (0, -1)].index((sign1, sign2))
                s1n, s2n = next_s[idx]
                b2 = base.copy()
                b2[perp1] += s1n * ah * 0.4
                b2[perp2] += s2n * ah * 0.4
                glVertex3dv(tip)
                glVertex3dv(b1)
                glVertex3dv(b2)
            glEnd()

        # --- Scale handles: small cubes at face centers ---
        for axis in range(3):
            for sign in (+1, -1):
                is_hov = (hov is not None and hov[0] == 'scale'
                          and hov[1] == axis and hov[2] == sign)
                r, g, b = AXIS_COLORS[axis]
                if is_hov:
                    r, g, b = 1.0, 1.0, 0.3
                glColor4f(r, g, b, 0.9 if is_hov else 0.7)
                fc = c.copy()
                fc[axis] += sign * bbox.size[axis] / 2.0
                p1 = (axis + 1) % 3
                p2 = (axis + 2) % 3
                glBegin(GL_QUADS)
                for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    v = fc.copy()
                    v[p1] += s1 * sh
                    v[p2] += s2 * sh
                    glVertex3dv(v)
                glEnd()
                # Outline
                glLineWidth(2.0 if is_hov else 1.0)
                glColor4f(r, g, b, 1.0)
                glBegin(GL_LINE_LOOP)
                for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    v = fc.copy()
                    v[p1] += s1 * sh
                    v[p2] += s2 * sh
                    glVertex3dv(v)
                glEnd()

        # --- Rotation ring ---
        is_rot_hov = (hov is not None and hov[0] == 'rotate')
        radius = max(bbox.size[0], bbox.size[1]) * 0.6
        if radius > 0.05:
            if is_rot_hov:
                glLineWidth(3.0)
                glColor4f(1.0, 1.0, 0.3, 0.9)
            else:
                glLineWidth(1.5)
                glColor4f(0.3, 0.3, 1.0, 0.6)
            segments = 48
            glBegin(GL_LINE_LOOP)
            for seg in range(segments):
                angle = 2.0 * math.pi * seg / segments
                x = c[0] + radius * math.cos(angle)
                y = c[1] + radius * math.sin(angle)
                glVertex3d(x, y, c[2])
            glEnd()

        glEnable(GL_DEPTH_TEST)
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
            from OpenGL.GL import glDisable, glEnable, glLineWidth, glBegin, glEnd, GL_DEPTH_TEST
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)

        rp = self.ref_point
        sz = max(0.1, self.cam_distance * 0.01)

        for axis in range(3):
            r, g, b = AXIS_COLORS[axis]
            glColor4f(r, g, b, 0.9)
            glBegin(GL_LINES)
            glVertex3dv(rp)
            glVertex3dv(rp + np.array([sz if axis == 0 else 0,
                                                 sz if axis == 1 else 0,
                                                 sz if axis == 2 else 0]))
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_scene_bboxes(self):
        """Draw pipeline bboxes in scene (post-correction) coordinates."""
        visible = [b for b in self.scene_bboxes if b.visible]
        if not visible:
            return
        try:
            from OpenGL.GL import (glDisable, glEnable, glLineWidth, glBegin, glEnd,
                                   GL_DEPTH_TEST)
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        for bbox in visible:
            glLineWidth(3.0)
            r, g, b = bbox.color[:3]
            glColor4f(r, g, b, 1.0)
            corners = bbox.corners()
            glBegin(GL_LINES)
            for a, b_idx in AABB_EDGES:
                glVertex3dv(corners[a])
                glVertex3dv(corners[b_idx])
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_gap_annotations(self):
        """Draw bracket lines for gap annotations."""
        visible_gaps = [g for g in self.gaps if g.visible]
        if not visible_gaps:
            return

        try:
            from OpenGL.GL import (glDisable, glEnable, glLineWidth, glBegin, glEnd,
                                   GL_DEPTH_TEST)
        except ImportError:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0)

        arrow_sz = 0.02  # arrowhead size
        default_color = (1.0, 0.2, 0.2)

        for gap in visible_gaps:
            gc = gap.color or default_color
            glColor4f(gc[0], gc[1], gc[2], 1.0)
            a = gap.edge_a  # bracket endpoint A
            b = gap.edge_b  # bracket endpoint B

            # Tick lines: from anchor (bbox face) through edge, extended past bracket
            a_extended = a + gap.tick_dir
            b_extended = b + gap.tick_dir

            # Arrow direction vector (a→b)
            horiz = b - a
            length = np.linalg.norm(horiz)
            if length < 1e-6:
                continue
            d = horiz / length
            d_arrow = d * arrow_sz

            # Barb perpendicular to both the arrow and the tick direction
            barb = np.cross(d, gap.tick_dir)
            barb_len = np.linalg.norm(barb)
            if barb_len > 1e-6:
                barb = barb / barb_len * (arrow_sz * 0.5)
            else:
                # Fallback: barb in Z if arrow and tick are coplanar
                barb = np.array([0, 0, arrow_sz * 0.5])

            glBegin(GL_LINES)
            # Tick A: from anchor through edge to extended
            glVertex3dv(a_extended)
            glVertex3dv(gap.anchor_a)
            # Tick B
            glVertex3dv(b_extended)
            glVertex3dv(gap.anchor_b)
            # Arrow shaft
            glVertex3dv(a)
            glVertex3dv(b)
            # Arrowhead A
            glVertex3dv(a); glVertex3dv(a + d_arrow + barb)
            glVertex3dv(a); glVertex3dv(a + d_arrow - barb)
            # Arrowhead B
            glVertex3dv(b); glVertex3dv(b - d_arrow + barb)
            glVertex3dv(b); glVertex3dv(b - d_arrow - barb)
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # Store scene-space modelview for paintEvent label projection
        from OpenGL.GL import glGetFloatv, GL_MODELVIEW_MATRIX
        self._gl_modelview_scene = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float64)

    def paintEvent(self, event):
        """Paint GL content then overlay gap labels with QPainter."""
        super().paintEvent(event)

        visible_gaps = [g for g in self.gaps if g.visible]
        mv = getattr(self, '_gl_modelview_scene', None)
        proj = self._gl_projection
        vp = self._gl_viewport
        if not visible_gaps or mv is None or proj is None or vp is None:
            return

        from PySide6.QtGui import QPainter, QFont, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        painter.setFont(font)

        # White outline for readability over point cloud
        outline = QColor(255, 255, 255)
        default_color = (1.0, 0.2, 0.2)

        for gap in visible_gaps:
            gc = gap.color or default_color
            text_color = QColor(int(gc[0] * 255), int(gc[1] * 255), int(gc[2] * 255))

            sa_x, sa_y = project_to_screen(gap.edge_a, mv, proj, vp)
            sb_x, sb_y = project_to_screen(gap.edge_b, mv, proj, vp)
            mid = (gap.edge_a + gap.edge_b) / 2.0
            sx, sy = project_to_screen(mid, mv, proj, vp)

            text = f"{gap.gap_mm:.0f}mm"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text)

            bracket_px = max(abs(sb_x - sa_x), abs(sb_y - sa_y))
            if tw + 4 < bracket_px:
                tx, ty = int(sx - tw / 2), int(sy - 8)
            else:
                # Place text past the bracket (in tick_dir direction)
                offset_mid = mid + gap.tick_dir * 1.5
                px, py = project_to_screen(offset_mid, mv, proj, vp)
                tx, ty = int(px - tw / 2), int(py - 4)

            # Draw white outline then red text for contrast
            painter.setPen(outline)
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                painter.drawText(tx + dx, ty + dy, text)
            painter.setPen(text_color)
            painter.drawText(tx, ty, text)

        painter.end()

    def _gizmo_len(self, bbox):
        return max(0.3, float(np.max(bbox.size)) * 0.6)

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_to_screen_local(self, world_point):
        """Project 3D world point to 2D screen coordinates (Qt convention)."""
        if self._gl_modelview is None:
            return 0, 0
        return project_to_screen(world_point, self._gl_modelview,
                                 self._gl_projection, self._gl_viewport)

    def _hit_test_gizmo(self, screen_x, screen_y):
        """Test screen-space proximity to gizmo handles.

        All test points are batch-projected in a single NumPy call for
        performance (no per-point GL calls).

        Priority: scale handles > move arrows > rotation ring.
        Scale handles cannot be overridden by move arrows unless the arrow
        is significantly closer (>5px margin).

        Returns (operation, axis, sign) or None.
        """
        if self.selected_idx < 0 or self.selected_idx >= len(self.annotations):
            return None
        if self._gl_modelview is None:
            return None

        bbox = self.annotations[self.selected_idx]
        c = bbox.center_pos
        gizmo_len = self._gizmo_len(bbox)
        thr = GIZMO_HIT_PX

        # --- Build all test points and their metadata ---
        points = []    # 3D world positions
        meta = []      # (operation, axis, sign) for each point

        # Scale handles: 6 face centers
        for axis in range(3):
            for sign in (+1, -1):
                pt = c.copy()
                pt[axis] += sign * bbox.size[axis] / 2.0
                points.append(pt)
                meta.append(('scale', axis, sign))

        # Move arrows: 5 samples along each axis shaft
        for axis in range(3):
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                pt = c.copy()
                pt[axis] += gizmo_len * frac
                points.append(pt)
                meta.append(('move', axis, 0))

        # Rotation ring: 24 samples
        radius = max(bbox.size[0], bbox.size[1]) * 0.6
        if radius > 0.05:
            for seg in range(24):
                angle = 2.0 * math.pi * seg / 24
                points.append(np.array([
                    c[0] + radius * math.cos(angle),
                    c[1] + radius * math.sin(angle),
                    c[2],
                ]))
                meta.append(('rotate', 2, 0))

        if not points:
            return None

        # --- Batch project all points to screen ---
        pts_array = np.array(points, dtype=np.float64)
        screen_pts = project_points_to_screen(
            pts_array, self._gl_modelview, self._gl_projection, self._gl_viewport)

        # --- Compute distances ---
        dists = np.hypot(screen_pts[:, 0] - screen_x,
                         screen_pts[:, 1] - screen_y)

        # --- Find best per category ---
        n_scale = 6
        n_move = 15
        scale_dists = dists[:n_scale]
        move_dists = dists[n_scale:n_scale + n_move]
        rot_dists = dists[n_scale + n_move:]

        scale_dist, scale_hit = thr + 1, None
        if len(scale_dists):
            idx = int(np.argmin(scale_dists))
            if scale_dists[idx] <= thr:
                scale_dist = float(scale_dists[idx])
                scale_hit = meta[idx]

        move_dist, move_hit = thr + 1, None
        if len(move_dists):
            idx = int(np.argmin(move_dists))
            if move_dists[idx] <= thr:
                move_dist = float(move_dists[idx])
                move_hit = meta[n_scale + idx]

        rot_dist, rot_hit = thr + 1, None
        if len(rot_dists):
            idx = int(np.argmin(rot_dists))
            if rot_dists[idx] <= thr:
                rot_dist = float(rot_dists[idx])
                rot_hit = meta[n_scale + n_move + idx]

        # --- Resolve with priority ---
        best_dist, best_hit = thr + 1, None

        if scale_dist <= thr:
            best_dist, best_hit = scale_dist, scale_hit

        if move_dist <= thr:
            margin = 5.0 if best_hit is not None else 0.0
            if move_dist < best_dist - margin:
                best_dist, best_hit = move_dist, move_hit

        if rot_dist <= thr and rot_dist < best_dist:
            best_dist, best_hit = rot_dist, rot_hit

        return best_hit

    def _find_nearest_bbox(self, origin, direction):
        """Ray-cast against all bboxes, return nearest index or -1."""
        best_t, best_idx = float('inf'), -1
        for i, bbox in enumerate(self.annotations):
            if not bbox.visible:
                continue
            t = ray_aabb_intersect(origin, direction, bbox.bb_min, bbox.bb_max)
            if t is not None and t < best_t:
                best_t, best_idx = t, i
        return best_idx

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()

        if event.button() == Qt.MouseButton.LeftButton:
            mods = event.modifiers()

            # Shift+Click: skip bbox interaction, pass to camera for panning
            if mods & Qt.KeyboardModifier.ShiftModifier:
                super().mousePressEvent(event)
                return

            # Reference point picking mode
            if self._picking_ref_point:
                self.makeCurrent()
                pt = self._pick_3d(sx, sy)
                self.doneCurrent()
                if pt is not None:
                    self._picking_ref_point = False
                    self.point_picked.emit(float(pt[0]), float(pt[1]), float(pt[2]))
                return

            # Ctrl+Click: pick point → create new bbox (any tool mode)
            if mods & Qt.KeyboardModifier.ControlModifier:
                self.makeCurrent()
                pt = self._pick_3d(sx, sy)
                self.doneCurrent()
                if pt is not None:
                    self.point_picked.emit(float(pt[0]), float(pt[1]), float(pt[2]))
                return

            # --- Gizmo handle hit test ---
            if self.selected_idx >= 0 and self.selected_idx < len(self.annotations):
                self.makeCurrent()
                ghit = self._hit_test_gizmo(sx, sy)
                self.doneCurrent()
                if ghit is not None:
                    bbox = self.annotations[self.selected_idx]
                    op, axis, sign = ghit
                    # Emit undo snapshot before modification
                    self.transform_committed.emit(self.selected_idx, {
                        'center': bbox.center_pos.copy(),
                        'size': bbox.size.copy(),
                        'rotation_z': bbox.rotation_z,
                    })
                    self._drag_start = (sx, sy)
                    self._drag_orig_center = bbox.center_pos.copy()
                    self._drag_orig_size = bbox.size.copy()
                    self._drag_plane_z = float(bbox.center_pos[2])

                    if op == 'move':
                        self._drag_mode = 'gizmo_move'
                        self._drag_axis = axis
                    elif op == 'scale':
                        self._drag_mode = 'gizmo_scale'
                        self._drag_axis = axis
                        self._drag_sign = sign
                    elif op == 'rotate':
                        self._drag_mode = 'gizmo_rotate'
                        self._drag_orig_rot = bbox.rotation_z
                    return

            # --- Fallback: old tool-mode drags ---
            if self.tool == TOOL_MOVE and self.selected_idx >= 0:
                bbox = self.annotations[self.selected_idx]
                self._drag_mode = 'move'
                self._drag_start = (sx, sy)
                self._drag_orig_center = bbox.center_pos.copy()
                self._drag_orig_size = bbox.size.copy()
                self._drag_plane_z = float(bbox.center_pos[2])
                self.transform_committed.emit(self.selected_idx, {
                    'center': bbox.center_pos.copy(),
                    'size': bbox.size.copy(),
                    'rotation_z': bbox.rotation_z,
                })
                return

            if self.tool == TOOL_ROTATE and self.selected_idx >= 0:
                bbox = self.annotations[self.selected_idx]
                self._drag_mode = 'rotate'
                self._drag_start = (sx, sy)
                self._drag_orig_rot = bbox.rotation_z
                self._drag_orig_center = bbox.center_pos.copy()
                self._drag_orig_size = bbox.size.copy()
                self._drag_plane_z = float(bbox.center_pos[2])
                self.transform_committed.emit(self.selected_idx, {
                    'center': bbox.center_pos.copy(),
                    'size': bbox.size.copy(),
                    'rotation_z': bbox.rotation_z,
                })
                return

            # --- Click on bbox to select ---
            self.makeCurrent()
            origin, direction = ray_from_mouse(
                sx, sy, self._gl_modelview, self._gl_projection, self._gl_viewport)
            self.doneCurrent()
            hit_idx = self._find_nearest_bbox(origin, direction)

            if hit_idx >= 0:
                self.selected_idx = hit_idx
                self.bbox_selected.emit(hit_idx)
                self.update()
                self._last_mouse = event.position()
                self._mouse_btn = event.button()
                return

            # --- Click on background → deselect ---
            if self.selected_idx >= 0:
                self.selected_idx = -1
                self._hovered_gizmo = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.bbox_selected.emit(-1)
                self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()

        # --- Gizmo drags ---
        if self._drag_mode == 'gizmo_move' and self.selected_idx >= 0:
            axis = self._drag_axis
            c_screen = np.array(self._project_to_screen_local(self._drag_orig_center))
            axis_end = self._drag_orig_center.copy()
            axis_end[axis] += 1.0
            a_screen = np.array(self._project_to_screen_local(axis_end))
            axis_dir_screen = a_screen - c_screen
            axis_len_screen = np.linalg.norm(axis_dir_screen)
            if axis_len_screen > 0.5:
                axis_dir_screen /= axis_len_screen
                mouse_delta = np.array([sx - self._drag_start[0], sy - self._drag_start[1]])
                px_along = np.dot(mouse_delta, axis_dir_screen)
                world_delta = px_along / axis_len_screen
                new_center = self._drag_orig_center.copy()
                new_center[axis] += world_delta
                self.annotations[self.selected_idx].center_pos = new_center
                self.bbox_moved.emit(self.selected_idx)
                self.update()
            return

        if self._drag_mode == 'gizmo_scale' and self.selected_idx >= 0:
            axis = self._drag_axis
            c_screen = np.array(self._project_to_screen_local(self._drag_orig_center))
            axis_end = self._drag_orig_center.copy()
            axis_end[axis] += 1.0
            a_screen = np.array(self._project_to_screen_local(axis_end))
            axis_dir_screen = a_screen - c_screen
            axis_len_screen = np.linalg.norm(axis_dir_screen)
            if axis_len_screen > 0.5:
                axis_dir_screen /= axis_len_screen
                mouse_delta = np.array([sx - self._drag_start[0], sy - self._drag_start[1]])
                px_along = np.dot(mouse_delta, axis_dir_screen)
                world_delta = px_along / axis_len_screen
                new_size = self._drag_orig_size.copy()
                new_size[axis] = max(0.02, self._drag_orig_size[axis] + world_delta * self._drag_sign)

                if self.scale_from_corner:
                    # Corner mode: anchor the opposite face, shift center
                    size_change = new_size[axis] - self._drag_orig_size[axis]
                    new_center = self._drag_orig_center.copy()
                    new_center[axis] += size_change * self._drag_sign * 0.5
                    self.annotations[self.selected_idx].center_pos = new_center

                self.annotations[self.selected_idx].size = new_size
                self.bbox_moved.emit(self.selected_idx)
                self.update()
            return

        if self._drag_mode == 'gizmo_rotate' and self.selected_idx >= 0:
            dx = sx - self._drag_start[0]
            self.annotations[self.selected_idx].rotation_z = self._drag_orig_rot + dx * 0.5
            self.bbox_moved.emit(self.selected_idx)
            self.update()
            return

        # --- Legacy tool-mode drags ---
        if self._drag_mode == 'move' and self.selected_idx >= 0:
            if self.axis_constraint is not None:
                axis = self.axis_constraint
                c_screen = np.array(self._project_to_screen_local(self._drag_orig_center))
                axis_end = self._drag_orig_center.copy()
                axis_end[axis] += 1.0
                a_screen = np.array(self._project_to_screen_local(axis_end))
                axis_dir_screen = a_screen - c_screen
                axis_len_screen = np.linalg.norm(axis_dir_screen)
                if axis_len_screen > 0.5:
                    axis_dir_screen /= axis_len_screen
                    mouse_delta = np.array([sx - self._drag_start[0], sy - self._drag_start[1]])
                    px_along = np.dot(mouse_delta, axis_dir_screen)
                    world_delta = px_along / axis_len_screen
                    new_center = self._drag_orig_center.copy()
                    new_center[axis] += world_delta
                    self.annotations[self.selected_idx].center_pos = new_center
                    self.bbox_moved.emit(self.selected_idx)
                    self.update()
            else:
                from ..utils.math import project_point_to_camera_plane
                start_world = project_point_to_camera_plane(
                    self._drag_start[0], self._drag_start[1], self._drag_orig_center,
                    self._gl_modelview, self._gl_projection, self._gl_viewport)
                current_world = project_point_to_camera_plane(
                    sx, sy, self._drag_orig_center,
                    self._gl_modelview, self._gl_projection, self._gl_viewport)
                if start_world is not None and current_world is not None:
                    delta = current_world - start_world
                    self.annotations[self.selected_idx].center_pos = self._drag_orig_center + delta
                    self.bbox_moved.emit(self.selected_idx)
                    self.update()
            return

        if self._drag_mode == 'rotate' and self.selected_idx >= 0:
            dx = sx - self._drag_start[0]
            self.annotations[self.selected_idx].rotation_z = self._drag_orig_rot + dx * 0.5
            self.bbox_moved.emit(self.selected_idx)
            self.update()
            return

        # --- Hover detection (no drag active) ---
        if self._drag_mode is None and self.selected_idx >= 0:
            old_hov = self._hovered_gizmo
            self._hovered_gizmo = self._hit_test_gizmo(sx, sy)
            if self._hovered_gizmo != old_hov:
                if self._hovered_gizmo is not None:
                    op = self._hovered_gizmo[0]
                    if op == 'move':
                        self.setCursor(Qt.CursorShape.SizeAllCursor)
                    elif op == 'scale':
                        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    elif op == 'rotate':
                        self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_mode in ('move', 'rotate', 'gizmo_move', 'gizmo_scale', 'gizmo_rotate'):
            self._drag_mode = None
            self._drag_start = None
            if self.selected_idx >= 0:
                self.bbox_moved.emit(self.selected_idx)
            super().mouseReleaseEvent(event)
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _pick_3d(self, screen_x, screen_y):
        """Pick a 3D point at the given screen coordinates using depth buffer."""
        if self._gl_modelview is None:
            return None
        from ..utils.math import project_point_to_plane
        return project_point_to_plane(
            screen_x, screen_y, 0.0,
            self._gl_modelview, self._gl_projection, self._gl_viewport)
