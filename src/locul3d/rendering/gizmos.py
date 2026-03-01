"""3D gizmo system for object manipulation."""

import math
import numpy as np
from typing import Optional, Tuple, List

try:
    from OpenGL.GL import *
    from OpenGL.GL import GL_LIGHTING
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from locul3d.core.constants import AXIS_COLORS, GIZMO_HIT_PX
from locul3d.utils.math import project_to_screen


class GizmoSystem:
    """Manages 3D gizmo rendering and interaction."""

    def __init__(self):
        self.hovered_gizmo = None  # ('move', axis, sign) | ('scale', axis, sign) | ('rotate', axis, sign) | None

    def draw_move_gizmo(self, center, size, hovered=None):
        """Draw move gizmo arrows."""
        if not HAS_OPENGL:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0 if hovered else 1.5)

        for axis in range(3):
            is_hov = (hovered is not None and hovered[0] == 'move' and hovered[1] == axis)
            r, g, b = AXIS_COLORS[axis]
            if is_hov:
                r, g, b = 1.0, 1.0, 0.3  # highlight yellow
            glColor4f(r, g, b, 1.0)
            
            tip = center.copy()
            tip[axis] += size
            glBegin(GL_LINES)
            glVertex3dv(center)
            glVertex3dv(tip)
            glEnd()
            
            # Arrow head (simple cone using triangle fan)
            ah = size * 0.2
            perp1 = (axis + 1) % 3
            perp2 = (axis + 2) % 3
            base = center.copy()
            base[axis] += size - ah
            
            glBegin(GL_TRIANGLES)
            for sign1, sign2 in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                b1 = base.copy()
                b1[perp1] += sign1 * ah * 0.4
                b1[perp2] += sign2 * ah * 0.4
                glVertex3dv(tip)
                glVertex3dv(b1)
                
                next_s = [(0, 1), (-1, 0), (0, -1), (1, 0)]
                idx = [(1, 0), (0, 1), (-1, 0), (0, -1)].index((sign1, sign2))
                s1n, s2n = next_s[idx]
                b2 = base.copy()
                b2[perp1] += s1n * ah * 0.4
                b2[perp2] += s2n * ah * 0.4
                glVertex3dv(b2)
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_scale_gizmo(self, center, bbox_size, hovered=None):
        """Draw scale gizmo handles (cubes at face centers)."""
        if not HAS_OPENGL:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        sh = max(0.05, np.max(bbox_size) * 0.06)  # Scale handle size

        for axis in range(3):
            for sign in (+1, -1):
                is_hov = (hovered is not None and hovered[0] == 'scale'
                          and hovered[1] == axis and hovered[2] == sign)
                r, g, b = AXIS_COLORS[axis]
                if is_hov:
                    r, g, b = 1.0, 1.0, 0.3
                glColor4f(r, g, b, 0.9 if is_hov else 0.7)
                
                fc = center.copy()
                fc[axis] += sign * bbox_size[axis] / 2.0
                
                # Draw small square (filled quad)
                p1 = (axis + 1) % 3
                p2 = (axis + 2) % 3
                glBegin(GL_QUADS)
                for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    v = fc.copy()
                    v[p1] += s1 * sh
                    v[p2] += s2 * sh
                    glVertex3dv(v)
                glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_rotate_gizmo(self, center, bbox_size, hovered=None):
        """Draw rotate gizmo ring."""
        if not HAS_OPENGL:
            return

        radius = max(bbox_size[0], bbox_size[1]) * 0.6
        if radius < 0.05:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        is_hov = (hovered is not None and hovered[0] == 'rotate')
        if is_hov:
            glLineWidth(3.0)
            glColor4f(1.0, 1.0, 0.3, 0.9)
        else:
            glLineWidth(1.5)
            glColor4f(0.3, 0.3, 1.0, 0.6)

        segments = 48
        glBegin(GL_LINE_LOOP)
        for seg in range(segments):
            angle = 2.0 * math.pi * seg / segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            glVertex3d(x, y, center[2])
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def hit_test(self, screen_x, screen_y, center, bbox_size,
                  modelview, projection, viewport):
        """Test screen-space proximity to gizmo handles."""
        thr = GIZMO_HIT_PX
        best_dist, best_hit = thr + 1, None

        # Scale handles — 6 face centers
        for axis in range(3):
            for sign in (+1, -1):
                handle = center.copy()
                handle[axis] += sign * bbox_size[axis] / 2.0
                sx, sy = project_to_screen(handle, modelview, projection, viewport)
                d = math.hypot(screen_x - sx, screen_y - sy)
                if d < best_dist:
                    best_dist, best_hit = d, ('scale', axis, sign)

        # Move arrow tips
        gizmo_len = max(0.3, float(np.max(bbox_size)) * 0.6)
        for axis in range(3):
            tip = center.copy()
            tip[axis] += gizmo_len
            sx, sy = project_to_screen(tip, modelview, projection, viewport)
            d = math.hypot(screen_x - sx, screen_y - sy)
            if d < best_dist:
                best_dist, best_hit = d, ('move', axis, 0)

        # Rotation ring — sample 24 points
        radius = max(bbox_size[0], bbox_size[1]) * 0.6
        if radius > 0.05:
            for seg in range(24):
                angle = 2.0 * math.pi * seg / 24
                ring_pt = np.array([
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                    center[2],
                ])
                sx, sy = project_to_screen(ring_pt, modelview, projection, viewport)
                d = math.hypot(screen_x - sx, screen_y - sy)
                if d < best_dist:
                    best_dist, best_hit = d, ('rotate', 2, 0)

        if best_dist <= thr:
            return best_hit
        return None
