"""Immersive 360° panorama renderer — inside-out textured sphere."""

import math
import numpy as np

try:
    from OpenGL.GL import (
        glGenTextures, glDeleteTextures, glBindTexture, glTexImage2D,
        glTexParameteri, glEnable, glDisable, glColor4f,
        glMatrixMode, glLoadIdentity, glEnableClientState,
        glDisableClientState, glVertexPointer, glTexCoordPointer,
        glDrawElements, glBlendFunc,
        glPushMatrix, glPopMatrix, glTranslatef, glScalef, glMultMatrixf,
        glRotated,
        GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_FLOAT, GL_UNSIGNED_INT,
        GL_LINEAR, GL_REPEAT, GL_CLAMP_TO_EDGE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_VERTEX_ARRAY, GL_TEXTURE_COORD_ARRAY,
        GL_TRIANGLES, GL_PROJECTION, GL_MODELVIEW,
        GL_LIGHTING, GL_DEPTH_TEST, GL_BLEND,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    )
    from OpenGL.GLU import gluPerspective, gluLookAt
    HAS_GL = True
except ImportError:
    HAS_GL = False

import numpy as np
from .geometry import build_sphere

# Default sphere radius (meters) — sized to enclose nearby geometry
DEFAULT_SPHERE_RADIUS = 15.0


def _quat_to_gl_matrix(quat):
    """Convert (w, x, y, z) quaternion to a 4×4 column-major GL matrix.

    Uses the conjugate (inverse) because the equirectangular image is
    horizontally mirrored for inside-out sphere UV, reversing handedness.
    """
    w, x, y, z = quat
    x, y, z = -x, -y, -z
    return np.array([
        1 - 2*(y*y + z*z),  2*(x*y + w*z),      2*(x*z - w*y),      0,
        2*(x*y - w*z),      1 - 2*(x*x + z*z),   2*(y*z + w*x),      0,
        2*(x*z + w*y),      2*(y*z - w*x),        1 - 2*(x*x + y*y),  0,
        0,                  0,                    0,                   1,
    ], dtype=np.float32)

class ImmersiveRenderer:
    """Renders an equirectangular panorama on an inside-out sphere.

    Usage::

        renderer = ImmersiveRenderer()
        renderer.enter(pil_image)           # upload texture
        renderer.paint(yaw, pitch, fov, aspect)  # each frame
        renderer.exit()                     # cleanup
    """

    def __init__(self):
        self._tex_id: int = 0
        self._sphere_verts = None
        self._sphere_uvs = None
        self._sphere_tris = None
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enter(self, equirect_img) -> None:
        """Upload equirectangular image and enter immersive mode.

        Parameters
        ----------
        equirect_img : PIL.Image
            RGB equirectangular panorama image.
        """
        if not HAS_GL or equirect_img is None:
            return

        self._upload_texture(equirect_img)
        if self._sphere_verts is None:
            self._sphere_verts, self._sphere_uvs, self._sphere_tris = (
                build_sphere(64, 128)
            )
        self._active = True

    def exit(self) -> None:
        """Release texture and leave immersive mode."""
        self._active = False
        if self._tex_id:
            try:
                glDeleteTextures([self._tex_id])
            except Exception:
                pass
            self._tex_id = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_gl_matrix(quat):
        """Convert E57 quaternion (w,x,y,z) to a column-major 4x4 GL matrix.
        
        The quaternion rotates from scanner-local to world coordinates.
        We apply the inverse (conjugate) to rotate the sphere from
        world orientation back to align with the scanner's captured image.
        """
        w, x, y, z = quat
        # Conjugate (inverse rotation): negate x,y,z
        x, y, z = -x, -y, -z
        
        # Column-major 4x4 rotation matrix from quaternion
        return np.array([
            1-2*(y*y+z*z), 2*(x*y+w*z),   2*(x*z-w*y),   0,
            2*(x*y-w*z),   1-2*(x*x+z*z), 2*(y*z+w*x),   0,
            2*(x*z+w*y),   2*(y*z-w*x),   1-2*(x*x+y*y), 0,
            0,             0,             0,             1,
        ], dtype=np.float32)

    def paint(self, yaw: float, pitch: float, fov: float,
              aspect: float, opacity: float = 1.0,
              rotation=None,
              manual_offsets: tuple = (0, 0, 0)) -> None:
        """Render the panorama sphere for the current view."""
        if not self._active or self._tex_id == 0:
            return

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, aspect, 0.01, 10.0)

        # View — look from origin
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        dx = math.cos(pitch_r) * math.cos(yaw_r)
        dy = math.cos(pitch_r) * math.sin(yaw_r)
        dz = math.sin(pitch_r)
        gluLookAt(0, 0, 0, dx, dy, dz, 0, 0, 1)

        # Apply station quaternion rotation (aligns image with world)
        if rotation is not None:
            mat = self._quat_to_gl_matrix(rotation)
            glMultMatrixf(mat)

        # Apply manual alignment fine-tuning on top
        myaw, mpitch, mroll = manual_offsets
        if abs(myaw) > 0.001:
            glRotated(myaw, 0, 0, 1)
        if abs(mpitch) > 0.001:
            glRotated(mpitch, 0, 1, 0)
        if abs(mroll) > 0.001:
            glRotated(mroll, 1, 0, 0)

        self._draw_sphere(opacity)

    def paint_in_scene(self, position, opacity: float = 1.0,
                       radius: float = DEFAULT_SPHERE_RADIUS,
                       rotation=None,
                       manual_offsets: tuple = (0, 0, 0)) -> None:
        """Draw the panorama sphere at a world position."""
        if not self._active or self._tex_id == 0:
            return

        glPushMatrix()
        glTranslatef(float(position[0]), float(position[1]),
                     float(position[2]))
        
        # Apply station quaternion rotation
        if rotation is not None:
            mat = self._quat_to_gl_matrix(rotation)
            glMultMatrixf(mat)

        # Apply manual alignment fine-tuning
        myaw, mpitch, mroll = manual_offsets
        if abs(myaw) > 0.001:
            glRotated(myaw, 0, 0, 1)
        if abs(mpitch) > 0.001:
            glRotated(mpitch, 0, 1, 0)
        if abs(mroll) > 0.001:
            glRotated(mroll, 1, 0, 0)

        glScalef(radius, radius, radius)
        self._draw_sphere(opacity)
        glPopMatrix()

    def _draw_sphere(self, opacity: float) -> None:
        """Core sphere draw call — assumes projection/modelview set."""
        # State
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)

        # Alpha blending for opacity < 1
        if opacity < 0.99:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1, 1, 1, opacity)

        # Draw sphere
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._sphere_verts)
        glTexCoordPointer(2, GL_FLOAT, 0, self._sphere_uvs)
        glDrawElements(GL_TRIANGLES, len(self._sphere_tris) * 3,
                       GL_UNSIGNED_INT, self._sphere_tris)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

        # Restore
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upload_texture(self, img) -> None:
        """Upload a PIL image as GL_TEXTURE_2D."""
        if self._tex_id:
            try:
                glDeleteTextures([self._tex_id])
            except Exception:
                pass

        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        data = img.tobytes()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._tex_id = tex_id
