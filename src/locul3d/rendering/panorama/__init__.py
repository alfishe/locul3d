"""Panorama subpackage — modular panorama extraction and rendering.

This package is fully self-contained.  To disable panorama support,
remove this directory and the ``PanoramaManager`` instantiation in
``rendering/gl/viewport.py``.

Modules
-------
extractor       E57 images2D extraction (libe57 + PIL)
station_marker  Configurable 3D diamond gizmo at camera positions
immersive       360° inside-out sphere renderer
geometry        UV sphere mesh generation (pure function)
"""

from __future__ import annotations

import math
from typing import Optional, Callable, List

import numpy as np

from .station_marker import draw_station_marker
from .immersive import ImmersiveRenderer
from .extractor import extract_panoramas, PANORAMA_LAYER_COLOR, PANORAMA_LAYER_OPACITY


class PanoramaManager:
    """Thin coordinator that composes the panorama submodules.

    The viewport creates one instance and delegates all panorama work
    through it — no panorama logic lives in the viewport itself.
    """

    def __init__(self):
        self._renderer = ImmersiveRenderer()
        # Immersive camera state
        self._pano_yaw: float = 0.0
        self._pano_pitch: float = 0.0
        self._pano_fov: float = 90.0
        self._active_layer = None
        self._selected_layer = None  # non-immersive selection (marker click)
        # Manual fine-tuning offsets (degrees)
        self._manual_yaw: float = 0.0
        self._manual_pitch: float = 0.0
        self._manual_roll: float = 0.0

    def handle_key_event(self, key, modifiers=None) -> bool:
        """Handle keyboard fine-tuning in panorama mode.
        
        Arrow/WASD = 0.1° steps, Shift+Arrow/WASD = 1.0° steps.
        Returns True if the event was handled.
        """
        from PySide6.QtCore import Qt
        step = 1.0 if (modifiers and modifiers & Qt.ShiftModifier) else 0.1
        changed = False
        
        # Arrows or WASD for yaw/pitch
        if key in (Qt.Key_Left, Qt.Key_A):
            self._manual_yaw -= step
            changed = True
        elif key in (Qt.Key_Right, Qt.Key_D):
            self._manual_yaw += step
            changed = True
        elif key in (Qt.Key_Up, Qt.Key_W):
            self._manual_pitch += step
            changed = True
        elif key in (Qt.Key_Down, Qt.Key_S):
            self._manual_pitch -= step
            changed = True
        
        # Q/E for roll
        elif key == Qt.Key_Q:
            self._manual_roll -= step
            changed = True
        elif key == Qt.Key_E:
            self._manual_roll += step
            changed = True
            
        # R to reset
        elif key == Qt.Key_R:
            self._manual_yaw = 0.0
            self._manual_pitch = 0.0
            self._manual_roll = 0.0
            changed = True
            
        if changed:
            print(f"Fine-tuning correction: yaw={self._manual_yaw:+.1f}°, "
                  f"pitch={self._manual_pitch:+.1f}°, roll={self._manual_roll:+.1f}°")
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when in immersive 360° panorama mode."""
        return self._renderer.is_active

    @property
    def active_layer(self):
        """The LayerData currently being viewed immersively, or None."""
        return self._active_layer

    @property
    def selected_layer(self):
        """The LayerData selected by clicking its marker (non-immersive)."""
        return self._selected_layer

    def select_layer(self, layer):
        """Set the selected panorama layer (or None to clear)."""
        self._selected_layer = layer

    # ------------------------------------------------------------------
    # Extraction (called during E57 import)
    # ------------------------------------------------------------------

    @staticmethod
    def extract(path: str,
                log_fn: Optional[Callable[[str], None]] = None) -> List[dict]:
        """Extract panorama stations from an E57 file.

        Returns a list of dicts suitable for creating ``LayerData``
        objects.  See :func:`extractor.extract_panoramas` for the dict
        schema.
        """
        return extract_panoramas(path, log_fn)

    # ------------------------------------------------------------------
    # Scene markers (called per visible panorama layer each frame)
    # ------------------------------------------------------------------

    def draw_marker(self, layer, scene_radius: float) -> None:
        """Draw a station marker for a panorama layer.

        Uses ``layer.color`` and ``layer.opacity`` so the marker
        appearance matches the layer panel.  When a panorama is
        active OR selected, non-active/non-selected markers render gray.
        """
        if layer.pano_position is None:
            return

        # Hide ALL markers in immersive mode (they obscure the view)
        if self._active_layer is not None:
            return

        # Determine if this marker should be highlighted or dimmed
        highlight_layer = self._active_layer or self._selected_layer
        is_highlighted = (highlight_layer is not None
                          and highlight_layer is layer)
        if highlight_layer is not None and not is_highlighted:
            color = (0.5, 0.5, 0.5)
            opacity = 0.25
        else:
            color = tuple(layer.color[:3]) if layer.color else None
            opacity = None  # use marker default (0.6)
        draw_station_marker(
            position=layer.pano_position,
            scene_radius=scene_radius,
            color=color,
            opacity=opacity,
        )

    def hit_test(self, layers, screen_x, screen_y,
                 viewport_w, viewport_h, threshold=20) -> Optional[object]:
        """Find the nearest panorama marker to screen coordinates.

        Projects each panorama station position to screen space using
        the current GL modelview/projection matrices, and returns the
        closest layer within *threshold* pixels.

        Parameters
        ----------
        layers : list of LayerData
            All visible panorama layers.
        screen_x, screen_y : float
            Click/hover position in widget coordinates (origin top-left).
        viewport_w, viewport_h : int
            Viewport dimensions in pixels.
        threshold : int
            Maximum distance in pixels to consider a hit.

        Returns
        -------
        LayerData or None
        """
        try:
            from OpenGL.GL import glGetDoublev, glGetIntegerv
            from OpenGL.GL import GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
            from OpenGL.GLU import gluProject
        except ImportError:
            return None

        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)

        # GL screen has origin at bottom-left; widget has origin top-left
        gl_y = viewport_h - screen_y

        best_layer = None
        best_dist = threshold + 1

        for layer in layers:
            if layer.layer_type != "panorama" or layer.pano_position is None:
                continue
            if not layer.visible:
                continue
            pos = layer.pano_position
            try:
                sx, sy, sz = gluProject(
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    mv, proj, vp)
            except Exception:
                continue

            # sz > 1 means behind camera
            if sz > 1.0 or sz < 0.0:
                continue

            dist = math.sqrt((sx - screen_x) ** 2 + (sy - gl_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_layer = layer

        return best_layer

    # ------------------------------------------------------------------
    # Immersive 360° mode
    # ------------------------------------------------------------------

    def enter(self, layer, camera_state: dict) -> None:
        """Enter immersive panorama mode.

        Parameters
        ----------
        layer : LayerData
            The panorama layer to view.
        camera_state : dict
            Current camera state to restore on exit.  Expected keys:
            distance, azimuth, elevation, target, fov.
        """
        equirect = layer.pano_equirect
        if equirect is None and layer.pano_faces:
            equirect = self._assemble_equirect(layer.pano_faces)
            layer.pano_equirect = equirect  # cache for next time

        if equirect is None:
            return

        self._saved_camera = camera_state
        self._active_layer = layer
        self._pano_yaw = 0.0
        self._pano_pitch = 0.0
        self._pano_fov = 90.0
        self._renderer.enter(equirect)

    def exit(self) -> dict:
        """Exit immersive mode, return saved camera state."""
        self._renderer.exit()
        self._active_layer = None
        saved = self._saved_camera
        self._saved_camera = {}
        return saved

    def _get_station_rotation(self):
        """Get station quaternion for GL rotation (spherical/cylindrical only).
        
        Cubemap panoramas handle orientation via face sorting,
        so no GL rotation should be applied.
        """
        if not self._active_layer:
            return None
        ptype = self._active_layer.pano_type
        if ptype in ("cubemap", "visual_ref"):
            return None
        return self._active_layer.pano_rotation

    def paint(self, aspect: float) -> None:
        """Render the immersive panorama for the current frame."""
        opacity = self._active_layer.opacity if self._active_layer else 1.0
        offsets = (self._manual_yaw, self._manual_pitch, self._manual_roll)
        rotation = self._get_station_rotation()
        self._renderer.paint(self._pano_yaw, self._pano_pitch,
                             self._pano_fov, aspect, opacity,
                             rotation=rotation,
                             manual_offsets=offsets)

    def paint_in_scene(self, scene_radius: float) -> None:
        """Draw the panorama sphere at the station's world position."""
        if not self._active_layer:
            return
        pos = self._active_layer.pano_position
        if pos is None:
            return
        opacity = self._active_layer.opacity if self._active_layer else 1.0
        radius = max(scene_radius * 0.3, 5.0)
        offsets = (self._manual_yaw, self._manual_pitch, self._manual_roll)
        rotation = self._get_station_rotation()
        self._renderer.paint_in_scene(pos, opacity, radius,
                                      rotation=rotation,
                                      manual_offsets=offsets)

    def handle_mouse_move(self, dx: float, dy: float) -> None:
        """Update yaw/pitch from mouse movement (in immersive mode)."""
        self._pano_yaw += dx * 0.3
        self._pano_pitch = max(-89, min(89, self._pano_pitch + dy * 0.3))

    def handle_scroll(self, delta: float) -> None:
        """Adjust FOV from scroll wheel (in immersive mode)."""
        self._pano_fov = max(20, min(120, self._pano_fov - delta * 0.1))

    # ------------------------------------------------------------------
    # Internal — cubemap assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_equirect(faces: list):
        """Convert cubemap face images to a single equirectangular image."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            return None

        # Find first non-None face for sizing
        first = next((f for f in faces if f is not None), None)
        if first is None:
            return None

        face_size = first.size[0]
        n_faces = len(faces)
        out_w = face_size * 4
        out_h = face_size * 2

        face_arrays = []
        for f in faces:
            if f is None:
                face_arrays.append(np.zeros((face_size, face_size, 3), dtype=np.float32))
            elif f.mode != "RGB":
                face_arrays.append(np.array(f.convert("RGB"), dtype=np.float32))
            else:
                face_arrays.append(np.array(f, dtype=np.float32))

        # Build equirect pixel coordinates
        u = np.linspace(0, 1, out_w, endpoint=False) + 0.5 / out_w
        v = np.linspace(0, 1, out_h, endpoint=False) + 0.5 / out_h
        uu, vv = np.meshgrid(u, v)

        # Spherical coordinates
        theta = uu * 2 * np.pi - np.pi   # azimuth
        phi = np.pi / 2 - vv * np.pi     # elevation

        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)

        ax = np.abs(x)
        ay = np.abs(y)
        az = np.abs(z)

        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        if n_faces >= 6:
            face_idx = np.zeros((out_h, out_w), dtype=int)
            face_idx[(ax >= ay) & (ax >= az) & (x > 0)] = 0   # +X
            face_idx[(ax >= ay) & (ax >= az) & (x <= 0)] = 1  # -X
            face_idx[(ay > ax) & (ay >= az) & (y > 0)] = 2    # +Y
            face_idx[(ay > ax) & (ay >= az) & (y <= 0)] = 3   # -Y
            face_idx[(az > ax) & (az > ay) & (z > 0)] = 4     # +Z
            face_idx[(az > ax) & (az > ay) & (z <= 0)] = 5    # -Z

            for fi in range(6):
                fm = face_idx == fi
                if not np.any(fm):
                    continue
                fa = face_arrays[fi] if fi < len(face_arrays) else face_arrays[0]
                fh, fw = fa.shape[:2]

                if fi == 0:
                    uc = -y[fm] / ax[fm] * 0.5 + 0.5
                    vc = -z[fm] / ax[fm] * 0.5 + 0.5
                elif fi == 1:
                    uc = y[fm] / ax[fm] * 0.5 + 0.5
                    vc = -z[fm] / ax[fm] * 0.5 + 0.5
                elif fi == 2:
                    uc = x[fm] / ay[fm] * 0.5 + 0.5
                    vc = -z[fm] / ay[fm] * 0.5 + 0.5
                elif fi == 3:
                    uc = -x[fm] / ay[fm] * 0.5 + 0.5
                    vc = -z[fm] / ay[fm] * 0.5 + 0.5
                elif fi == 4:
                    uc = y[fm] / az[fm] * 0.5 + 0.5
                    vc = x[fm] / az[fm] * 0.5 + 0.5
                else:
                    uc = y[fm] / az[fm] * 0.5 + 0.5
                    vc = -x[fm] / az[fm] * 0.5 + 0.5

                px = np.clip((uc * fw).astype(int), 0, fw - 1)
                py = np.clip((vc * fh).astype(int), 0, fh - 1)
                out[fm] = fa[py, px].astype(np.uint8)
        else:
            fa = face_arrays[0].astype(np.uint8)
            out_img = PILImage.fromarray(fa).resize((out_w, out_h))
            out = np.array(out_img)

        return PILImage.fromarray(out)
