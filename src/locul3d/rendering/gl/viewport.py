"""Base OpenGL viewport widget."""

import math
import time
import numpy as np
from typing import Optional, Dict, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QToolTip
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat

# Disable PyOpenGL error checking BEFORE importing GL functions.
# By default PyOpenGL calls glGetError() after every GL function,
# forcing a GPU→CPU pipeline flush (~30-100µs each on Apple Silicon's
# Metal translation layer).  With many draw calls this dominates
# frame time.
import OpenGL

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False

from OpenGL import contextdata

contextdata.getContext = lambda *args: 1

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GL import (
        shaders,
        GL_ARRAY_BUFFER,
        GL_ELEMENT_ARRAY_BUFFER,
        GL_VERTEX_ARRAY,
        GL_COLOR_ARRAY,
        GL_NORMAL_ARRAY,
        GL_STATIC_DRAW,
        GL_LIGHTING,
    )

    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from locul3d.core.layer import LayerManager
from locul3d.core.constants import COLORS
from locul3d.core.correction import SceneCorrection

try:
    from locul3d.rendering.panorama import PanoramaManager

    HAS_PANORAMA = True
except ImportError:
    HAS_PANORAMA = False


class BaseGLViewport(QOpenGLWidget):
    """Base OpenGL viewport widget for 3D rendering."""

    marker_selected = Signal(object)  # single-click: highlight + scroll only
    marker_activated = Signal(object)  # double-click: open info panel
    fps_updated = Signal(float)

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        fmt.setDepthBufferSize(24)
        fmt.setVersion(2, 1)
        self.setFormat(fmt)

        self.setMinimumSize(600, 400)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)  # enable hover events

        self.layer_manager = layer_manager

        # Camera
        self.cam_distance = 50.0
        self.cam_azimuth = 45.0
        self.cam_elevation = 30.0
        self.cam_target = np.array([0.0, 0.0, 0.0])
        self.cam_fov = 45.0

        # Mouse state
        self._last_mouse = None
        self._mouse_btn = None
        self._click_pos = None  # position at mousePress (for drag detection)

        # Render settings
        self.point_size = 2.0
        self.show_axes = True
        self.show_grid = True
        self.use_layer_colors = (
            False  # False = per-vertex RGB by default; folder mode sets True
        )
        self.fps_movement = (
            False  # True = WASD/QE moves camera; False = scene correction
        )
        self.fps_camera = False  # True = first-person camera (cam_distance=0)
        self.point_attenuation = False  # True = perspective-correct 1/d point size falloff
        self.auto_scale_small_points = True  # True = uniform adaptive point sizing
        self._saved_cam_distance = None  # orbital distance saved when entering FPS
        self._fps_movement_was_manual = (
            False  # track if user had fps_movement on before fps_camera
        )
        self.bg_color = COLORS["gl_bg"]

        # Scene bounds (set after loading)
        self._scene_center = np.zeros(3)
        self._scene_radius = 10.0
        self._grid_size = 10.0

        # VBO management: maps (layer_id, buffer_kind) -> GL buffer id
        # buffer_kind is one of: 'pts', 'rgba', 'normals', 'tris', 'lines'
        self._vbos: Dict[Tuple[str, str], int] = {}

        # Interactive decimation state
        self._interacting = False  # mouse-drag orbit/pan or scroll
        self._interact_timer = QTimer(self)
        self._interact_timer.setSingleShot(True)
        self._interact_timer.timeout.connect(self._stop_interaction)
        self._adjusting_opacity = False  # opacity slider being dragged

        # Scene clipping: None = no clip, or (x0, x1, y0, y1, z0, z1)
        self.scene_clip = None

        # Scene correction (rotation + shift for axis alignment)
        self.scene_correction = SceneCorrection()
        self._correction_diag = None  # CorrectionDiagnostics for debug overlays

        # Panorama support (fully optional — see rendering/panorama/)
        self._panorama = PanoramaManager() if HAS_PANORAMA else None

        # FPS tracking
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    def _start_interaction(self):
        """Enable interactive decimation for high FPS (30+)."""
        self._interacting = True
        self._interact_timer.start(300)  # Keep low-res for 300ms after last event
        self.update()

    def _stop_interaction(self):
        """Redraw in high-res once movement stops."""
        self._interacting = False
        self.update()

    # --- Camera Control ---

    def fit_to_scene(self):
        """Reset camera to fit entire scene and adjust grid."""
        center, radius = self.layer_manager.get_scene_bounds()
        self._scene_center = center
        self._scene_radius = radius
        self.cam_target = center.copy()
        self.cam_distance = radius * 2.5
        if self.fps_camera:
            self._saved_cam_distance = self.cam_distance
            self.cam_distance = 0.0
        self.cam_azimuth = 45.0
        self.cam_elevation = 30.0

        # Adjust grid size to fit scene
        scene_diameter = radius * 2
        desired_grid = scene_diameter * 1.2

        # Round to nice grid values
        if desired_grid < 5:
            self._grid_size = 5.0
        elif desired_grid < 10:
            self._grid_size = 10.0
        elif desired_grid < 20:
            self._grid_size = 20.0
        elif desired_grid < 50:
            self._grid_size = 50.0
        elif desired_grid < 100:
            self._grid_size = 100.0
        elif desired_grid < 200:
            self._grid_size = 200.0
        else:
            self._grid_size = np.ceil(desired_grid / 100) * 100

        self.update()

    def reset_camera(self):
        self.cam_target = self._scene_center.copy()
        self.cam_distance = self._scene_radius * 2.5
        if self.fps_camera:
            self._saved_cam_distance = self.cam_distance
            self.cam_distance = 0.0
        self.cam_azimuth = 45.0
        self.cam_elevation = 30.0
        self.update()

    def reset(self):
        """Reset all viewport state to initial values.

        Subclasses should call ``super().reset()`` and then clear their
        own state so a single ``reset()`` call propagates through the
        entire hierarchy.
        """
        self.delete_all_vbos()
        self.scene_correction = SceneCorrection()
        self.scene_clip = None
        self._correction_diag = None
        if self._panorama and self._panorama.is_active:
            self.exit_panorama()

    def set_view(self, azimuth: float, elevation: float):
        self.cam_azimuth = azimuth
        self.cam_elevation = elevation
        self.update()

    def set_fps_camera(self, enabled: bool):
        """Toggle between orbital and first-person camera mode."""
        if enabled and not self.fps_camera:
            self._saved_cam_distance = self.cam_distance
            self._fps_movement_was_manual = self.fps_movement
            self.cam_distance = 0.0
            self.fps_movement = True
            self.fps_camera = True
        elif not enabled and self.fps_camera:
            self.cam_distance = self._saved_cam_distance or self._scene_radius * 2.5
            self._saved_cam_distance = None
            if not self._fps_movement_was_manual:
                self.fps_movement = False
            self.fps_camera = False
        self.update()

    # --- GL Lifecycle ---

    def initializeGL(self):
        if not HAS_OPENGL:
            return
        try:
            self.makeCurrent()
            glClearColor(*self.bg_color)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

            # Lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
            self._gl_ok = True
        except Exception:
            self._gl_ok = False

    def resizeGL(self, w, h):
        if not HAS_OPENGL:
            return
        glViewport(0, 0, w, h)

    def paintGL(self):
        if not HAS_OPENGL:
            return
        if not getattr(self, "_gl_ok", True):
            return

        try:
            self._paintGL_inner()
        except Exception:
            import traceback

            traceback.print_exc()
            try:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            except Exception:
                pass

    # --- VBO helpers ---

    def _get_or_create_vbo(
        self, layer_id: str, kind: str, data: np.ndarray, target: int = GL_ARRAY_BUFFER
    ) -> int:
        """Return an existing VBO or upload *data* to a new one.

        On Apple-Silicon UMA the OpenGL→Metal translation layer maps a
        ``glBufferData`` upload to a ``MTLBuffer`` with shared storage,
        so the GPU reads directly from unified RAM on every subsequent
        draw — no per-frame copy.
        """
        key = (layer_id, kind)
        vbo = self._vbos.get(key)
        if vbo is not None:
            return vbo

        vbo = int(glGenBuffers(1))
        glBindBuffer(target, vbo)
        glBufferData(target, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(target, 0)
        self._vbos[key] = vbo

        # Mark the source layer as GPU-resident so it can release
        # redundant CPU-side float32 caches (the VBO holds the data).
        for layer in self.layer_manager.layers:
            if layer.id == layer_id:
                layer.gpu_resident = True
                # Release the float32 color expansion — uint8 compact
                # stays for potential re-upload after hide/show cycle.
                layer.release_source_data_after_vbo()
                break

        return vbo

    # --- Rendering ---

    def _paintGL_inner(self):
        """Inner paint implementation - override in subclasses."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Panorama mode — render point cloud AND panorama sphere from
        # the station's exact viewpoint using ONE shared camera setup.
        if self._panorama and self._panorama.is_active:
            pano_layer = self._panorama.active_layer

            # Render scene from station viewpoint (camera + point cloud)
            if pano_layer and pano_layer.pano_position is not None:
                self._render_scene_from_station(pano_layer)

            # Render panorama sphere on top, in the SAME GL matrix
            if pano_layer and pano_layer.visible and pano_layer.opacity > 0.01:
                self._panorama.paint_in_scene(self._scene_radius)

            self._frame_count += 1
            return

        self._render_normal_scene()

        self._frame_count += 1

    def _render_normal_scene(self):
        """Render the normal 3D scene (projection, camera, grid, layers)."""
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        far = max(self._scene_radius * 4.0, self.cam_distance * 10, 100.0)
        near = max(0.01, far * 1e-5)
        gluPerspective(self.cam_fov, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera
        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)
        cam_x = self.cam_target[0] + self.cam_distance * math.cos(el) * math.cos(az)
        cam_y = self.cam_target[1] + self.cam_distance * math.cos(el) * math.sin(az)
        cam_z = self.cam_target[2] + self.cam_distance * math.sin(el)

        # When cam_distance is 0 (panorama mode), eye == target.
        # Compute a look-at point from the view direction instead.
        if self.cam_distance < 0.001:
            look_x = cam_x - math.cos(el) * math.cos(az)
            look_y = cam_y - math.cos(el) * math.sin(az)
            look_z = cam_z - math.sin(el)
        else:
            look_x = self.cam_target[0]
            look_y = self.cam_target[1]
            look_z = self.cam_target[2]

        gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 0, 1)

        # Light follows camera
        glLightfv(GL_LIGHT0, GL_POSITION, [cam_x, cam_y, cam_z, 1.0])

        # Global ground reference plane at Z=0 (before correction)
        if getattr(self, "show_ground_plane", False):
            self._draw_ground_plane()

        # Grid and axes in absolute world space (before correction)
        # so they serve as a fixed reference for the scene alignment
        if self.show_grid:
            self._draw_grid()
        if self.show_axes:
            self._draw_axes()

        # Hook for subclass overlays drawn in global (pre-correction) space
        self._draw_global_overlays()

        # Apply scene correction (rotation + shift for axis alignment)
        # Order: rotate first (fix tilt), then shift in world space (e.g. floor to Z=0)
        sc = self.scene_correction
        if not sc.is_identity:
            if sc.rotate_x != 0:
                glRotatef(sc.rotate_x, 1, 0, 0)
            if sc.rotate_y != 0:
                glRotatef(sc.rotate_y, 0, 1, 0)
            if sc.rotate_z != 0:
                glRotatef(sc.rotate_z, 0, 0, 1)
            if sc.shift_x != 0 or sc.shift_y != 0 or sc.shift_z != 0:
                glTranslatef(sc.shift_x, sc.shift_y, sc.shift_z)

        # Enable scene clip planes (AABB clipping for all layers)
        clip = self.scene_clip
        if clip is not None:
            x0, x1, y0, y1, z0, z1 = clip
            # 6 clip planes: +X, -X, +Y, -Y, +Z, -Z
            glClipPlane(GL_CLIP_PLANE0, [1, 0, 0, -x0])  # x >= x0
            glClipPlane(GL_CLIP_PLANE1, [-1, 0, 0, x1])  # x <= x1
            glClipPlane(GL_CLIP_PLANE2, [0, 1, 0, -y0])  # y >= y0
            glClipPlane(GL_CLIP_PLANE3, [0, -1, 0, y1])  # y <= y1
            glClipPlane(GL_CLIP_PLANE4, [0, 0, 1, -z0])  # z >= z0
            glClipPlane(GL_CLIP_PLANE5, [0, 0, -1, z1])  # z <= z1
            for i in range(6):
                glEnable(GL_CLIP_PLANE0 + i)

        # Render layers: opaque first, then transparent (simple depth sort)
        visible = self.layer_manager.visible_layers()
        opaque = [l for l in visible if l.opacity >= 0.99]
        transparent = [l for l in visible if l.opacity < 0.99]

        # Evict VBOs for layers that became invisible since last frame.
        # This frees GPU RAM for large hidden layers (e.g. 348M-pt raw scan).
        visible_ids = {l.id for l in visible}
        for layer in self.layer_manager.layers:
            if layer.id not in visible_ids and layer.gpu_resident:
                self.delete_vbos_for_layer(layer.id)
                layer.gpu_resident = False

        # Global interactive stride: cap total drawn points during
        # mouse-drag so aggregate draw stays within budget.
        INTERACTIVE_BUDGET = 25_000_000
        total_vis = sum(
            l.point_count for l in visible if l.layer_type == "pointcloud"
        )
        self._total_vis_pts = total_vis  # cached for uniform point sizing
        if self._interacting:
            self._global_interact_stride = (
                max(1, total_vis // INTERACTIVE_BUDGET)
                if total_vis > INTERACTIVE_BUDGET
                else 1
            )
        else:
            self._global_interact_stride = 1

        for layer in opaque + transparent:
            if layer.layer_type == "pointcloud":
                self._draw_point_layer(layer)
            elif layer.layer_type == "mesh":
                self._draw_mesh_layer(layer)
            elif layer.layer_type == "wireframe":
                self._draw_wireframe_layer(layer)
            elif layer.layer_type == "panorama" and self._panorama:
                self._panorama.draw_marker(layer, self._scene_radius)

        # Disable clip planes
        if clip is not None:
            for i in range(6):
                glDisable(GL_CLIP_PLANE0 + i)

        # Debug overlays for auto-detect diagnostics
        if self._correction_diag is not None:
            self._draw_correction_diagnostics()

    def _render_scene_from_station(self, pano_layer):
        """Render the scene from a panorama station's viewpoint.

        Used in immersive see-through mode so the point cloud aligns
        with the panorama texture visible through the semi-transparent sphere.
        """
        pos = pano_layer.pano_position
        if pos is None:
            return

        # Use the same FOV / look direction as the panorama view
        pano_yaw = self._panorama._pano_yaw
        pano_pitch = self._panorama._pano_pitch
        pano_fov = self._panorama._pano_fov

        # Projection — match panorama FOV
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        far = max(self._scene_radius * 4.0, 200.0)
        near = max(0.01, far * 1e-5)
        gluPerspective(pano_fov, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera at station position, looking in panorama direction
        yaw_r = math.radians(pano_yaw)
        pitch_r = math.radians(pano_pitch)
        dx = math.cos(pitch_r) * math.cos(yaw_r)
        dy = math.cos(pitch_r) * math.sin(yaw_r)
        dz = math.sin(pitch_r)
        gluLookAt(
            pos[0], pos[1], pos[2], pos[0] + dx, pos[1] + dy, pos[2] + dz, 0, 0, 1
        )

        # Light at station
        glLightfv(GL_LIGHT0, GL_POSITION, [pos[0], pos[1], pos[2], 1.0])

        # NOTE: SceneCorrection is intentionally NOT applied here.
        # Both point cloud and panorama positions are in the same
        # aligned frame from E57 import — applying correction would
        # shift the scene relative to the camera (at pano_position).

        # Render visible layers (no grid/axes/wireframes in immersive)
        visible = self.layer_manager.visible_layers()
        for layer in visible:
            if layer.layer_type == "pointcloud":
                self._draw_point_layer(layer)
            elif layer.layer_type == "mesh":
                self._draw_mesh_layer(layer)

    # --- Layer rendering (VBO path: upload once, draw many) ---

    def _draw_point_layer(self, layer):
        """Render a point cloud layer using VBOs (upload once, draw many)."""

        pts = layer.get_pts_array()
        if pts is None:
            return

        glDisable(GL_LIGHTING)

        base_size = self.point_size

        if self.point_attenuation:
            ref_d = max(self.cam_distance, 0.5)
            c_att = 1.0 / (ref_d * ref_d)
            glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, [0.0, 0.0, c_att])
            glPointParameterf(GL_POINT_SIZE_MIN, 0.5)
            glPointParameterf(GL_POINT_SIZE_MAX, 128.0)

        glPointSize(base_size)

        render_on_top = False
        if render_on_top:
            glDisable(GL_DEPTH_TEST)

        # Determine coloring mode
        use_uniform = self.use_layer_colors and layer.color is not None
        has_vtx_colors = False

        if use_uniform:
            r, g, b = layer.color[:3]
            glColor4f(r, g, b, layer.opacity)
        elif layer.colors is not None or getattr(layer, 'colors_u8', None) is not None:
            has_vtx_colors = True
        else:
            glColor4f(0.7, 0.7, 0.8, layer.opacity)

        # --- Stride-based LOD ---
        # On Apple-Silicon, PyOpenGL → Metal bridge limits throughput.
        # Use full density when stationary (MAX_STATIC_PTS=1G), but
        # use aggressive stride during interaction for high FPS.
        MAX_STATIC_PTS = 1_000_000_000   # Stationary: 100% density (full 348M)
        MAX_INTERACTIVE_PTS = 25_000_000  # Drag: High detail preview
        MAX_OPACITY_PREVIEW_PTS = 500_000 # Slider: Instant response

        if self._adjusting_opacity and layer.point_count > MAX_OPACITY_PREVIEW_PTS:
            stride = max(1, layer.point_count // MAX_OPACITY_PREVIEW_PTS)
        elif self._interacting:
            stride = max(1, layer.point_count // MAX_INTERACTIVE_PTS)
            # Also respect the global budget stride
            stride = max(stride, getattr(self, "_global_interact_stride", 1))
        elif layer.point_count > MAX_STATIC_PTS:
            stride = max(1, layer.point_count // MAX_STATIC_PTS)
        else:
            stride = 1

        pts_stride = stride * 3 * 4  # bytes between consecutive vertices
        rgb_stride = stride * 3 * 4  # same layout for 3-component colors
        draw_count = layer.point_count // stride

        # --- VBO path ---
        vbo_pts = self._get_or_create_vbo(layer.id, "pts", pts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, pts_stride, None)

        if has_vtx_colors:
            # Prefer compact uint8 colors (3× less VRAM than float32).
            # GL_UNSIGNED_BYTE with glColorPointer normalizes 0-255 → 0.0-1.0.
            colors_u8 = getattr(layer, 'colors_u8', None)
            if colors_u8 is not None:
                if not colors_u8.flags['C_CONTIGUOUS']:
                    layer.colors_u8 = np.ascontiguousarray(colors_u8)
                    colors_u8 = layer.colors_u8
                    import gc
                    gc.collect()
                rgb_stride_u8 = stride * 3 * 1  # uint8: 3 bytes per vertex
                vbo_rgb = self._get_or_create_vbo(layer.id, "rgb", colors_u8)
                glBindBuffer(GL_ARRAY_BUFFER, vbo_rgb)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_UNSIGNED_BYTE, rgb_stride_u8, None)
            else:
                colors = layer.get_colors_array()
                if colors is not None:
                    vbo_rgb = self._get_or_create_vbo(layer.id, "rgb", colors)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_rgb)
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointer(3, GL_FLOAT, rgb_stride, None)

        # Apply layer opacity as a GPU-side blend — no data regeneration
        # needed.  RGB vertex colors have implicit alpha=1.0; we modulate
        # the final fragment alpha via GL_CONSTANT_ALPHA so that changing
        # the opacity slider is instant even for 100M-point layers.
        #
        # For opaque layers: DISABLE blending to prevent color accumulation
        # when overlapping layers share coordinates (GL_POINT_SMOOTH edges
        # would bleed through otherwise).
        needs_blend = layer.opacity < 0.99
        if needs_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
            glBlendColor(0.0, 0.0, 0.0, layer.opacity)
        else:
            glDisable(GL_BLEND)

        glDrawArrays(GL_POINTS, 0, draw_count)

        if needs_blend:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if self.point_attenuation:
            # Reset to flat (non-attenuated) point sizes for other rendering
            glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, [1.0, 0.0, 0.0])

        if render_on_top:
            glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_mesh_layer(self, layer):
        """Render a triangle mesh layer with lighting (VBO path)."""
        pts = layer.get_pts_array()
        tris = layer.get_tris_array()
        if pts is None or tris is None:
            return

        glEnable(GL_LIGHTING)

        # Determine coloring mode
        use_uniform = self.use_layer_colors and layer.color is not None
        has_vtx_colors = False

        if use_uniform:
            r, g, b = layer.color[:3]
            glColor4f(r, g, b, layer.opacity)
        elif layer.colors is not None or getattr(layer, 'colors_u8', None) is not None:
            has_vtx_colors = True
        else:
            glColor4f(0.6, 0.65, 0.7, layer.opacity)

        # --- VBO path ---
        vbo_pts = self._get_or_create_vbo(layer.id, "pts", pts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        # Normals
        normals = layer.get_normals_array()
        if normals is not None:
            vbo_n = self._get_or_create_vbo(layer.id, "normals", normals)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_n)
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, None)

        # Colors (RGB, opacity applied via blend state)
        if has_vtx_colors:
            colors = layer.get_colors_array()
            if colors is not None:
                vbo_rgb = self._get_or_create_vbo(layer.id, "rgb", colors)
                glBindBuffer(GL_ARRAY_BUFFER, vbo_rgb)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_FLOAT, 0, None)

        # Index buffer
        vbo_idx = self._get_or_create_vbo(
            layer.id, "tris", tris, target=GL_ELEMENT_ARRAY_BUFFER
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_idx)

        needs_blend = layer.opacity < 0.99
        if needs_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
            glBlendColor(0.0, 0.0, 0.0, layer.opacity)
        else:
            glDisable(GL_BLEND)

        glDrawElements(GL_TRIANGLES, layer.tri_count * 3, GL_UNSIGNED_INT, None)

        if needs_blend:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def _draw_wireframe_layer(self, layer):
        """Render a wireframe layer (OBB edges) using immediate mode.

        Wireframe overlays have very few vertices (typically <200),
        so immediate mode avoids VBO state issues at no performance cost.
        """
        if layer.line_points is None or len(layer.line_points) == 0:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(3.0)

        # Wireframes always use the layer's swatch color (uniform) so all
        # edges in a single PLY overlay share one colour.
        if layer.color is not None:
            r, g, b = layer.color[:3]
            glColor4f(r, g, b, layer.opacity)
        else:
            glColor4f(1.0, 0.5, 0.0, layer.opacity)

        needs_blend = layer.opacity < 0.99
        if needs_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
            glBlendColor(0.0, 0.0, 0.0, layer.opacity)
        else:
            glDisable(GL_BLEND)

        pts = layer.line_points
        glBegin(GL_LINES)
        for i in range(len(pts)):
            glVertex3fv(pts[i])
        glEnd()

        if needs_blend:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glEnable(GL_LIGHTING)

    def _draw_axes(self):
        """Draw coordinate axes."""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        length = self._scene_radius * 0.15 if self._scene_radius > 0 else 1.0
        
        glBegin(GL_LINES)
        # Main axes
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        glColor3f(0.2, 0.9, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        
        # Labels
        d = length * 0.08
        s = length * 0.04

        # X label
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(length + d, -s, -s)
        glVertex3f(length + d, s, s)
        glVertex3f(length + d, -s, s)
        glVertex3f(length + d, s, -s)

        # Y label
        glColor3f(0.2, 0.9, 0.2)
        glVertex3f(-s, length + d, s)
        glVertex3f(0, length + d, 0)
        glVertex3f(s, length + d, s)
        glVertex3f(0, length + d, 0)
        glVertex3f(0, length + d, 0)
        glVertex3f(0, length + d, -s)

        # Z label
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(-s, 0, length + d + s)
        glVertex3f(s, 0, length + d + s)
        glVertex3f(s, 0, length + d + s)
        glVertex3f(-s, 0, length + d - s)
        glVertex3f(-s, 0, length + d - s)
        glVertex3f(s, 0, length + d - s)
        
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_grid(self):
        """Draw XY grid at Z=0, centered on scene and dynamically sized."""
        glDisable(GL_LIGHTING)
        glLineWidth(0.5)
        glColor4f(0.25, 0.25, 0.3, 0.4)

        # Grid centered on scene center (XY plane at Z=0)
        center_x = self._scene_center[0]
        center_y = self._scene_center[1]

        # Grid size and spacing
        half_size = self._grid_size / 2
        spacing = self._grid_size / 10

        glBegin(GL_LINES)
        # Draw 11 lines in each direction (10 divisions)
        for i in range(11):
            offset = -half_size + i * spacing
            # Lines parallel to X-axis
            y = center_y + offset
            glVertex3f(center_x - half_size, y, 0)
            glVertex3f(center_x + half_size, y, 0)
            # Lines parallel to Y-axis
            x = center_x + offset
            glVertex3f(x, center_y - half_size, 0)
            glVertex3f(x, center_y + half_size, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_global_overlays(self):
        """Hook for overlays drawn in global (pre-correction) space.

        Override in subclasses to draw planes, markers, etc. at
        absolute world coordinates (e.g. Z=0 reference planes).
        """
        pass

    def _draw_ground_plane(self):
        """Draw a wireframe reference plane at absolute Z=0.

        Drawn in world space BEFORE scene correction, so it stays fixed
        regardless of rotation/shift — a true reference for floor alignment.
        """
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor4f(0.0, 0.9, 0.9, 0.6)  # cyan

        half = max(self._scene_radius * 1.5, 20.0)
        divs = 20
        step = half * 2 / divs

        glBegin(GL_LINES)
        for i in range(divs + 1):
            t = -half + i * step
            # Centered on absolute origin (0, 0, 0)
            glVertex3f(t, -half, 0.0)
            glVertex3f(t, half, 0.0)
            glVertex3f(-half, t, 0.0)
            glVertex3f(half, t, 0.0)
        glEnd()
        glEnable(GL_LIGHTING)

    def set_correction_diagnostics(self, diag):
        """Set or clear diagnostic overlay data from auto-detection."""
        self._correction_diag = diag
        self.update()

    def _draw_correction_diagnostics(self):
        """Draw debug overlays showing voxelized wall surface detection.

        Color-codes cells by their angle relative to the dominant direction:
          - Green: cells aligned with dominant wall direction
          - Blue: cells perpendicular to dominant direction
          - Gray: outlier cells (other angles)
          - Normal arrows: surface orientation (top cells only)
          - Cyan crosshair: target axis alignment
          - Magenta line: current dominant wall direction
        """
        import math as _m

        diag = self._correction_diag
        if diag is None:
            return

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ── Wall-band points (very dim background) ────────────
        if diag.wall_band_points is not None and len(diag.wall_band_points) > 0:
            glPointSize(1.5)
            glColor4f(0.9, 0.8, 0.0, 0.08)
            pts = diag.wall_band_points
            if len(pts) > 30000:
                idx = np.random.default_rng(1).choice(len(pts), 30000, replace=False)
                pts = pts[idx]
            glBegin(GL_POINTS)
            for p in pts:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

        if not diag.wall_planes:
            glEnable(GL_LIGHTING)
            return

        # Determine the peak angle for color-coding
        peak = getattr(diag, "peak_angle_deg", 0.0)

        # Color-code each cell by its angle relative to the peak
        # 0-10° from peak → green (parallel wall)
        # 80-90° from peak → blue (perpendicular wall)
        # else → dim gray (outlier)
        scene_cx = float(np.mean([p.centroid[0] for p in diag.wall_planes]))
        scene_cy = float(np.mean([p.centroid[1] for p in diag.wall_planes]))

        for plane in diag.wall_planes:
            cx, cy, cz = (
                float(plane.centroid[0]),
                float(plane.centroid[1]),
                float(plane.centroid[2]),
            )
            nx, ny, nz = (
                float(plane.normal[0]),
                float(plane.normal[1]),
                float(plane.normal[2]),
            )
            # Compute extent from bounding box
            if hasattr(plane, "bbox_max") and plane.bbox_max is not None:
                span = plane.bbox_max - plane.bbox_min
                ext = max(float(np.max(span)) * 0.5, 0.3)
            else:
                ext = max(getattr(plane, "extent", 1.0) * 0.5, 0.3)

            # Angle difference from peak (mod 90°)
            diff = abs(plane.angle_deg - peak)
            if diff > 45:
                diff = 90 - diff

            # Color by qualifying status
            is_qualifying = getattr(plane, "qualifying", False)
            is_large = getattr(plane, "area", 0) >= 5.0

            if is_qualifying:
                r, g, b, a = 0.0, 1.0, 0.3, 0.35  # bright green — qualifying
            elif is_large:
                r, g, b, a = 0.8, 0.4, 0.0, 0.2  # dim orange — large but not qualifying
            else:
                continue  # skip small non-qualifying surfaces

            # Plane tangent vectors
            up = (
                np.array([0.0, 0.0, 1.0])
                if abs(nz) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
            t1 = np.cross(plane.normal, up)
            t1 /= np.linalg.norm(t1) + 1e-10
            t2 = np.cross(plane.normal, t1)
            t2 /= np.linalg.norm(t2) + 1e-10

            # Translucent cell quad
            glColor4f(r, g, b, a)
            corners = [
                plane.centroid + ext * (-t1 - t2),
                plane.centroid + ext * (t1 - t2),
                plane.centroid + ext * (t1 + t2),
                plane.centroid + ext * (-t1 + t2),
            ]
            glBegin(GL_QUADS)
            for c in corners:
                glVertex3f(float(c[0]), float(c[1]), float(c[2]))
            glEnd()

            # Wireframe outline
            glColor4f(r, g, b, 0.6)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            for c in corners:
                glVertex3f(float(c[0]), float(c[1]), float(c[2]))
            glEnd()

        # ── Normal arrows for top qualifying surfaces ───────────
        top = sorted(
            [p for p in diag.wall_planes if getattr(p, "area", 0) >= 5.0],
            key=lambda p: p.point_count,
            reverse=True,
        )[:20]
        for plane in top:
            cx, cy, cz = (
                float(plane.centroid[0]),
                float(plane.centroid[1]),
                float(plane.centroid[2]),
            )
            nx, ny, nz = (
                float(plane.normal[0]),
                float(plane.normal[1]),
                float(plane.normal[2]),
            )
            if getattr(plane, "qualifying", False):
                glColor4f(0.0, 1.0, 0.3, 1.0)
            else:
                glColor4f(0.8, 0.5, 0.0, 0.7)
            if hasattr(plane, "bbox_max") and plane.bbox_max is not None:
                span = plane.bbox_max - plane.bbox_min
                arrow_len = max(float(np.max(span)) * 0.4, 0.5)
            else:
                arrow_len = max(getattr(plane, "extent", 1.0) * 0.8, 0.5)
            glLineWidth(2.5)
            glBegin(GL_LINES)
            glVertex3f(cx, cy, cz)
            glVertex3f(cx + nx * arrow_len, cy + ny * arrow_len, cz + nz * arrow_len)
            glEnd()

        # ── Fiducial cross markers on Z=0 ─────────────────────
        # NOTE: The GL modelview already includes the scene correction
        # rotation (rotate_z).  To draw these markers in TRUE world
        # coordinates we must counter-rotate by -rotate_z first.
        glPushMatrix()
        sc = self.scene_correction
        if sc.rotate_z != 0:
            glRotatef(-sc.rotate_z, 0, 0, 1)

        grid_extent = max(self._scene_radius * 0.6, 8.0)
        grid_spacing = 5.0  # metres between markers
        cross_arm = 1.0  # half-length of each cross arm

        # Blue markers — final position after alignment correction (axis-aligned = 0°)
        self._draw_fiducial_grid(
            cx=scene_cx,
            cy=scene_cy,
            cz=0.0,
            angle_deg=0.0,
            extent=grid_extent,
            spacing=grid_spacing,
            arm_len=cross_arm,
            color=(0.3, 0.5, 1.0, 0.7),
            line_width=2.0,
        )

        # Magenta cross = original scene position detected by wall planes
        if abs(diag.wall_correction_deg) > 0.001:
            indicator_len = max(self._scene_radius * 0.4, 5.0)
            self._draw_fiducial_cross(
                cx=scene_cx,
                cy=scene_cy,
                cz=0.0,
                angle_deg=-diag.wall_correction_deg,
                arm_len=indicator_len,
                color=(1.0, 0.0, 1.0, 0.8),
                line_width=4.0,
            )

        glPopMatrix()

        glLineWidth(1.0)
        glPointSize(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_fiducial_cross(
        self,
        cx: float,
        cy: float,
        cz: float,
        angle_deg: float,
        arm_len: float,
        color: tuple[float, float, float, float],
        line_width: float,
    ):
        """Draw a single rotated cross at (cx, cy, cz)."""
        import math as _m

        rad = _m.radians(angle_deg)
        cos_a = _m.cos(rad)
        sin_a = _m.sin(rad)

        glColor4f(*color)
        glLineWidth(line_width)
        glBegin(GL_LINES)
        # Arm 1
        glVertex3f(cx - arm_len * cos_a, cy - arm_len * sin_a, cz)
        glVertex3f(cx + arm_len * cos_a, cy + arm_len * sin_a, cz)
        # Arm 2 (perpendicular)
        glVertex3f(cx + arm_len * sin_a, cy - arm_len * cos_a, cz)
        glVertex3f(cx - arm_len * sin_a, cy + arm_len * cos_a, cz)
        glEnd()

    def _draw_fiducial_grid(
        self,
        cx: float,
        cy: float,
        cz: float,
        angle_deg: float,
        extent: float,
        spacing: float,
        arm_len: float,
        color: tuple[float, float, float, float],
        line_width: float,
    ):
        """Draw a grid of small rotated crosses within 'extent' distance of (cx, cy)."""
        import math as _m

        rad = _m.radians(angle_deg)
        cos_a = _m.cos(rad)
        sin_a = _m.sin(rad)

        n_lines = int(extent / spacing) + 1
        glLineWidth(line_width)
        glColor4f(*color)
        for ix in range(-n_lines, n_lines + 1):
            for iy in range(-n_lines, n_lines + 1):
                px = cx + ix * spacing
                py = cy + iy * spacing
                # Arm 1
                glBegin(GL_LINES)
                glVertex3f(px - arm_len * cos_a, py - arm_len * sin_a, cz)
                glVertex3f(px + arm_len * cos_a, py + arm_len * sin_a, cz)
                glEnd()
                # Arm 2 (perpendicular)
                glBegin(GL_LINES)
                glVertex3f(px + arm_len * sin_a, py - arm_len * cos_a, cz)
                glVertex3f(px - arm_len * sin_a, py + arm_len * cos_a, cz)
                glEnd()

    # --- VBO Management ---

    def delete_vbos_for_layer(self, layer_id: str):
        """Free all GPU buffers associated with *layer_id*."""
        to_delete = [k for k in self._vbos if k[0] == layer_id]
        if not to_delete:
            return
        ids = [self._vbos.pop(k) for k in to_delete]
        try:
            self.makeCurrent()
            glDeleteBuffers(len(ids), ids)
        except Exception:
            pass
        # Mark layer as no longer GPU-resident
        for layer in self.layer_manager.layers:
            if layer.id == layer_id:
                layer.gpu_resident = False
                break

    def delete_all_vbos(self):
        """Free every VBO (e.g. when loading a new file)."""
        if not self._vbos:
            return
        ids = list(self._vbos.values())
        try:
            self.makeCurrent()
            glDeleteBuffers(len(ids), ids)
        except Exception:
            pass
        self._vbos.clear()

    # --- Mouse Events ---

    _CLICK_THRESHOLD = 5  # pixels — drag past this is not a click

    def mousePressEvent(self, event):
        self.setFocus()
        self._last_mouse = event.position()
        self._click_pos = event.position()  # remember for drag detection
        self._mouse_btn = event.button()
        self._interacting = True
        self.update()  # dropdown to interactive budget immediately

    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera control and marker hover."""
        # Hover tooltip (no button pressed)
        if self._last_mouse is None and self._panorama:
            if not self._panorama.is_active:
                self._handle_marker_hover(event)
            return

        if self._last_mouse is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()

        # Check for Shift+Left button (pan/strafe)
        if (
            self._mouse_btn == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            # Shift+Left drag = pan (same as middle mouse)
            scale = max(self.cam_distance, self._scene_radius * 0.5) * 0.002
            az = math.radians(self.cam_azimuth)
            el = math.radians(self.cam_elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.0])
            up = np.array(
                [
                    -math.cos(az) * math.sin(el),
                    -math.sin(az) * math.sin(el),
                    math.cos(el),
                ]
            )
            self.cam_target -= right * dx * scale
            self.cam_target += up * dy * scale
        elif self._mouse_btn == Qt.MouseButton.LeftButton:
            # Panorama mode: yaw/pitch
            if self._panorama and self._panorama.is_active:
                self._panorama.handle_mouse_move(-dx, dy)
            else:
                # Left drag = orbit camera
                self.cam_azimuth -= dx * 0.3
                self.cam_elevation = max(-89, min(89, self.cam_elevation + dy * 0.3))
        elif self._mouse_btn == Qt.MouseButton.MiddleButton:
            # Middle drag = pan camera
            scale = max(self.cam_distance, self._scene_radius * 0.5) * 0.002
            az = math.radians(self.cam_azimuth)
            el = math.radians(self.cam_elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.0])
            up = np.array(
                [
                    -math.cos(az) * math.sin(el),
                    -math.sin(az) * math.sin(el),
                    math.cos(el),
                ]
            )
            self.cam_target -= right * dx * scale
            self.cam_target += up * dy * scale
        elif self._mouse_btn == Qt.MouseButton.RightButton:
            # Right drag = dolly (zoom by moving forward, not just changing distance)
            az = math.radians(self.cam_azimuth)
            el = math.radians(self.cam_elevation)
            forward = np.array(
                [
                    -math.cos(el) * math.cos(az),
                    -math.cos(el) * math.sin(az),
                    -math.sin(el),
                ]
            )
            step = self._scene_radius * 0.005 * dy
            self.cam_target += forward * step
            # Tighten orbit radius to stay close to target
            self.cam_distance = min(self.cam_distance, self._scene_radius * 0.3)
            self.cam_distance = max(0.01, self.cam_distance)

        self._last_mouse = pos
        self._start_interaction()

    def mouseDoubleClickEvent(self, event):
        """Double-click on panorama marker → select + open info panel."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._panorama
            and not self._panorama.is_active
        ):
            pos = event.position()
            self.makeCurrent()
            layers = self.layer_manager.layers
            hit = self._panorama.hit_test(
                layers, pos.x(), pos.y(), self.width(), self.height()
            )
            self.doneCurrent()
            if hit:
                self._panorama.select_layer(hit)
                self.marker_selected.emit(hit)
                self.marker_activated.emit(hit)
                self.update()
                return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        # Check for click (not drag) on a panorama marker
        if (
            self._click_pos is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._panorama
            and not self._panorama.is_active
        ):
            pos = event.position()
            drag = math.sqrt(
                (pos.x() - self._click_pos.x()) ** 2
                + (pos.y() - self._click_pos.y()) ** 2
            )
            if drag < self._CLICK_THRESHOLD:
                self._handle_marker_click(pos)

        self._last_mouse = None
        self._click_pos = None
        self._mouse_btn = None
        self._start_interaction()  # redraw happens after 300ms delay

    def wheelEvent(self, event):
        angle = event.angleDelta()
        delta = angle.y() if abs(angle.y()) >= abs(angle.x()) else angle.x()

        # Panorama mode: wheel adjusts FOV (zoom in/out)
        # Both cam_fov and _pano_fov must stay in sync so the point cloud
        # projection matches the panorama sphere exactly.
        if self._panorama and self._panorama.is_active:
            self._start_interaction()
            step = -2.0 if delta > 0 else 2.0  # scroll up = zoom in
            new_fov = max(20.0, min(120.0, self._panorama._pano_fov + step))
            self._panorama._pano_fov = new_fov
            self.cam_fov = new_fov  # keep scene projection in sync
            return

        # Normal mode: dolly through scene
        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)
        forward = np.array(
            [
                -math.cos(el) * math.cos(az),
                -math.cos(el) * math.sin(az),
                -math.sin(el),
            ]
        )
        step = self._scene_radius * 0.03 * (1 if delta > 0 else -1)
        self.cam_target += forward * step
        self.cam_distance = min(self.cam_distance, self._scene_radius * 0.3)
        self.cam_distance = max(0.01, self.cam_distance)
        self._start_interaction()

    # --- Scene correction keyboard step sizes ---
    _SHIFT_STEP = 0.05  # metres per key press
    _ROTATE_STEP = 0.1  # degrees per key press
    _FPS_MOVE_STEP = 0.01  # multiplied by scene_radius for scale-independent speed

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.

        Non-panoramic mode:
          WASD  — shift scene along X / Y
          Q / E — shift scene along Z
          ← → ↑ ↓ — rotate scene around Z / X
          Hold Shift for 10× step, Ctrl for 100×.
        """
        if self._panorama and self._panorama.is_active:
            if self._panorama.handle_key_event(event.key(), event.modifiers()):
                self.update()
                return
            super().keyPressEvent(event)
            return

        key = event.key()
        mods = event.modifiers()

        # Modifier multiplier: Shift = 10×, Ctrl = 100×
        mult = 1.0
        if mods & Qt.KeyboardModifier.ShiftModifier:
            mult = 10.0
        elif mods & Qt.KeyboardModifier.ControlModifier:
            mult = 100.0

        if self.fps_movement:
            fps_handled = True
            step = self._scene_radius * self._FPS_MOVE_STEP * mult
            az = math.radians(self.cam_azimuth)
            forward = np.array([-math.cos(az), -math.sin(az), 0.0])
            right = np.array([-math.sin(az), math.cos(az), 0.0])

            if key == Qt.Key.Key_W:
                self.cam_target += forward * step
            elif key == Qt.Key.Key_S:
                self.cam_target -= forward * step
            elif key == Qt.Key.Key_A:
                self.cam_target -= right * step
            elif key == Qt.Key.Key_D:
                self.cam_target += right * step
            elif key == Qt.Key.Key_Q:
                self.cam_target[2] -= step
            elif key == Qt.Key.Key_E:
                self.cam_target[2] += step
            else:
                fps_handled = False

            if fps_handled:
                self.update()
                return

        sc = self.scene_correction
        handled = True

        if key == Qt.Key.Key_W:
            sc.shift_y += self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_S:
            sc.shift_y -= self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_A:
            sc.shift_x -= self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_D:
            sc.shift_x += self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_Q:
            sc.shift_z -= self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_E:
            sc.shift_z += self._SHIFT_STEP * mult
        elif key == Qt.Key.Key_Left:
            sc.rotate_z += self._ROTATE_STEP * mult
        elif key == Qt.Key.Key_Right:
            sc.rotate_z -= self._ROTATE_STEP * mult
        elif key == Qt.Key.Key_Up:
            sc.rotate_x += self._ROTATE_STEP * mult
        elif key == Qt.Key.Key_Down:
            sc.rotate_x -= self._ROTATE_STEP * mult
        else:
            handled = False

        if handled:
            # Round to avoid float drift
            sc.shift_x = round(sc.shift_x, 4)
            sc.shift_y = round(sc.shift_y, 4)
            sc.shift_z = round(sc.shift_z, 4)
            sc.rotate_x = round(sc.rotate_x, 4)
            sc.rotate_y = round(sc.rotate_y, 4)
            sc.rotate_z = round(sc.rotate_z, 4)
            print(
                f"correction: rot=({sc.rotate_x}, {sc.rotate_y}, {sc.rotate_z})°  "
                f"shift=({sc.shift_x}, {sc.shift_y}, {sc.shift_z})"
            )
            self.update()
            return

        super().keyPressEvent(event)

    # --- FPS ---

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_fps_time
        if dt > 0:
            fps = self._frame_count / dt
        else:
            fps = 0.0
        self._frame_count = 0
        self._last_fps_time = now
        self.fps_updated.emit(fps)

    def grab_screenshot(self, path: str):
        """Save current viewport as image."""
        img = self.grabFramebuffer()
        img.save(path)

    # --- Marker interaction ---

    def _handle_marker_click(self, pos):
        """Hit-test panorama markers and select the nearest one."""
        if not self._panorama:
            return
        self.makeCurrent()
        layers = self.layer_manager.layers
        hit = self._panorama.hit_test(
            layers, pos.x(), pos.y(), self.width(), self.height()
        )
        self.doneCurrent()
        self._panorama.select_layer(hit)
        self.marker_selected.emit(hit)
        self.update()

    def _handle_marker_hover(self, event):
        """Show tooltip with panorama ID on marker hover."""
        if not self._panorama:
            return
        self.makeCurrent()
        pos = event.position()
        layers = self.layer_manager.layers
        hit = self._panorama.hit_test(
            layers, pos.x(), pos.y(), self.width(), self.height()
        )
        self.doneCurrent()
        if hit:
            QToolTip.showText(event.globalPosition().toPoint(), hit.name)
        else:
            QToolTip.hideText()

    # --- Panorama enter/exit ---

    def enter_panorama(self, layer) -> None:
        """Enter immersive 360° panorama view for the given layer."""
        if not self._panorama or layer.layer_type != "panorama":
            return
        self.setFocus()
        camera_state = {
            "distance": self.cam_distance,
            "azimuth": self.cam_azimuth,
            "elevation": self.cam_elevation,
            "target": self.cam_target.copy(),
            "fov": self.cam_fov,
        }
        self._panorama.enter(layer, camera_state)
        self.cam_fov = self._panorama._pano_fov  # sync scene FOV to pano FOV
        self.update()

    def exit_panorama(self) -> None:
        """Exit immersive panorama mode and restore normal camera."""
        if not self._panorama or not self._panorama.is_active:
            return
        saved = self._panorama.exit()
        if saved:
            self.cam_distance = saved["distance"]
            self.cam_azimuth = saved["azimuth"]
            self.cam_elevation = saved["elevation"]
            self.cam_target = saved["target"]
            self.cam_fov = saved["fov"]
        self.update()
