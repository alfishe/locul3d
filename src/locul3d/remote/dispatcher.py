"""Command dispatcher — single point of contact between network and Qt.

Every viewer/editor mutation funnelled through ``invoke_on_qt()`` to
guarantee thread safety.  The dispatcher also owns the WebSocket client
registry and event broadcasting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import numpy as np

from .bridge import QtBridge
from .schemas import (
    BBoxCreate,
    BBoxUpdate,
    CameraPreset,
    CameraState,
    CameraUpdate,
    ClipState,
    CorrectionState,
    DynamicLayerCreate,
    DynamicLayerInfo,
    DynamicLayerPatch,
    GeometryType,
    LayerInfo,
    LayerUpdate,
    LookAtRequest,
    PlaneCreate,
    ScalarValue,
    SystemStatus,
    Vec3Value,
    ViewportSettings,
)

if TYPE_CHECKING:
    from aiohttp.web import WebSocketResponse
    from PySide6.QtWidgets import QMainWindow

log = logging.getLogger(__name__)


class CommandDispatcher:
    """Validate, route, and execute commands on the Qt main thread.

    The dispatcher holds a reference to the active window
    (``ViewerWindow`` or ``EditorWindow``), the :class:`QtBridge`,
    and optionally the :class:`AnimationEngine`.
    """

    def __init__(self, window: "QMainWindow", bridge: QtBridge, port: int = 8350) -> None:
        self._window = window
        self._bridge = bridge
        self._port = port
        self._ws_clients: Set["WebSocketResponse"] = set()
        self._animation_engine: Any = None  # set by server startup
        self._event_loop: Any = None  # set by server to the asyncio loop
        # Dynamic layers created via API, keyed by layer_id
        self._dynamic_layers: Dict[str, dict] = {}  # layer_id → metadata

        # Video recorder — lazily created on first use so importing
        # the recording package (which probes ffmpeg) doesn't happen
        # at server startup.  Stored on the dispatcher because it has
        # the same lifecycle as the server.
        self._recorder: Any = None

    # ── Animation engine attachment ───────────────────────────────────

    def set_animation_engine(self, engine: Any) -> None:
        """Called during startup if the animation package is available."""
        self._animation_engine = engine

    def set_event_loop(self, loop) -> None:
        """Store the asyncio event loop for scheduling broadcasts."""
        self._event_loop = loop

    # ── Internal property helpers ─────────────────────────────────────

    @property
    def _viewport(self):
        """Return the active GL viewport (viewer or editor)."""
        if hasattr(self._window, "viewport"):
            return self._window.viewport  # ViewerWindow
        return self._window.gl_viewport  # EditorWindow

    @property
    def _layer_manager(self):
        return self._window.layer_manager

    def _rebuild_layer_panel(self):
        """Refresh the UI layer panel after adding/removing layers."""
        panel = getattr(self._window, "layer_panel", None)
        if panel is not None:
            panel.rebuild()

    def _read_camera_state(self, vp=None) -> CameraState:
        vp = vp or self._viewport
        return CameraState(
            azimuth=vp.cam_azimuth,
            elevation=vp.cam_elevation,
            distance=vp.cam_distance,
            target=vp.cam_target.tolist(),
            fov=vp.cam_fov,
        )

    # ── Camera ────────────────────────────────────────────────────────

    async def get_camera_state(self) -> CameraState:
        return await self._bridge.invoke_on_qt(self._read_camera_state)

    async def set_camera(self, update: CameraUpdate) -> CameraState:
        def _set():
            vp = self._viewport
            if update.azimuth is not None:
                vp.cam_azimuth = update.azimuth
            if update.elevation is not None:
                vp.cam_elevation = update.elevation
            if update.distance is not None:
                vp.cam_distance = update.distance
            if update.target is not None:
                vp.cam_target = np.array(update.target, dtype=np.float64)
            if update.fov is not None:
                vp.cam_fov = update.fov
            vp.update()
            return self._read_camera_state(vp)

        return await self._bridge.invoke_on_qt(_set)

    async def fit_camera(self) -> dict:
        def _fit():
            self._viewport.fit_to_scene()
            return {"status": "ok"}

        return await self._bridge.invoke_on_qt(_fit)

    async def camera_preset(self, preset: str) -> CameraState:
        def _preset():
            vp = self._viewport
            presets = {
                "Top": (0, 89),
                "Front": (0, 0),
                "Right": (90, 0),
                "Isometric": (45, 35.264),
            }
            if preset in presets:
                vp.cam_azimuth, vp.cam_elevation = presets[preset]
                vp.update()
            return self._read_camera_state(vp)

        return await self._bridge.invoke_on_qt(_preset)

    async def camera_look_at(self, req: LookAtRequest) -> CameraState:
        def _look():
            vp = self._viewport
            vp.cam_target = np.array(req.target, dtype=np.float64)
            if req.distance is not None:
                vp.cam_distance = req.distance
            vp.update()
            return self._read_camera_state(vp)

        return await self._bridge.invoke_on_qt(_look)

    # ── Layers ────────────────────────────────────────────────────────

    async def get_layers(self) -> list:
        def _get():
            result = []
            for layer in self._layer_manager.layers:
                result.append(
                    LayerInfo(
                        id=layer.id,
                        name=layer.name,
                        type=layer.layer_type,
                        visible=layer.visible,
                        opacity=layer.opacity,
                        point_count=getattr(layer, "point_count", 0),
                        tri_count=getattr(layer, "tri_count", 0),
                        dynamic=getattr(layer, "dynamic", False),
                    ).model_dump()
                )
            return result

        return await self._bridge.invoke_on_qt(_get)

    async def update_layer(self, layer_id: str, update: LayerUpdate) -> dict:
        def _update():
            layer = self._layer_manager.get_layer(layer_id)
            if layer is None:
                raise ValueError(f"Layer not found: {layer_id}")
            if update.visible is not None:
                layer.visible = update.visible
            if update.opacity is not None:
                layer.opacity = update.opacity
            self._viewport.update()
            return {"status": "ok", "layer_id": layer_id}

        return await self._bridge.invoke_on_qt(_update)

    # ── Viewport ──────────────────────────────────────────────────────

    async def get_viewport_settings(self) -> dict:
        def _get():
            vp = self._viewport
            return ViewportSettings(
                point_size=vp.point_size,
                show_axes=vp.show_axes,
                show_grid=vp.show_grid,
                use_layer_colors=vp.use_layer_colors,
                fps_movement=vp.fps_movement,
                point_attenuation=vp.point_attenuation,
                bg_color=list(vp.bg_color) if vp.bg_color else None,
                vsync=bool(getattr(vp, "vsync", False)),
            ).model_dump()

        return await self._bridge.invoke_on_qt(_get)

    async def set_viewport_settings(self, settings: ViewportSettings) -> dict:
        def _set():
            vp = self._viewport
            if settings.point_size is not None:
                vp.point_size = settings.point_size
            if settings.show_axes is not None:
                vp.show_axes = settings.show_axes
            if settings.show_grid is not None:
                vp.show_grid = settings.show_grid
            if settings.use_layer_colors is not None:
                vp.use_layer_colors = settings.use_layer_colors
            if settings.fps_movement is not None:
                vp.fps_movement = settings.fps_movement
            if settings.point_attenuation is not None:
                vp.point_attenuation = settings.point_attenuation
            if settings.bg_color is not None:
                vp.bg_color = tuple(settings.bg_color)

            result = {"status": "ok"}
            if settings.vsync is not None:
                # Vsync is locked into the GL context at creation;
                # we can update the module-level default so the next
                # process honors it, but the live viewport is fixed.
                from locul3d.rendering.gl.viewport import set_default_vsync
                set_default_vsync(settings.vsync)
                current = bool(getattr(vp, "vsync", False))
                if settings.vsync != current:
                    result["vsync_restart_required"] = True
                    result["vsync_current"] = current
                    result["vsync_pending"] = settings.vsync
            vp.update()
            return result

        return await self._bridge.invoke_on_qt(_set)

    async def get_correction(self) -> dict:
        def _get():
            sc = self._viewport.scene_correction
            return CorrectionState(
                rotate_x=sc.rotate_x,
                rotate_y=sc.rotate_y,
                rotate_z=sc.rotate_z,
                shift_x=sc.shift_x,
                shift_y=sc.shift_y,
                shift_z=sc.shift_z,
            ).model_dump()

        return await self._bridge.invoke_on_qt(_get)

    async def set_correction(self, state: CorrectionState) -> dict:
        def _set():
            sc = self._viewport.scene_correction
            sc.rotate_x = state.rotate_x
            sc.rotate_y = state.rotate_y
            sc.rotate_z = state.rotate_z
            sc.shift_x = state.shift_x
            sc.shift_y = state.shift_y
            sc.shift_z = state.shift_z
            self._viewport.update()
            return {"status": "ok"}

        return await self._bridge.invoke_on_qt(_set)

    async def get_clip(self) -> dict:
        def _get():
            clip = self._viewport.scene_clip
            if clip is None:
                return {"active": False}
            return {
                "active": True,
                **ClipState(
                    x_min=clip[0],
                    x_max=clip[1],
                    y_min=clip[2],
                    y_max=clip[3],
                    z_min=clip[4],
                    z_max=clip[5],
                ).model_dump(),
            }

        return await self._bridge.invoke_on_qt(_get)

    async def set_clip(self, state: ClipState) -> dict:
        def _set():
            self._viewport.scene_clip = (
                state.x_min,
                state.x_max,
                state.y_min,
                state.y_max,
                state.z_min,
                state.z_max,
            )
            self._viewport.update()
            return {"status": "ok"}

        return await self._bridge.invoke_on_qt(_set)

    async def clear_clip(self) -> dict:
        def _clear():
            self._viewport.scene_clip = None
            self._viewport.update()
            return {"status": "ok"}

        return await self._bridge.invoke_on_qt(_clear)

    # ── Dynamic Layers ─────────────────────────────────────────────────

    async def list_dynamic_layers(self) -> list:
        def _list():
            result = []
            for lid, meta in self._dynamic_layers.items():
                layer = self._layer_manager.get_layer(lid)
                result.append(DynamicLayerInfo(
                    layer_id=lid,
                    name=meta["name"],
                    geometry_type=meta["geometry_type"],
                    visible=layer.visible if layer else meta.get("visible", True),
                    opacity=layer.opacity if layer else meta.get("opacity", 1.0),
                    color=layer.color[:3] if layer and layer.color else meta.get("color"),
                    point_count=layer.point_count if layer else 0,
                    tri_count=layer.tri_count if layer else 0,
                ).model_dump())
            return result
        return await self._bridge.invoke_on_qt(_list)

    async def create_dynamic_layer(self, req: DynamicLayerCreate) -> dict:
        def _create():
            from locul3d.core.layer import LayerData

            layer_id = f"dyn_{req.name}"
            if layer_id in self._dynamic_layers:
                raise ValueError(f"Dynamic layer already exists: {layer_id}")

            gt = req.geometry_type.value if isinstance(req.geometry_type, GeometryType) else req.geometry_type

            # Determine layer_type for LayerData
            if gt == "pointcloud":
                layer_type = "pointcloud"
            elif gt == "mesh":
                layer_type = "mesh"
            elif gt in ("bboxes", "surfaces"):
                layer_type = "mesh"  # rendered as triangle geometry
            elif gt == "file":
                layer_type = "mesh"
            else:
                layer_type = "pointcloud"

            color = list(req.color) if req.color else [0.5, 0.5, 0.5]
            layer_def = {
                "id": layer_id,
                "name": req.name,
                "type": layer_type,
                "visible": req.visible,
                "opacity": req.opacity,
                "color": color + [1.0],
            }
            layer = LayerData(layer_def, self._layer_manager.base_dir)
            layer.loaded = True
            layer.dynamic = True

            # Populate geometry based on type
            if gt == "pointcloud":
                if not req.points:
                    raise ValueError("pointcloud requires 'points'")
                layer.points = np.array(req.points, dtype=np.float32)
                layer.point_count = len(layer.points)
                if req.colors:
                    c = np.array(req.colors, dtype=np.float32)
                    # Normalise 0-255 range to 0-1
                    if c.max() > 1.0:
                        c = c / 255.0
                    layer.colors = c

            elif gt == "mesh":
                if not req.vertices or not req.triangles:
                    raise ValueError("mesh requires 'vertices' and 'triangles'")
                layer.points = np.array(req.vertices, dtype=np.float32)
                layer.triangles = np.array(req.triangles, dtype=np.uint32)
                layer.point_count = len(layer.points)
                layer.tri_count = len(layer.triangles)
                if req.normals:
                    layer.normals = np.array(req.normals, dtype=np.float32)

            elif gt == "bboxes":
                if not req.bboxes:
                    raise ValueError("bboxes requires 'bboxes' list")
                pts, tris = self._bboxes_to_mesh(req.bboxes)
                layer.points = pts
                layer.triangles = tris
                layer.point_count = len(pts)
                layer.tri_count = len(tris)

            elif gt == "surfaces":
                if not req.surfaces:
                    raise ValueError("surfaces requires 'surfaces' list")
                pts, tris = self._surfaces_to_mesh(req.surfaces)
                layer.points = pts
                layer.triangles = tris
                layer.point_count = len(pts)
                layer.tri_count = len(tris)

            elif gt == "file":
                if not req.path:
                    raise ValueError("file requires 'path'")
                import os
                if not os.path.exists(req.path):
                    raise ValueError(f"File not found: {req.path}")
                from locul3d.utils.io import load_geometry
                load_geometry(req.path, layer)

            # Add to scene
            self._layer_manager.layers.append(layer)
            self._layer_manager.invalidate_scene_aabb()

            # Track metadata
            self._dynamic_layers[layer_id] = {
                "name": req.name,
                "geometry_type": gt,
                "visible": req.visible,
                "opacity": req.opacity,
                "color": color,
            }

            self._rebuild_layer_panel()
            self._viewport.update()
            self.fire_event("event.dynamic_created", {
                "layer_id": layer_id, "name": req.name, "geometry_type": gt,
            })
            return DynamicLayerInfo(
                layer_id=layer_id,
                name=req.name,
                geometry_type=gt,
                visible=layer.visible,
                opacity=layer.opacity,
                color=color,
                point_count=layer.point_count,
                tri_count=layer.tri_count,
            ).model_dump()

        return await self._bridge.invoke_on_qt(_create)

    async def get_dynamic_layer(self, layer_id: str) -> dict:
        def _get():
            if layer_id not in self._dynamic_layers:
                raise ValueError(f"Dynamic layer not found: {layer_id}")
            meta = self._dynamic_layers[layer_id]
            layer = self._layer_manager.get_layer(layer_id)
            return DynamicLayerInfo(
                layer_id=layer_id,
                name=meta["name"],
                geometry_type=meta["geometry_type"],
                visible=layer.visible if layer else True,
                opacity=layer.opacity if layer else 1.0,
                color=layer.color[:3] if layer and layer.color else meta.get("color"),
                point_count=layer.point_count if layer else 0,
                tri_count=layer.tri_count if layer else 0,
            ).model_dump()
        return await self._bridge.invoke_on_qt(_get)

    async def update_dynamic_layer(self, layer_id: str, data: dict) -> dict:
        def _update():
            if layer_id not in self._dynamic_layers:
                raise ValueError(f"Dynamic layer not found: {layer_id}")
            meta = self._dynamic_layers[layer_id]
            layer = self._layer_manager.get_layer(layer_id)
            if layer is None:
                raise ValueError(f"Dynamic layer not found: {layer_id}")

            gt = meta["geometry_type"]

            # Replace geometry
            if gt == "pointcloud":
                if "points" in data:
                    layer.points = np.array(data["points"], dtype=np.float32)
                    layer.point_count = len(layer.points)
                    layer.evict_byte_caches()
                    layer.gpu_resident = False
                if "colors" in data:
                    c = np.array(data["colors"], dtype=np.float32)
                    if c.max() > 1.0:
                        c = c / 255.0
                    layer.colors = c
                    layer.evict_byte_caches()
                    layer.gpu_resident = False

            elif gt == "mesh":
                if "vertices" in data:
                    layer.points = np.array(data["vertices"], dtype=np.float32)
                    layer.point_count = len(layer.points)
                    layer.evict_byte_caches()
                    layer.gpu_resident = False
                if "triangles" in data:
                    layer.triangles = np.array(data["triangles"], dtype=np.uint32)
                    layer.tri_count = len(layer.triangles)
                    layer.evict_byte_caches()
                    layer.gpu_resident = False
                if "normals" in data:
                    layer.normals = np.array(data["normals"], dtype=np.float32)
                    layer.evict_byte_caches()
                    layer.gpu_resident = False

            elif gt == "bboxes" and "bboxes" in data:
                from .schemas import BBoxSpec
                specs = [BBoxSpec(**b) if isinstance(b, dict) else b for b in data["bboxes"]]
                pts, tris = self._bboxes_to_mesh(specs)
                layer.points = pts
                layer.triangles = tris
                layer.point_count = len(pts)
                layer.tri_count = len(tris)
                layer.evict_byte_caches()
                layer.gpu_resident = False

            elif gt == "surfaces" and "surfaces" in data:
                from .schemas import SurfaceSpec
                specs = [SurfaceSpec(**s) if isinstance(s, dict) else s for s in data["surfaces"]]
                pts, tris = self._surfaces_to_mesh(specs)
                layer.points = pts
                layer.triangles = tris
                layer.point_count = len(pts)
                layer.tri_count = len(tris)
                layer.evict_byte_caches()
                layer.gpu_resident = False

            # Update display properties if provided
            if "visible" in data:
                layer.visible = data["visible"]
            if "opacity" in data:
                layer.opacity = data["opacity"]
            if "color" in data:
                layer.color = list(data["color"]) + [1.0]
                meta["color"] = list(data["color"])

            self._layer_manager.invalidate_scene_aabb()
            self._viewport.update()
            self.fire_event("event.dynamic_updated", {"layer_id": layer_id})
            return DynamicLayerInfo(
                layer_id=layer_id,
                name=meta["name"],
                geometry_type=gt,
                visible=layer.visible,
                opacity=layer.opacity,
                color=layer.color[:3] if layer.color else meta.get("color"),
                point_count=layer.point_count,
                tri_count=layer.tri_count,
            ).model_dump()

        return await self._bridge.invoke_on_qt(_update)

    async def patch_dynamic_layer(self, layer_id: str, patch: DynamicLayerPatch) -> dict:
        def _patch():
            if layer_id not in self._dynamic_layers:
                raise ValueError(f"Dynamic layer not found: {layer_id}")
            meta = self._dynamic_layers[layer_id]
            layer = self._layer_manager.get_layer(layer_id)
            if layer is None:
                raise ValueError(f"Dynamic layer not found: {layer_id}")

            if patch.visible is not None:
                layer.visible = patch.visible
            if patch.opacity is not None:
                layer.opacity = patch.opacity
                layer.evict_byte_caches()  # RGBA bytes depend on opacity
            if patch.color is not None:
                layer.color = list(patch.color) + [1.0]
                meta["color"] = list(patch.color)

            self._viewport.update()
            return DynamicLayerInfo(
                layer_id=layer_id,
                name=meta["name"],
                geometry_type=meta["geometry_type"],
                visible=layer.visible,
                opacity=layer.opacity,
                color=layer.color[:3] if layer.color else meta.get("color"),
                point_count=layer.point_count,
                tri_count=layer.tri_count,
            ).model_dump()

        return await self._bridge.invoke_on_qt(_patch)

    async def delete_dynamic_layer(self, layer_id: str) -> dict:
        def _delete():
            if layer_id not in self._dynamic_layers:
                raise ValueError(f"Dynamic layer not found: {layer_id}")
            del self._dynamic_layers[layer_id]
            self._layer_manager.remove_layer(layer_id)
            self._rebuild_layer_panel()
            self._viewport.update()
            self.fire_event("event.dynamic_deleted", {"layer_id": layer_id})
            return {"status": "ok", "layer_id": layer_id}
        return await self._bridge.invoke_on_qt(_delete)

    async def clear_dynamic_layers(self) -> dict:
        def _clear():
            ids = list(self._dynamic_layers.keys())
            for lid in ids:
                self._layer_manager.remove_layer(lid)
            self._dynamic_layers.clear()
            self._rebuild_layer_panel()
            self._viewport.update()
            for lid in ids:
                self.fire_event("event.dynamic_deleted", {"layer_id": lid})
            return {"status": "ok"}
        return await self._bridge.invoke_on_qt(_clear)

    # ── Geometry Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bboxes_to_mesh(bboxes) -> tuple:
        """Convert a list of BBoxSpec to triangle mesh (vertices, indices)."""
        all_pts = []
        all_tris = []
        offset = 0
        for bbox in bboxes:
            cx, cy, cz = bbox.center
            sx, sy, sz = bbox.size
            hs = [sx / 2, sy / 2, sz / 2]
            corners = np.array([
                [cx - hs[0], cy - hs[1], cz - hs[2]],
                [cx + hs[0], cy - hs[1], cz - hs[2]],
                [cx + hs[0], cy + hs[1], cz - hs[2]],
                [cx - hs[0], cy + hs[1], cz - hs[2]],
                [cx - hs[0], cy - hs[1], cz + hs[2]],
                [cx + hs[0], cy - hs[1], cz + hs[2]],
                [cx + hs[0], cy + hs[1], cz + hs[2]],
                [cx - hs[0], cy + hs[1], cz + hs[2]],
            ], dtype=np.float32)

            if bbox.rotation_z != 0.0:
                import math
                rad = math.radians(bbox.rotation_z)
                c, s = math.cos(rad), math.sin(rad)
                rel = corners - [cx, cy, cz]
                x, y = rel[:, 0].copy(), rel[:, 1].copy()
                rel[:, 0] = c * x - s * y
                rel[:, 1] = s * x + c * y
                corners = rel + [cx, cy, cz]

            # 12 triangles (2 per face)
            faces = [
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 6, 5], [4, 7, 6],  # top
                [0, 5, 1], [0, 4, 5],  # front
                [2, 7, 3], [2, 6, 7],  # back
                [0, 3, 7], [0, 7, 4],  # left
                [1, 5, 6], [1, 6, 2],  # right
            ]
            all_pts.append(corners)
            all_tris.append(np.array(faces, dtype=np.uint32) + offset)
            offset += 8

        if not all_pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
        return np.vstack(all_pts), np.vstack(all_tris)

    @staticmethod
    def _surfaces_to_mesh(surfaces) -> tuple:
        """Convert a list of SurfaceSpec to triangle mesh (vertices, indices)."""
        all_pts = []
        all_tris = []
        offset = 0
        for surf in surfaces:
            cx, cy, cz = surf.center
            w, h = surf.size
            hw, hh = w / 2, h / 2
            axis = surf.axis
            if axis == "xy":
                corners = np.array([
                    [cx - hw, cy - hh, cz],
                    [cx + hw, cy - hh, cz],
                    [cx + hw, cy + hh, cz],
                    [cx - hw, cy + hh, cz],
                ], dtype=np.float32)
            elif axis == "xz":
                corners = np.array([
                    [cx - hw, cy, cz - hh],
                    [cx + hw, cy, cz - hh],
                    [cx + hw, cy, cz + hh],
                    [cx - hw, cy, cz + hh],
                ], dtype=np.float32)
            else:  # yz
                corners = np.array([
                    [cx, cy - hw, cz - hh],
                    [cx, cy + hw, cz - hh],
                    [cx, cy + hw, cz + hh],
                    [cx, cy - hw, cz + hh],
                ], dtype=np.float32)
            faces = [[0, 1, 2], [0, 2, 3]]
            all_pts.append(corners)
            all_tris.append(np.array(faces, dtype=np.uint32) + offset)
            offset += 4

        if not all_pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
        return np.vstack(all_pts), np.vstack(all_tris)

    # ── System ────────────────────────────────────────────────────────

    async def get_status(self) -> dict:
        def _status():
            lm = self._layer_manager
            mode = "editor" if hasattr(self._window, "gl_viewport") else "viewer"
            total_points = sum(
                getattr(l, "point_count", 0) for l in lm.layers
            )
            return SystemStatus(
                mode=mode,
                layers_count=len(lm.layers),
                dynamic_layers_count=len(self._dynamic_layers),
                total_points=total_points,
                fps=0.0,  # TODO: read from viewport FPS tracker
                server_port=self._port,
            ).model_dump()

        return await self._bridge.invoke_on_qt(_status)

    async def take_screenshot(self) -> bytes:
        """Grab the viewport framebuffer as PNG bytes."""
        def _grab():
            from PySide6.QtCore import QBuffer, QIODevice

            img = self._viewport.grabFramebuffer()
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            img.save(buf, "PNG")
            return bytes(buf.data())

        return await self._bridge.invoke_on_qt(_grab)

    # ── Annotations (Editor Overlays) ────────────────────────────────

    @property
    def _annotations(self):
        """Editor annotations list (raises AttributeError if viewer)."""
        return self._window.annotations

    @property
    def _planes(self):
        """Editor planes list (raises AttributeError if viewer)."""
        return self._window.planes

    async def list_bboxes(self) -> list:
        def _list():
            return [bbox.to_dict() for bbox in self._annotations]
        return await self._bridge.invoke_on_qt(_list)

    async def create_bbox(self, req: BBoxCreate) -> dict:
        def _create():
            from locul3d.core.geometry import BBoxItem
            bbox = BBoxItem(
                label=req.label,
                center=req.center,
                size=req.size,
                rotation_z=req.rotation_z,
                color=req.color,
                fill_opacity=req.fill_opacity,
            )
            self._annotations.append(bbox)
            self._viewport.update()
            idx = len(self._annotations) - 1
            self.fire_event("event.bbox_created", {"index": idx, "bbox": bbox.to_dict()})
            return {"status": "ok", "index": idx, **bbox.to_dict()}
        return await self._bridge.invoke_on_qt(_create)

    async def update_bbox(self, index: int, update: BBoxUpdate) -> dict:
        def _update():
            annots = self._annotations
            if index < 0 or index >= len(annots):
                raise IndexError(f"BBox index out of range: {index}")
            bbox = annots[index]
            if update.label is not None:
                bbox.label = update.label
            if update.center is not None:
                bbox.center_pos = np.array(update.center, dtype=np.float64)
            if update.size is not None:
                bbox.size = np.array(update.size, dtype=np.float64)
            if update.color is not None:
                bbox.color = list(update.color)
            if update.rotation_z is not None:
                bbox.rotation_z = update.rotation_z
            if update.fill_opacity is not None:
                bbox.fill_opacity = update.fill_opacity
            self._viewport.update()
            return {"status": "ok", "index": index, **bbox.to_dict()}
        return await self._bridge.invoke_on_qt(_update)

    async def delete_bbox(self, index: int) -> dict:
        def _delete():
            annots = self._annotations
            if index < 0 or index >= len(annots):
                raise IndexError(f"BBox index out of range: {index}")
            annots.pop(index)
            self._viewport.update()
            self.fire_event("event.bbox_deleted", {"index": index})
            return {"status": "ok", "index": index}
        return await self._bridge.invoke_on_qt(_delete)

    async def list_planes(self) -> list:
        def _list():
            return [plane.to_dict() for plane in self._planes]
        return await self._bridge.invoke_on_qt(_list)

    async def create_plane(self, req: PlaneCreate) -> dict:
        def _create():
            from locul3d.core.geometry import PlaneItem
            plane = PlaneItem(
                axis=req.axis,
                center=req.center,
                size=req.size,
                color=req.color,
                opacity=req.opacity,
            )
            self._planes.append(plane)
            self._viewport.update()
            idx = len(self._planes) - 1
            return {"status": "ok", "index": idx, **plane.to_dict()}
        return await self._bridge.invoke_on_qt(_create)

    async def delete_plane(self, index: int) -> dict:
        def _delete():
            planes = self._planes
            if index < 0 or index >= len(planes):
                raise IndexError(f"Plane index out of range: {index}")
            planes.pop(index)
            self._viewport.update()
            return {"status": "ok", "index": index}
        return await self._bridge.invoke_on_qt(_delete)

    # ── Recording ────────────────────────────────────────────────────

    @property
    def recorder(self):
        """Lazily-created VideoRecorder.

        Importing the recording package probes ``ffmpeg``; we defer
        that until the first request so the editor still starts
        cleanly on systems without ffmpeg.
        """
        if self._recorder is None:
            from locul3d.recording.recorder import VideoRecorder
            self._recorder = VideoRecorder()
            if self._animation_engine is not None:
                self._animation_engine.attach_recorder(self._recorder)
        return self._recorder

    def _set_input_locked(self, locked: bool) -> None:
        """Lock/unlock input for the recording lifecycle.

        Locks the GL viewport's own input handlers so mouse/keyboard
        events are ignored.  We intentionally do NOT call
        ``window.setEnabled(False)`` — on Windows that disables the
        entire widget tree including the QOpenGLWidget, which
        prevents ``makeCurrent()`` from activating the GL context
        and causes ``render_to_buffer()`` to produce blank frames.
        """
        try:
            self._viewport.set_input_locked(locked)
        except Exception:
            pass

    async def start_recording(
        self,
        *,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str,
        hw_pref: str,
        bitrate_kbps: Optional[int],
        grid: Optional[bool] = None,
        axes: Optional[bool] = None,
        bg_color: Optional[list] = None,
    ) -> dict:
        def _start():
            rec = self.recorder
            if self._animation_engine is None:
                raise RuntimeError("animation engine not available")
            cfg = rec.start(
                path=path,
                width=width, height=height, fps=fps,
                codec=codec, hw_pref=hw_pref,
                bitrate_kbps=bitrate_kbps,
            )
            # Switch engine into capture mode and lock input.
            self._animation_engine._render_mode = "capture"
            self._animation_engine.attach_recorder(rec)
            self._animation_engine._frame_number = 0
            try:
                self._viewport.set_capture_in_progress(True)
            except AttributeError:
                pass

            # Per-recording viewport overrides.  Save the originals
            # so we can restore them on stop().  None ⇒ inherit, no
            # save/restore needed.
            vp = self._viewport
            self._rec_overrides = {}
            if grid is not None:
                self._rec_overrides["show_grid"] = vp.show_grid
                vp.show_grid = bool(grid)
            if axes is not None:
                self._rec_overrides["show_axes"] = vp.show_axes
                vp.show_axes = bool(axes)
            if bg_color is not None:
                self._rec_overrides["bg_color"] = tuple(vp.bg_color)
                # Accept rgba (4 floats) or rgb (3 floats, alpha=1).
                rgba = list(bg_color)
                if len(rgba) == 3:
                    rgba.append(1.0)
                vp.bg_color = tuple(rgba)

            self._set_input_locked(True)
            return {
                "status": "ok",
                "config": {
                    "path": str(cfg.path),
                    "width": cfg.width, "height": cfg.height,
                    "fps": cfg.fps, "codec": cfg.codec,
                    "encoder": cfg.encoder, "encoder_kind": cfg.encoder_kind,
                    "bitrate_kbps": cfg.bitrate_kbps,
                    "show_grid": bool(vp.show_grid),
                    "show_axes": bool(vp.show_axes),
                    "bg_color": list(vp.bg_color),
                },
                "warnings": list(rec.stats.warnings),
            }
        return await self._bridge.invoke_on_qt(_start)

    async def stop_recording(self) -> dict:
        def _stop():
            rec = self.recorder
            stats = rec.stop()
            # Restore engine to realtime and re-enable input.
            if self._animation_engine is not None:
                self._animation_engine._render_mode = "realtime"
                try:
                    self._animation_engine._stop_capture_session()
                except Exception:
                    pass
            # Restore any viewport overrides applied at start.
            overrides = getattr(self, "_rec_overrides", None)
            if overrides:
                vp = self._viewport
                for attr, value in overrides.items():
                    try:
                        setattr(vp, attr, value)
                    except Exception:
                        pass
                self._rec_overrides = {}
                try:
                    vp.update()
                except Exception:
                    pass
            self._set_input_locked(False)
            return {
                "status": "ok",
                "frames_written": stats.frames_written,
                "frames_dropped": stats.frames_dropped,
                "bytes_written": stats.bytes_written,
                "duration_s": stats.duration_s,
                "last_error": stats.last_error,
            }
        return await self._bridge.invoke_on_qt(_stop)

    async def pause_recording(self) -> dict:
        def _pause():
            self.recorder.pause()
            return {"status": "ok", "state": self.recorder.state}
        return await self._bridge.invoke_on_qt(_pause)

    async def resume_recording(self) -> dict:
        def _resume():
            self.recorder.resume()
            return {"status": "ok", "state": self.recorder.state}
        return await self._bridge.invoke_on_qt(_resume)

    # ── WS Command Routing ────────────────────────────────────────────

    _WS_ANIMATION_COMMANDS = frozenset({
        "camera.animate",
        "camera.transform_continuous",
        "dynamic.animate",
        "dynamic.transform_continuous",
        "dynamic.transform",
        "animation.stop",
        "transform.stop",
        "transform.stop_all",
        "animation.set_realtime_fps",
        "animation.get_realtime_fps",
        "animation.set_preview_mode",
        "animation.set_time_scale",
    })

    async def handle_ws_command(self, msg_type: str, data: dict) -> dict:
        """Route a WS command to the appropriate handler."""

        # Animation commands → AnimationEngine (on Qt thread)
        if msg_type in self._WS_ANIMATION_COMMANDS:
            if self._animation_engine is None:
                raise ValueError("Animation engine not available")
            return await self._bridge.invoke_on_qt(
                lambda: self._animation_engine.handle_command(msg_type, data)
            )

        # Render mode commands
        if msg_type.startswith("render."):
            return await self._bridge.invoke_on_qt(
                lambda: self._handle_render_command(msg_type, data)
            )

        # Camera
        if msg_type == "camera.set":
            return (await self.set_camera(CameraUpdate(**data))).model_dump()
        if msg_type == "camera.set_azimuth":
            v = ScalarValue(**data)
            return (await self.set_camera(CameraUpdate(azimuth=v.value))).model_dump()
        if msg_type == "camera.set_elevation":
            v = ScalarValue(**data)
            return (await self.set_camera(CameraUpdate(elevation=v.value))).model_dump()
        if msg_type == "camera.set_distance":
            v = ScalarValue(**data)
            return (await self.set_camera(CameraUpdate(distance=v.value))).model_dump()
        if msg_type == "camera.set_fov":
            v = ScalarValue(**data)
            return (await self.set_camera(CameraUpdate(fov=v.value))).model_dump()
        if msg_type == "camera.set_target":
            v = Vec3Value(**data)
            return (await self.set_camera(CameraUpdate(target=v.value))).model_dump()
        if msg_type == "camera.preset":
            p = CameraPreset(**data)
            return (await self.camera_preset(p.preset)).model_dump()
        if msg_type == "camera.fit":
            return await self.fit_camera()

        # Layer
        if msg_type == "layer.set":
            lid = data.pop("layer_id", None)
            if lid is None:
                raise ValueError("layer_id required")
            return await self.update_layer(lid, LayerUpdate(**data))
        if msg_type == "layer.set_all":
            visible = data.get("visible")
            if visible is None:
                raise ValueError("visible required")
            return await self._bridge.invoke_on_qt(
                lambda: self._do_set_all_visible(visible)
            )

        # Dynamic layers
        if msg_type == "dynamic.create":
            return await self.create_dynamic_layer(DynamicLayerCreate(**data))
        if msg_type == "dynamic.update":
            lid = data.pop("layer_id", None)
            if lid is None:
                raise ValueError("layer_id required")
            return await self.update_dynamic_layer(lid, data)
        if msg_type == "dynamic.patch":
            lid = data.pop("layer_id", None)
            if lid is None:
                raise ValueError("layer_id required")
            return await self.patch_dynamic_layer(lid, DynamicLayerPatch(**data))
        if msg_type == "dynamic.delete":
            lid = data.get("layer_id")
            if lid is None:
                raise ValueError("layer_id required")
            return await self.delete_dynamic_layer(lid)
        if msg_type == "dynamic.clear":
            return await self.clear_dynamic_layers()

        # Annotations (editor overlays)
        if msg_type == "bbox.create":
            return await self.create_bbox(BBoxCreate(**data))
        if msg_type == "bbox.update":
            idx = data.pop("index", None)
            if idx is None:
                raise ValueError("index required")
            return await self.update_bbox(int(idx), BBoxUpdate(**data))
        if msg_type == "bbox.delete":
            idx = data.get("index")
            if idx is None:
                raise ValueError("index required")
            return await self.delete_bbox(int(idx))

        # Scene
        if msg_type == "scene.load":
            from .schemas import SceneLoadRequest
            req = SceneLoadRequest(**data)
            return await self._bridge.invoke_on_qt(
                lambda: self._do_scene_load(req.paths)
            )
        if msg_type == "scene.clear":
            return await self._bridge.invoke_on_qt(self._do_scene_clear)

        # Viewport
        if msg_type == "viewport.set":
            return await self.set_viewport_settings(ViewportSettings(**data))
        if msg_type == "correction.set":
            return await self.set_correction(CorrectionState(**data))
        if msg_type == "clip.set":
            return await self.set_clip(ClipState(**data))
        if msg_type == "clip.clear":
            return await self.clear_clip()

        # Screenshot
        if msg_type == "screenshot.capture":
            import base64

            png = await self.take_screenshot()
            return {"image": base64.b64encode(png).decode(), "format": "png"}

        raise ValueError(f"Unknown command type: {msg_type}")

    # ── WS helper callables (run on Qt thread) ───────────────────────

    def _do_set_all_visible(self, visible: bool) -> dict:
        self._layer_manager.set_all_visible(visible)
        self._viewport.update()
        return {"status": "ok"}

    def _do_scene_load(self, paths: list) -> dict:
        window = self._window
        for path in paths:
            if hasattr(window, "_load_files"):
                window._load_files([path])
            elif hasattr(window, "load_files"):
                window.load_files([path])
        self._rebuild_layer_panel()
        self.fire_event("event.scene_loaded", {
            "layers": [
                {"id": l.id, "name": l.name, "type": l.layer_type,
                 "point_count": getattr(l, "point_count", 0)}
                for l in self._layer_manager.layers
            ]
        })
        return {"status": "ok", "loaded": len(paths)}

    def _do_scene_clear(self) -> dict:
        self._layer_manager.clear()
        self._dynamic_layers.clear()
        self._rebuild_layer_panel()
        self._viewport.update()
        self.fire_event("event.scene_cleared", {})
        return {"status": "ok"}

    # ── Binary WS (high-perf point streaming) ──────────────────────

    async def handle_ws_binary(self, raw: bytes) -> dict:
        """Handle a binary WS message for high-throughput point streaming.

        Byte layout:
          [0..3]   uint32  message_type (1=create_points, 2=append_points)
          [4..7]   uint32  name_length (N)
          [8..8+N] utf8    layer_name
          [8+N..]  float32 interleaved XYZ XYZ ... (points)
        """
        import struct

        if len(raw) < 8:
            raise ValueError("Binary message too short")

        msg_type, name_len = struct.unpack_from("<II", raw, 0)
        if len(raw) < 8 + name_len:
            raise ValueError("Binary message truncated (name)")
        layer_name = raw[8:8 + name_len].decode("utf-8")
        point_data = raw[8 + name_len:]

        if len(point_data) % 12 != 0:
            raise ValueError("Point data not aligned to 12-byte XYZ float32 triples")

        points = np.frombuffer(point_data, dtype=np.float32).reshape(-1, 3).copy()
        layer_id = f"dyn_{layer_name}"

        if msg_type == 1:
            # create_points — create a new dynamic point cloud layer
            req = DynamicLayerCreate(
                name=layer_name,
                geometry_type=GeometryType.POINTCLOUD,
                points=points.tolist(),
            )
            return await self.create_dynamic_layer(req)

        elif msg_type == 2:
            # append_points — append to existing dynamic layer
            def _append():
                if layer_id not in self._dynamic_layers:
                    raise ValueError(f"Dynamic layer not found: {layer_id}")
                layer = self._layer_manager.get_layer(layer_id)
                if layer is None:
                    raise ValueError(f"Layer not found in scene: {layer_id}")
                if layer.points is not None and len(layer.points) > 0:
                    layer.points = np.vstack([layer.points, points])
                else:
                    layer.points = points
                layer.point_count = len(layer.points)
                layer.evict_byte_caches()
                layer.gpu_resident = False
                self._viewport.update()
                self.fire_event("event.dynamic_updated", {"layer_id": layer_id})
                return {"status": "ok", "layer_id": layer_id,
                        "point_count": layer.point_count}
            return await self._bridge.invoke_on_qt(_append)

        else:
            raise ValueError(f"Unknown binary message type: {msg_type}")

    def _handle_render_command(self, msg_type: str, data: dict) -> dict:
        """Handle render mode commands (runs on Qt thread)."""

        if msg_type == "render.set_mode":
            mode = data.get("mode", "realtime")
            width = data.get("width")
            height = data.get("height")
            return self._set_render_mode(mode, width, height)

        if msg_type == "render.get_mode":
            return self._get_render_mode()

        if msg_type == "render.capture_frame":
            save_to = data.get("save_to")
            fmt = data.get("format", "png")
            return self._capture_frame(save_to, fmt)

        if msg_type == "render.set_target_fps":
            fps = data.get("fps", 60)
            self._capture_target_fps = max(1, int(fps))
            return {"status": "ok", "target_fps": self._capture_target_fps}

        return {"status": "error", "message": f"Unknown render command: {msg_type}"}

    def _get_render_mode(self) -> dict:
        mode = getattr(self, "_render_mode", "realtime")
        return {
            "mode": mode,
            "width": getattr(self, "_capture_width", None),
            "height": getattr(self, "_capture_height", None),
            "target_fps": getattr(self, "_capture_target_fps", 60),
        }

    def _set_render_mode(self, mode: str, width=None, height=None) -> dict:
        self._render_mode = mode
        vp = self._viewport

        if mode == "capture":
            self._capture_width = width
            self._capture_height = height
            # Resize viewport for capture resolution
            if width and height:
                vp.setFixedSize(width, height)
            # Disable LOD for full quality
            vp._interacting = False
            if self._animation_engine:
                self._animation_engine._render_mode = "capture"
            try:
                vp.set_capture_in_progress(True)
            except AttributeError:
                pass
        else:
            # Restore realtime mode
            from PySide6.QtWidgets import QWIDGETSIZE_MAX
            vp.setMinimumSize(0, 0)
            vp.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
            if self._animation_engine:
                self._animation_engine._render_mode = "realtime"
            try:
                vp.set_capture_in_progress(False)
            except AttributeError:
                pass

        self.fire_event("event.render_mode_changed", {
            "mode": mode, "width": width, "height": height,
        })
        vp.update()
        return {"status": "ok", "mode": mode}

    def _capture_frame(self, save_to=None, fmt="png") -> dict:
        """Render one frame at full quality and return/save it."""
        vp = self._viewport

        # Advance animation engine by one tick if in capture mode
        if self._animation_engine and getattr(self, "_render_mode", "realtime") == "capture":
            self._animation_engine._tick()

        # Force synchronous full-quality render
        vp._interacting = False
        vp.update()
        vp.repaint()

        # Grab framebuffer
        img = vp.grabFramebuffer()
        w, h = img.width(), img.height()

        # Save to disk if requested
        if save_to:
            img.save(save_to, fmt.upper())
            # When saving to disk, skip expensive base64 transfer
            return {
                "status": "ok",
                "format": fmt,
                "width": w,
                "height": h,
                "saved_to": save_to,
            }

        # Return as base64 (no save_to — client wants the image data)
        import base64
        from PySide6.QtCore import QBuffer, QIODevice
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, fmt.upper())
        b64 = base64.b64encode(bytes(buf.data())).decode()

        return {
            "status": "ok",
            "format": fmt,
            "width": w,
            "height": h,
            "size_bytes": len(buf.data()),
            "image": b64,
        }

    # ── Event Broadcasting ────────────────────────────────────────────

    def register_ws(self, ws: "WebSocketResponse") -> None:
        self._ws_clients.add(ws)
        log.debug("WS client connected (%d total)", len(self._ws_clients))

    def unregister_ws(self, ws: "WebSocketResponse") -> None:
        self._ws_clients.discard(ws)
        log.debug("WS client disconnected (%d total)", len(self._ws_clients))

    def fire_event(self, event_type: str, data: dict) -> None:
        """Schedule an event broadcast from any thread (Qt or asyncio).

        Safe to call from inside ``invoke_on_qt`` callbacks — the actual
        send runs on the asyncio loop.
        """
        if not self._ws_clients:
            return
        if self._event_loop is None:
            return
        import asyncio
        asyncio.run_coroutine_threadsafe(
            self.broadcast_event(event_type, data), self._event_loop
        )

    async def broadcast_event(self, event_type: str, data: dict) -> None:
        """Push an event to all connected WebSocket clients."""
        msg = {"type": event_type, **data}
        dead: set = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead
