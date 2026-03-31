"""Command dispatcher — single point of contact between network and Qt.

Every viewer/editor mutation funnelled through ``invoke_on_qt()`` to
guarantee thread safety.  The dispatcher also owns the WebSocket client
registry and event broadcasting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

import numpy as np

from .bridge import QtBridge
from .schemas import (
    CameraPreset,
    CameraState,
    CameraUpdate,
    ClipState,
    CorrectionState,
    DynamicLayerInfo,
    DynamicLayerPatch,
    LayerInfo,
    LayerUpdate,
    LookAtRequest,
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

    # ── Animation engine attachment ───────────────────────────────────

    def set_animation_engine(self, engine: Any) -> None:
        """Called during startup if the animation package is available."""
        self._animation_engine = engine

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
            vp.update()
            return {"status": "ok"}

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

    def _handle_render_command(self, msg_type: str, data: dict) -> dict:
        """Handle render mode commands (runs on Qt thread)."""
        # Placeholder — will be implemented in Phase 5
        return {"status": "ok", "message": f"render command {msg_type} not yet implemented"}

    # ── Event Broadcasting ────────────────────────────────────────────

    def register_ws(self, ws: "WebSocketResponse") -> None:
        self._ws_clients.add(ws)
        log.debug("WS client connected (%d total)", len(self._ws_clients))

    def unregister_ws(self, ws: "WebSocketResponse") -> None:
        self._ws_clients.discard(ws)
        log.debug("WS client disconnected (%d total)", len(self._ws_clients))

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
