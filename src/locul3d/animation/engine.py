"""Animation engine — QTimer-driven tick loop managing all active tracks.

Runs entirely on the Qt main thread.  The server sends animation
*declarations*; the engine ticks them locally at 125 Hz for smooth,
latency-independent playback.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from .easing import resolve_easing
from .tracks import (
    AnimationTrack,
    CameraTrack,
    ContinuousTransformTrack,
    DynamicLayerTrack,
)

if TYPE_CHECKING:
    from locul3d.remote.dispatcher import CommandDispatcher

log = logging.getLogger(__name__)


class AnimationEngine(QObject):
    """Manages animation tracks and ticks them via a QTimer.

    The timer runs at ~125 Hz (8 ms interval) using PreciseTimer.
    Animation math uses ``time.perf_counter()`` deltas, so animations
    stay time-accurate even if frames drop.
    """

    frame_ready = Signal(int)  # frame number (for capture mode)

    def __init__(self, viewport: Any, dispatcher: Any = None) -> None:
        super().__init__()
        self._viewport = viewport
        self._dispatcher = dispatcher
        self._tracks: List[AnimationTrack] = []
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(8)  # 125 Hz
        self._frame_number = 0
        self._render_mode = "realtime"

    def start(self) -> None:
        """Start the animation timer."""
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """Stop the animation timer."""
        self._timer.stop()

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    # ── Track Management ─────────────────────────────────────────────

    def add_track(self, track: AnimationTrack) -> None:
        """Add a track, replacing any existing track with the same ID."""
        self.remove_track(track.id)
        self._tracks.append(track)
        if not self._timer.isActive():
            self.start()

    def remove_track(self, track_id: str) -> bool:
        """Remove a track by ID. Returns True if found."""
        for i, track in enumerate(self._tracks):
            if track.id == track_id:
                track.done = True
                self._tracks.pop(i)
                return True
        return False

    def remove_all(self) -> None:
        """Remove all tracks."""
        for track in self._tracks:
            track.done = True
        self._tracks.clear()

    def get_track(self, track_id: str) -> Optional[AnimationTrack]:
        for track in self._tracks:
            if track.id == track_id:
                return track
        return None

    # ── Timer Tick ───────────────────────────────────────────────────

    def _tick(self) -> None:
        """Called by QTimer at ~125 Hz. Ticks all active tracks."""
        now = time.perf_counter()
        dirty = False

        for track in self._tracks:
            if track.tick(now):
                dirty = True

        # Remove completed tracks
        self._tracks = [t for t in self._tracks if not t.done]

        # Stop timer if no more tracks
        if not self._tracks:
            self._timer.stop()

        if dirty:
            if self._render_mode == "capture":
                self._viewport._interacting = False
                self._viewport.update()
                self._viewport.repaint()
                self.frame_ready.emit(self._frame_number)
                self._frame_number += 1
            else:
                self._viewport.update()

    # ── Command Handler (called from dispatcher) ─────────────────────

    def handle_command(self, msg_type: str, data: dict) -> dict:
        """Handle an animation WS command. Runs on the Qt main thread."""

        if msg_type == "camera.animate":
            return self._cmd_camera_animate(data)
        if msg_type == "camera.transform_continuous":
            return self._cmd_camera_continuous(data)
        if msg_type == "dynamic.animate":
            return self._cmd_dynamic_animate(data)
        if msg_type == "dynamic.transform_continuous":
            return self._cmd_dynamic_continuous(data)
        if msg_type == "dynamic.transform":
            return self._cmd_dynamic_instant(data)
        if msg_type == "animation.stop":
            return self._cmd_stop(data)
        if msg_type == "transform.stop":
            return self._cmd_stop(data)
        if msg_type == "transform.stop_all":
            return self._cmd_stop_all()

        raise ValueError(f"Unknown animation command: {msg_type}")

    # ── Camera Animate ───────────────────────────────────────────────

    def _cmd_camera_animate(self, data: dict) -> dict:
        track_id = data.get("track_id") or data.get("id", "camera-anim")
        keyframes = data.get("keyframes", [])
        duration_ms = data.get("duration_ms", 3000)
        easing_spec = data.get("easing", "ease_in_out")
        loop = data.get("loop", False)
        ping_pong = data.get("ping_pong", False)
        repeat_count = data.get("repeat_count", 0)

        easing_fn = resolve_easing(easing_spec)

        # Convert keyframes from Pydantic models or dicts
        kf_dicts = []
        for kf in keyframes:
            if isinstance(kf, dict):
                kf_dicts.append(kf)
            else:
                kf_dicts.append(kf.model_dump(exclude_none=True))

        def on_done(tid):
            if self._dispatcher:
                self._dispatcher.fire_event("event.animation_done", {"id": tid})

        track = CameraTrack(
            track_id=track_id,
            viewport=self._viewport,
            keyframes=kf_dicts,
            duration_ms=duration_ms,
            easing=easing_fn,
            loop=loop,
            ping_pong=ping_pong,
            repeat_count=repeat_count,
            on_done=on_done,
        )
        self.add_track(track)

        if self._dispatcher:
            self._dispatcher.fire_event("event.animation_started", {
                "id": track_id, "type": "camera",
            })

        return {"status": "ok", "id": track_id}

    # ── Camera Continuous Transform ──────────────────────────────────

    def _cmd_camera_continuous(self, data: dict) -> dict:
        track_id = data.get("track_id") or data.get("id", "camera-continuous")
        prop = data.get("property", "azimuth")
        rate = data.get("rate", 0.0)
        target = data.get("target")
        duration_ms = data.get("duration_ms", 0)

        vp = self._viewport
        prop_map = {
            "azimuth": ("cam_azimuth", lambda: vp.cam_azimuth, lambda v: setattr(vp, "cam_azimuth", v)),
            "elevation": ("cam_elevation", lambda: vp.cam_elevation, lambda v: setattr(vp, "cam_elevation", v)),
            "distance": ("cam_distance", lambda: vp.cam_distance, lambda v: setattr(vp, "cam_distance", v)),
            "fov": ("cam_fov", lambda: vp.cam_fov, lambda v: setattr(vp, "cam_fov", v)),
        }

        if prop not in prop_map:
            raise ValueError(f"Unknown camera property: {prop}")

        _, getter, setter = prop_map[prop]

        def on_done(tid, reason):
            if self._dispatcher:
                self._dispatcher.fire_event("event.transform_stopped", {
                    "id": tid, "reason": reason,
                })

        track = ContinuousTransformTrack(
            track_id=track_id,
            target_obj=vp,
            property_name=prop,
            rate=rate,
            target_value=target,
            duration_ms=duration_ms,
            getter=getter,
            setter=setter,
            on_done=on_done,
        )
        self.add_track(track)

        if self._dispatcher:
            self._dispatcher.fire_event("event.transform_started", {
                "id": track_id, "property": prop,
            })

        return {"status": "ok", "id": track_id}

    # ── Dynamic Layer Animate ────────────────────────────────────────

    def _cmd_dynamic_animate(self, data: dict) -> dict:
        track_id = data.get("track_id") or data.get("id", "layer-anim")
        layer_id = data.get("layer_id")
        if not layer_id:
            raise ValueError("layer_id required")

        keyframes = data.get("keyframes", [])
        duration_ms = data.get("duration_ms", 3000)
        easing_spec = data.get("easing", "ease_in_out")
        loop = data.get("loop", False)
        ping_pong = data.get("ping_pong", False)
        repeat_count = data.get("repeat_count", 0)

        # Support per-property easing
        if isinstance(easing_spec, dict) and not any(
            k in easing_spec for k in ("cubic_bezier", "spring", "steps")
        ):
            # Per-property easing — use default for the main track
            easing_fn = resolve_easing("ease_in_out")
        else:
            easing_fn = resolve_easing(easing_spec)

        kf_dicts = []
        for kf in keyframes:
            if isinstance(kf, dict):
                kf_dicts.append(kf)
            else:
                kf_dicts.append(kf.model_dump(exclude_none=True))

        def on_done(tid):
            if self._dispatcher:
                self._dispatcher.fire_event("event.animation_done", {"id": tid})

        if not self._dispatcher:
            raise ValueError("Dispatcher not available")

        track = DynamicLayerTrack(
            track_id=track_id,
            layer_id=layer_id,
            dispatcher=self._dispatcher,
            keyframes=kf_dicts,
            duration_ms=duration_ms,
            easing=easing_fn,
            loop=loop,
            ping_pong=ping_pong,
            repeat_count=repeat_count,
            on_done=on_done,
        )
        self.add_track(track)

        if self._dispatcher:
            self._dispatcher.fire_event("event.animation_started", {
                "id": track_id, "type": "dynamic",
            })

        return {"status": "ok", "id": track_id}

    # ── Dynamic Layer Continuous Transform ────────────────────────────

    def _cmd_dynamic_continuous(self, data: dict) -> dict:
        track_id = data.get("track_id") or data.get("id", "layer-continuous")
        layer_id = data.get("layer_id")
        if not layer_id:
            raise ValueError("layer_id required")

        prop = data.get("property", "opacity")
        rate = data.get("rate", 0.0)
        target = data.get("target")
        duration_ms = data.get("duration_ms", 0)

        if not self._dispatcher:
            raise ValueError("Dispatcher not available")

        layer = self._dispatcher._layer_manager.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Layer not found: {layer_id}")

        # Build getter/setter for the property
        getter, setter = self._layer_prop_accessors(layer, layer_id, prop)

        def on_done(tid, reason):
            if self._dispatcher:
                self._dispatcher.fire_event("event.transform_stopped", {
                    "id": tid, "reason": reason,
                })

        track = ContinuousTransformTrack(
            track_id=track_id,
            target_obj=layer,
            property_name=prop,
            rate=rate,
            target_value=target,
            duration_ms=duration_ms,
            getter=getter,
            setter=setter,
            on_done=on_done,
        )
        self.add_track(track)

        if self._dispatcher:
            self._dispatcher.fire_event("event.transform_started", {
                "id": track_id, "property": prop,
            })

        return {"status": "ok", "id": track_id}

    # ── Dynamic Layer Instant Transform ──────────────────────────────

    def _cmd_dynamic_instant(self, data: dict) -> dict:
        layer_id = data.get("layer_id")
        if not layer_id:
            raise ValueError("layer_id required")

        if not self._dispatcher:
            raise ValueError("Dispatcher not available")

        layer = self._dispatcher._layer_manager.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Layer not found: {layer_id}")

        if "position" in data:
            layer._anim_position = data["position"]
        if "rotation_z" in data:
            layer._anim_rotation_z = data["rotation_z"]
        if "scale" in data:
            layer._anim_scale = data["scale"]
        if "color" in data:
            layer.color = list(data["color"]) + [1.0]
            meta = self._dispatcher._dynamic_layers.get(layer_id)
            if meta:
                meta["color"] = list(data["color"])
        if "opacity" in data:
            layer.opacity = data["opacity"]
            layer.evict_byte_caches()

        self._viewport.update()
        return {"status": "ok"}

    # ── Stop Commands ────────────────────────────────────────────────

    def _cmd_stop(self, data: dict) -> dict:
        track_id = data.get("track_id") or data.get("id")
        if not track_id:
            raise ValueError("id required")
        removed = self.remove_track(track_id)
        if self._dispatcher:
            self._dispatcher.fire_event("event.transform_stopped", {
                "id": track_id, "reason": "manual",
            })
        return {"status": "ok", "removed": removed}

    def _cmd_stop_all(self) -> dict:
        ids = [t.id for t in self._tracks]
        self.remove_all()
        if self._dispatcher:
            for tid in ids:
                self._dispatcher.fire_event("event.transform_stopped", {
                    "id": tid, "reason": "manual",
                })
        return {"status": "ok", "removed": len(ids)}

    # ── Helpers ──────────────────────────────────────────────────────

    def _layer_prop_accessors(self, layer, layer_id, prop):
        """Build getter/setter callables for a dynamic layer property."""
        if prop == "opacity":
            def getter():
                return layer.opacity
            def setter(v):
                layer.opacity = v
                layer.evict_byte_caches()
            return getter, setter

        if prop == "color":
            def getter():
                return layer.color[:3] if layer.color else [0.5, 0.5, 0.5]
            def setter(v):
                layer.color = list(v) + [1.0]
                meta = self._dispatcher._dynamic_layers.get(layer_id)
                if meta:
                    meta["color"] = list(v)
            return getter, setter

        if prop.startswith("color_"):
            channel = {"color_r": 0, "color_g": 1, "color_b": 2}.get(prop)
            if channel is not None:
                def getter():
                    return layer.color[channel] if layer.color else 0.5
                def setter(v):
                    if layer.color is None:
                        layer.color = [0.5, 0.5, 0.5, 1.0]
                    layer.color[channel] = v
                return getter, setter

        if prop == "rotation_z":
            def getter():
                return getattr(layer, "_anim_rotation_z", 0.0)
            def setter(v):
                layer._anim_rotation_z = v
            return getter, setter

        if prop.startswith("position_"):
            axis = {"position_x": 0, "position_y": 1, "position_z": 2}.get(prop)
            if axis is not None:
                def getter():
                    pos = getattr(layer, "_anim_position", [0, 0, 0])
                    return pos[axis]
                def setter(v):
                    pos = list(getattr(layer, "_anim_position", [0, 0, 0]))
                    pos[axis] = v
                    layer._anim_position = pos
                return getter, setter

        if prop.startswith("scale_"):
            axis = {"scale_x": 0, "scale_y": 1, "scale_z": 2}.get(prop)
            if axis is not None:
                def getter():
                    scale = getattr(layer, "_anim_scale", [1, 1, 1])
                    return scale[axis]
                def setter(v):
                    scale = list(getattr(layer, "_anim_scale", [1, 1, 1]))
                    scale[axis] = v
                    layer._anim_scale = scale
                return getter, setter

        raise ValueError(f"Unknown dynamic layer property: {prop}")
