"""Animation tracks — the individual units of animation.

Each track has a ``tick(now)`` method that returns True if the scene
needs a redraw, and a ``done`` flag that signals it should be removed.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .easing import EasingFn, resolve_easing
from .interpolation import interpolate_keyframes, lerp, lerp_angle, lerp_vec3


class AnimationTrack:
    """Base class for all animation tracks."""

    def __init__(self, track_id: str):
        self.id = track_id
        self.done = False
        self._start_time: Optional[float] = None

    def tick(self, now: float) -> bool:
        """Advance the track. Returns True if the scene is dirty."""
        raise NotImplementedError


# ── Camera Keyframe Track ────────────────────────────────────────────


class CameraTrack(AnimationTrack):
    """Keyframed camera animation with loop/ping_pong support."""

    PROPERTIES = ["azimuth", "elevation", "distance", "fov"]
    VEC3_PROPERTIES = {"target"}
    ANGLE_PROPERTIES = {"azimuth", "elevation"}
    ALL_PROPERTIES = PROPERTIES + ["target"]

    def __init__(
        self,
        track_id: str,
        viewport: Any,
        keyframes: List[dict],
        duration_ms: int,
        easing: EasingFn,
        loop: bool = False,
        ping_pong: bool = False,
        repeat_count: int = 0,
        on_done: Optional[Callable] = None,
    ):
        super().__init__(track_id)
        self._viewport = viewport
        self._keyframes = keyframes
        self._duration = duration_ms / 1000.0
        self._easing = easing
        self._loop = loop
        self._ping_pong = ping_pong
        self._repeat_count = repeat_count
        self._on_done = on_done
        self._cycle = 0
        self._forward = True

    def tick(self, now: float) -> bool:
        if self.done:
            return False
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        if self._duration <= 0:
            # Instant jump to last keyframe
            t = 1.0
            self.done = True
        else:
            raw_t = elapsed / self._duration

            if raw_t >= 1.0:
                if self._loop:
                    self._cycle += 1
                    if self._repeat_count > 0 and self._cycle >= self._repeat_count:
                        t = 1.0
                        self.done = True
                    else:
                        self._start_time = now
                        if self._ping_pong:
                            self._forward = not self._forward
                        raw_t = 0.0
                        t = self._easing(0.0) if self._forward else self._easing(1.0)
                else:
                    t = 1.0
                    self.done = True
            else:
                t = raw_t

            if not self.done:
                t = self._easing(t if self._forward else 1.0 - t)

        # Interpolate keyframes
        values = interpolate_keyframes(
            self._keyframes, t, self.ALL_PROPERTIES,
            angle_properties=self.ANGLE_PROPERTIES,
            vec3_properties=self.VEC3_PROPERTIES,
        )

        # Apply to viewport
        vp = self._viewport
        if "azimuth" in values:
            vp.cam_azimuth = values["azimuth"]
        if "elevation" in values:
            vp.cam_elevation = values["elevation"]
        if "distance" in values:
            vp.cam_distance = values["distance"]
        if "fov" in values:
            vp.cam_fov = values["fov"]
        if "target" in values:
            vp.cam_target = np.array(values["target"], dtype=np.float64)

        if self.done and self._on_done:
            self._on_done(self.id)

        return True


# ── Continuous Transform Track ───────────────────────────────────────


class ContinuousTransformTrack(AnimationTrack):
    """Continuous property change at a fixed rate (camera or dynamic layer)."""

    def __init__(
        self,
        track_id: str,
        target_obj: Any,
        property_name: str,
        rate: float = 0.0,
        target_value: Optional[List[float]] = None,
        duration_ms: int = 0,
        getter: Optional[Callable] = None,
        setter: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ):
        super().__init__(track_id)
        self._target_obj = target_obj
        self._property = property_name
        self._rate = rate
        self._target_value = target_value
        self._duration = duration_ms / 1000.0 if duration_ms > 0 else 0
        self._getter = getter
        self._setter = setter
        self._on_done = on_done
        self._last_time: Optional[float] = None
        self._initial_value = None

    def tick(self, now: float) -> bool:
        if self.done:
            return False
        if self._start_time is None:
            self._start_time = now
            self._last_time = now
            if self._target_value is not None and self._getter:
                self._initial_value = self._getter()
            return False

        dt = now - self._last_time
        self._last_time = now

        # Check duration expiry
        if self._duration > 0:
            elapsed = now - self._start_time
            if elapsed >= self._duration:
                self.done = True
                # Apply final value
                if self._target_value is not None and self._setter:
                    self._setter(self._target_value)
                if self._on_done:
                    self._on_done(self.id, "expired")
                return True

        # Apply rate-based or target-based change
        if self._target_value is not None and self._initial_value is not None:
            # Interpolate toward target over duration
            elapsed = now - self._start_time
            t = elapsed / self._duration if self._duration > 0 else 1.0
            t = min(t, 1.0)
            if isinstance(self._initial_value, (list, tuple)):
                val = lerp_vec3(list(self._initial_value), list(self._target_value), t)
            else:
                val = lerp(self._initial_value, self._target_value, t)
            if self._setter:
                self._setter(val)
        elif self._rate != 0 and self._setter and self._getter:
            current = self._getter()
            if isinstance(current, (int, float)):
                self._setter(current + self._rate * dt)

        return True


# ── Dynamic Layer Keyframe Track ─────────────────────────────────────


class DynamicLayerTrack(AnimationTrack):
    """Keyframed animation for a dynamic layer's transform properties."""

    SCALAR_PROPERTIES = ["rotation_z", "opacity", "point_size"]
    VEC3_PROPERTIES = {"position", "scale", "color"}
    ANGLE_PROPERTIES = {"rotation_z"}
    ALL_PROPERTIES = ["position", "rotation_z", "scale", "color", "opacity", "point_size"]

    def __init__(
        self,
        track_id: str,
        layer_id: str,
        dispatcher: Any,
        keyframes: List[dict],
        duration_ms: int,
        easing: EasingFn,
        loop: bool = False,
        ping_pong: bool = False,
        repeat_count: int = 0,
        on_done: Optional[Callable] = None,
    ):
        super().__init__(track_id)
        self._layer_id = layer_id
        self._dispatcher = dispatcher
        self._keyframes = keyframes
        self._duration = duration_ms / 1000.0
        self._easing = easing
        self._loop = loop
        self._ping_pong = ping_pong
        self._repeat_count = repeat_count
        self._on_done = on_done
        self._cycle = 0
        self._forward = True

    def tick(self, now: float) -> bool:
        if self.done:
            return False
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        if self._duration <= 0:
            t = 1.0
            self.done = True
        else:
            raw_t = elapsed / self._duration

            if raw_t >= 1.0:
                if self._loop:
                    self._cycle += 1
                    if self._repeat_count > 0 and self._cycle >= self._repeat_count:
                        t = 1.0
                        self.done = True
                    else:
                        self._start_time = now
                        if self._ping_pong:
                            self._forward = not self._forward
                        raw_t = 0.0
                        t = self._easing(0.0) if self._forward else self._easing(1.0)
                else:
                    t = 1.0
                    self.done = True
            else:
                t = raw_t

            if not self.done:
                t = self._easing(t if self._forward else 1.0 - t)

        # Interpolate
        values = interpolate_keyframes(
            self._keyframes, t, self.ALL_PROPERTIES,
            angle_properties=self.ANGLE_PROPERTIES,
            vec3_properties=self.VEC3_PROPERTIES,
        )

        # Apply to layer via dispatcher
        self._apply_to_layer(values)

        if self.done and self._on_done:
            self._on_done(self.id)

        return True

    def _apply_to_layer(self, values: dict) -> None:
        """Apply interpolated values to the dynamic layer.

        Uses GL-level transform uniforms where possible (no VBO rebuild).
        Falls back to property patches for opacity/color.
        """
        layer = self._dispatcher._layer_manager.get_layer(self._layer_id)
        if layer is None:
            self.done = True
            return

        # Store transform state on layer for GL-level rendering
        if "position" in values:
            layer._anim_position = values["position"]
        if "rotation_z" in values:
            layer._anim_rotation_z = values["rotation_z"]
        if "scale" in values:
            layer._anim_scale = values["scale"]

        # Direct property updates (no VBO rebuild needed)
        if "opacity" in values:
            layer.opacity = values["opacity"]
            layer.evict_byte_caches()
        if "color" in values:
            layer.color = list(values["color"]) + [1.0]
            meta = self._dispatcher._dynamic_layers.get(self._layer_id)
            if meta:
                meta["color"] = list(values["color"])
