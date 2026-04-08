"""Animation engine — QTimer-driven tick loop managing all active tracks.

Runs entirely on the Qt main thread.  The server sends animation
*declarations*; the engine ticks them locally at 500 Hz for smooth,
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

    The timer runs at ~500 Hz (2 ms interval) using PreciseTimer.
    Animation math uses ``time.perf_counter()`` deltas, so animations
    stay time-accurate even if frames drop.

    The high tick rate is intentional: it gives the realtime FPS gate
    fine-grained quantization. At 8 ms ticks the gate could only
    enable rendering on multiples of 8 ms, which forces effective FPS
    to snap to {125, 62, 42, 31, 25, …} — a too-coarse grid that hides
    the adaptive controller's work. At 2 ms the snap grid is dense
    enough to look continuous (~{500, 250, 167, 125, 100, 83, 71,
    62, 55, 50, 45, 41, 38, …}).
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
        self._timer.setInterval(2)  # 500 Hz — fine-grained gate quantization
        self._frame_number = 0
        self._render_mode = "realtime"

        # Realtime FPS control.  Track ticks still run at 500 Hz so the
        # animation math stays time-accurate, but viewport repaints
        # are gated.  Capture mode ignores the cap.
        #
        #   _realtime_max_fps:        user-requested CEILING
        #   _realtime_min_fps:        floor we never throttle below
        #   _realtime_effective_fps:  actual gate, adapted each tick
        #                             based on viewport._paint_ema_ms
        self._realtime_max_fps: float = 125.0
        self._realtime_min_fps: float = 1.0    # honest minimum, not a lie
        self._realtime_effective_fps: float = 125.0
        self._adaptive_enabled: bool = True
        self._fps_snap: float = 5.0   # snap effective FPS to multiples of 5
        # Tiny safety margin (ms) added to the paint period before
        # converting to FPS. Keeps the gate from firing exactly when
        # the next paint will start, leaving one event-loop pass for
        # Qt input/timer events. With vsync off and an 8-frame p80
        # signal, 2 ms is plenty under all loads.
        self._render_cooldown_ms: float = 2.0
        self._last_render_request_t: float = 0.0

        # Virtual animation clock — decoupled from wall time so we can
        # SLOW DOWN the animation's progression when the renderer
        # can't keep up.  At 60 FPS the clock advances 1:1 with wall
        # time; at 5 FPS it advances 5/60 = 0.083× wall time, so each
        # rendered frame represents 1/60 s of animation regardless of
        # how long the GPU spent painting it.  Net effect: a 15-second
        # camera flyover plays out over 180 wall seconds at 5 FPS,
        # but every frame shows the next "60 FPS step" of the path —
        # so the picture is no longer jumping by 12 frames at a time.
        self._anim_time_scale_auto: bool = True
        self._anim_time_scale_fixed: float = 1.0
        self._anim_nominal_fps: float = 60.0
        self._virtual_clock: float = 0.0
        self._real_t_prev: Optional[float] = None

        # Preview mode: hold target FPS, adapt LOD instead.
        # Mutually exclusive with the full-res adaptive path above.
        self._preview_mode: bool = False
        self._preview_target_fps: float = 60.0
        self._preview_budget_pts: int = 25_000_000
        self._preview_min_pts: int = 250_000
        self._preview_max_pts: int = 250_000_000

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
        """Called by QTimer at ~500 Hz. Ticks all active tracks."""
        real_now = time.perf_counter()

        # Advance the virtual animation clock by `dt_real * scale`.
        # Tracks see this monotonic-ish clock instead of perf_counter
        # so the controller can slow them down when paint is heavy
        # without breaking any of their delta math.
        if self._real_t_prev is None:
            self._real_t_prev = real_now
        real_dt = real_now - self._real_t_prev
        self._real_t_prev = real_now

        if self._anim_time_scale_auto:
            scale = min(
                1.0,
                self._realtime_effective_fps
                / max(self._anim_nominal_fps, 1.0),
            )
        else:
            scale = max(0.0, self._anim_time_scale_fixed)
        self._virtual_clock += real_dt * scale
        virtual_now = self._virtual_clock

        dirty = False
        for track in self._tracks:
            if track.tick(virtual_now):
                dirty = True

        # Remove completed tracks
        self._tracks = [t for t in self._tracks if not t.done]

        # Stop timer if no more tracks; restore default LOD behavior
        # (mouse-drag still gets stride decimation, idle gets static).
        if not self._tracks:
            self._timer.stop()
            self._last_render_request_t = 0.0
            self._virtual_clock = 0.0
            self._real_t_prev = None
            try:
                self._viewport._interacting = False
                self._viewport._force_full_res = False
                self._viewport._preview_mode = False
                self._viewport.update()
            except Exception:
                pass
            return

        if not dirty:
            return

        if self._render_mode == "capture":
            # Capture: every dirty tick = one rendered frame, full quality.
            self._viewport._interacting = False
            self._viewport._force_full_res = True
            self._viewport.update()
            self._viewport.repaint()
            self.frame_ready.emit(self._frame_number)
            self._frame_number += 1
            return

        if self._preview_mode:
            # Preview mode: hold target FPS, adapt LOD.
            self._viewport._interacting = False
            self._viewport._force_full_res = False
            self._viewport._preview_mode = True
            self._viewport._preview_budget_pts = self._preview_budget_pts
            self._adapt_preview_budget()
            self._realtime_effective_fps = self._preview_target_fps
        else:
            # Full-res adaptive: render every layer at FULL resolution.
            # We deliberately do NOT set _interacting (which would
            # activate stride LOD) — the goal is that every frame the
            # controller times is a frame we'd be happy to record.
            # Adaptive FPS does the throttling instead, so 1B-point
            # scenes drop to a few FPS but never lose detail.
            self._viewport._interacting = False
            self._viewport._force_full_res = True
            self._viewport._preview_mode = False
            if self._adaptive_enabled:
                self._adapt_effective_fps()
            else:
                self._realtime_effective_fps = self._realtime_max_fps

        # Gate.  Two independent constraints; BOTH must be satisfied:
        #
        #   A) target_period since the LAST RENDER REQUEST.
        #      This makes the visible cadence equal exactly the
        #      effective_fps that the controller chose. Paint time is
        #      *overlapped* with the wait, not added to it — so when
        #      the controller picks 2 FPS the user sees frames every
        #      500 ms regardless of whether each frame takes 100 or
        #      400 ms to paint.
        #
        #   B) cooldown_ms since the LAST PAINT END.
        #      Tiny floor (~2 ms) so we never queue an update() the
        #      same instant paintGL returns; that would deny the Qt
        #      event loop one pass and stutter input handling.
        #
        # Plus the _paint_in_progress guard: don't queue while Qt is
        # still inside paintGL on the main thread, or it'd coalesce
        # and burst the moment paint returns.
        vp = self._viewport
        if getattr(vp, "_paint_in_progress", False):
            return

        target_period_ms = 1000.0 / max(self._realtime_effective_fps, 1.0)
        since_request_ms = (real_now - self._last_render_request_t) * 1000.0
        last_paint_end_t = float(getattr(vp, "_last_paint_end_t", 0.0))
        since_paint_end_ms = (real_now - last_paint_end_t) * 1000.0

        if (since_request_ms >= target_period_ms
                and since_paint_end_ms >= self._render_cooldown_ms):
            self._last_render_request_t = real_now
            vp.update()

    def _adapt_effective_fps(self) -> None:
        """Closed-form FPS controller (single regime).

        Signal: ``viewport._paint_peak_ms`` — a slow-decaying peak of
        recent paint durations.  Jumps up instantly on spikes, decays
        ~3% per frame.  Compared to a percentile or EMA this is the
        only signal that stays *stable when paint cost is volatile*,
        which is exactly what we need to stop eff_fps from bouncing.

        Period budget:
          period = peak * safety_mul + cooldown_ms

        ``safety_mul = 1.25`` reserves ~20% timing headroom so the
        GPU isn't pinned at 100% — that headroom is what makes the
        difference between "rotating slowly but smoothly" and "GPU
        constantly overloaded, OS choppy". The cooldown is a tiny
        absolute floor (2 ms) so the gate yields one event-loop
        pass between cheap consecutive frames.
        """
        ceil_fps = self._realtime_max_fps
        floor_fps = self._realtime_min_fps
        vp = self._viewport
        # Slow-decaying peak is the primary signal; chain back through
        # p80 → EMA → ceiling for the bootstrap window.
        paint_ms = float(getattr(vp, "_paint_peak_ms", 0.0))
        if paint_ms <= 0.0:
            paint_ms = float(getattr(vp, "_paint_p80_ms", 0.0))
        if paint_ms <= 0.0:
            paint_ms = float(getattr(vp, "_paint_ema_ms", 0.0))
        if paint_ms <= 0.0:
            self._realtime_effective_fps = ceil_fps
            return

        SAFETY_MUL = 1.25
        period_ms = paint_ms * SAFETY_MUL + self._render_cooldown_ms
        achievable = 1000.0 / max(period_ms, 1.0)
        target = min(ceil_fps, max(floor_fps, achievable))

        # Snap to the coarse grid (default 5 FPS).
        snap = max(self._fps_snap, 1.0)
        if target >= snap:
            snapped = round(target / snap) * snap
        else:
            # Below the snap grid: keep 1-FPS resolution, never zero.
            snapped = max(1.0, round(target))
        snapped = max(floor_fps, min(ceil_fps, snapped))

        # Asymmetric hysteresis: drops are instant (react to spikes
        # immediately so we don't oversaturate the GPU), but recovery
        # has to clear at least one full snap step to avoid the
        # oscillating "5-10 FPS jumping" symptom on volatile loads.
        cur = self._realtime_effective_fps
        if snapped < cur:
            self._realtime_effective_fps = snapped
        elif snapped >= cur + snap:
            self._realtime_effective_fps = snapped
        # else: keep current — the proposed bump is smaller than one
        # grid step, not worth changing

    def _adapt_preview_budget(self) -> None:
        """Hold target FPS by tuning the global vertex budget.

        Inverse of ``_adapt_effective_fps``: paint time is the signal,
        but here it modulates the per-frame point budget instead of
        the render rate. Asymmetric for the same reason — back off
        fast when overloaded, recover slowly so we sit just below the
        cliff instead of bouncing across it.
        """
        target_dt_ms = 1000.0 / max(self._preview_target_fps, 1.0)
        paint_ms = float(getattr(self._viewport, "_paint_ema_ms", 0.0))

        if paint_ms <= 0.0:
            return

        budget = self._preview_budget_pts

        if paint_ms > target_dt_ms * 1.15:
            # Over-budget — slash the vertex budget.
            new_budget = int(budget * 0.7)
        elif paint_ms < target_dt_ms * 0.55:
            # Lots of headroom — grow gently.
            new_budget = int(budget * 1.10) + 50_000
        else:
            return

        new_budget = max(self._preview_min_pts,
                         min(self._preview_max_pts, new_budget))
        # Snap to ~250k steps so the displayed budget is interpretable.
        step = 250_000
        new_budget = (new_budget // step) * step
        new_budget = max(self._preview_min_pts, new_budget)
        self._preview_budget_pts = new_budget

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
        if msg_type == "animation.set_realtime_fps":
            if "fps" in data:
                self._realtime_max_fps = max(1.0, float(data["fps"]))
                # Snap effective rate up to the new ceiling so the
                # controller starts from optimism, not yesterday's value.
                self._realtime_effective_fps = self._realtime_max_fps
            if "min_fps" in data:
                self._realtime_min_fps = max(1.0, float(data["min_fps"]))
            if "adaptive" in data:
                self._adaptive_enabled = bool(data["adaptive"])
            return {
                "status": "ok",
                "max_fps": self._realtime_max_fps,
                "min_fps": self._realtime_min_fps,
                "adaptive": self._adaptive_enabled,
                "effective_fps": self._realtime_effective_fps,
            }
        if msg_type == "animation.get_realtime_fps":
            return {
                "max_fps": self._realtime_max_fps,
                "min_fps": self._realtime_min_fps,
                "adaptive": self._adaptive_enabled,
                "effective_fps": self._realtime_effective_fps,
                "preview_mode": self._preview_mode,
                "preview_target_fps": self._preview_target_fps,
                "preview_budget_pts": self._preview_budget_pts,
                "paint_ema_ms": float(
                    getattr(self._viewport, "_paint_ema_ms", 0.0)
                ),
                "paint_p80_ms": float(
                    getattr(self._viewport, "_paint_p80_ms", 0.0)
                ),
                "paint_peak_ms": float(
                    getattr(self._viewport, "_paint_peak_ms", 0.0)
                ),
                "paint_last_ms": float(
                    getattr(self._viewport, "_last_paint_ms", 0.0)
                ),
                "paint_last_cpu_ms": float(
                    getattr(self._viewport, "_last_paint_cpu_ms", 0.0)
                ),
                "time_scale_auto": self._anim_time_scale_auto,
                "time_scale_fixed": self._anim_time_scale_fixed,
                "time_scale_nominal_fps": self._anim_nominal_fps,
                "time_scale_active": (
                    min(1.0,
                        self._realtime_effective_fps
                        / max(self._anim_nominal_fps, 1.0))
                    if self._anim_time_scale_auto
                    else self._anim_time_scale_fixed
                ),
            }
        if msg_type == "animation.set_time_scale":
            if "auto" in data:
                self._anim_time_scale_auto = bool(data["auto"])
            if "scale" in data:
                self._anim_time_scale_fixed = max(0.0, float(data["scale"]))
            if "nominal_fps" in data:
                self._anim_nominal_fps = max(1.0, float(data["nominal_fps"]))
            return {
                "status": "ok",
                "auto": self._anim_time_scale_auto,
                "scale": self._anim_time_scale_fixed,
                "nominal_fps": self._anim_nominal_fps,
            }
        if msg_type == "animation.set_preview_mode":
            if "enable" in data:
                self._preview_mode = bool(data["enable"])
            if "target_fps" in data:
                self._preview_target_fps = max(1.0, float(data["target_fps"]))
            if "min_pts" in data:
                self._preview_min_pts = max(1000, int(data["min_pts"]))
            if "max_pts" in data:
                self._preview_max_pts = max(self._preview_min_pts,
                                            int(data["max_pts"]))
            return {
                "status": "ok",
                "preview_mode": self._preview_mode,
                "target_fps": self._preview_target_fps,
                "budget_pts": self._preview_budget_pts,
                "min_pts": self._preview_min_pts,
                "max_pts": self._preview_max_pts,
            }

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
