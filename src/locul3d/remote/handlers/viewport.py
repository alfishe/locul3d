"""Viewport REST endpoints — settings, correction, clip, render mode."""

from __future__ import annotations

from aiohttp import web

from ..dispatcher import CommandDispatcher
from ..schemas import ClipState, CorrectionState, RenderModeUpdate, ViewportSettings


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    # Settings
    app.router.add_get("/api/v1/viewport", _get_settings)
    app.router.add_put("/api/v1/viewport", _set_settings)
    # Correction
    app.router.add_get("/api/v1/viewport/correction", _get_correction)
    app.router.add_put("/api/v1/viewport/correction", _set_correction)
    # Clip
    app.router.add_get("/api/v1/viewport/clip", _get_clip)
    app.router.add_put("/api/v1/viewport/clip", _set_clip)
    app.router.add_delete("/api/v1/viewport/clip", _clear_clip)
    # Render mode
    app.router.add_get("/api/v1/viewport/render_mode", _get_render_mode)
    app.router.add_put("/api/v1/viewport/render_mode", _set_render_mode)
    # Point fade (shader)
    app.router.add_get("/api/v1/viewport/fade", _get_fade)
    app.router.add_put("/api/v1/viewport/fade", _set_fade)


async def _get_settings(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await dispatcher.get_viewport_settings()
    return web.json_response(data)


async def _set_settings(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    settings = ViewportSettings(**data)
    result = await dispatcher.set_viewport_settings(settings)
    return web.json_response(result)


async def _get_correction(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await dispatcher.get_correction()
    return web.json_response(data)


async def _set_correction(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    state = CorrectionState(**data)
    result = await dispatcher.set_correction(state)
    return web.json_response(result)


async def _get_clip(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await dispatcher.get_clip()
    return web.json_response(data)


async def _set_clip(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    state = ClipState(**data)
    result = await dispatcher.set_clip(state)
    return web.json_response(result)


async def _clear_clip(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    result = await dispatcher.clear_clip()
    return web.json_response(result)


async def _get_render_mode(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    bridge = dispatcher._bridge
    result = await bridge.invoke_on_qt(dispatcher._get_render_mode)
    return web.json_response(result)


def _fade_state(vp) -> dict:
    import numpy as _np
    bb_min = getattr(vp, "fade_aoi_min", _np.zeros(3))
    bb_max = getattr(vp, "fade_aoi_max", _np.zeros(3))
    hull = getattr(vp, "_fade_debug_hull_ndc", None) or []
    corners = getattr(vp, "_fade_debug_corners_ndc", None) or []
    return {
        "enable": bool(getattr(vp, "fade_enable", False)),
        "alpha_mul": float(getattr(vp, "fade_alpha_mul", 0.5)),
        "band": float(getattr(vp, "fade_band", 0.02)),
        "expansion": float(getattr(vp, "fade_expansion", 1.0)),
        "aoi_min": bb_min.tolist(),
        "aoi_max": bb_max.tolist(),
        "debug_overlay": bool(getattr(vp, "fade_debug_overlay", False)),
        "discard_culled": bool(getattr(vp, "fade_discard_culled", False)),
        "last_hull_ndc": [[float(x), float(y)] for (x, y) in hull],
        "last_corners_ndc": [
            [None if (isinstance(x, float) and x != x) else float(x),
             None if (isinstance(y, float) and y != y) else float(y)]
            for (x, y) in corners
        ],
        "available": bool(
            getattr(vp, "_point_shader", None)
            and vp._point_shader.available
        ),
    }


async def _get_fade(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    bridge = dispatcher._bridge
    def _read():
        return _fade_state(dispatcher._viewport)
    result = await bridge.invoke_on_qt(_read)
    return web.json_response(result)


async def _set_fade(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    bridge = dispatcher._bridge
    def _apply():
        import numpy as _np
        vp = dispatcher._viewport
        if "enable" in data:
            vp.fade_enable = bool(data["enable"])
        if "alpha_mul" in data:
            vp.fade_alpha_mul = float(data["alpha_mul"])
        if "band" in data:
            vp.fade_band = float(data["band"])
        if "expansion" in data:
            vp.fade_expansion = float(data["expansion"])
        if "debug_overlay" in data:
            vp.fade_debug_overlay = bool(data["debug_overlay"])
        if "discard_culled" in data:
            vp.fade_discard_culled = bool(data["discard_culled"])


        # Two ways to specify the AoI box:
        #   1) explicit min + max  (preferred)
        #   2) center + size       (auto-derived bbox)
        #   3) center + radius     (legacy: cube of side 2*radius — back-compat)
        if "aoi_min" in data and "aoi_max" in data:
            vp.fade_aoi_min = _np.asarray(data["aoi_min"], dtype=_np.float64)
            vp.fade_aoi_max = _np.asarray(data["aoi_max"], dtype=_np.float64)
        elif "aoi_center" in data and "aoi_size" in data:
            c = _np.asarray(data["aoi_center"], dtype=_np.float64)
            s = _np.asarray(data["aoi_size"], dtype=_np.float64)
            vp.fade_aoi_min = c - s * 0.5
            vp.fade_aoi_max = c + s * 0.5
        elif "aoi_center" in data and "aoi_radius" in data:
            c = _np.asarray(data["aoi_center"], dtype=_np.float64)
            r = float(data["aoi_radius"])
            vp.fade_aoi_min = c - r
            vp.fade_aoi_max = c + r
        elif "aoi_center" in data:
            # center alone — keep current extent, just shift it
            c = _np.asarray(data["aoi_center"], dtype=_np.float64)
            half = (vp.fade_aoi_max - vp.fade_aoi_min) * 0.5
            vp.fade_aoi_min = c - half
            vp.fade_aoi_max = c + half

        # Synchronous repaint so the new fade settings are visible
        # immediately. update() schedules a paint event for "later"
        # which is fine inside an animation loop but feels broken
        # when the user PUTs an alpha_mul change manually.
        try:
            vp.repaint()
        except Exception:
            vp.update()

        out = {"status": "ok"}
        out.update(_fade_state(vp))

        # Diagnostic hint — explain why a request might appear to do
        # nothing.  Common cases: fade_enable is False, or the AoI
        # bbox is degenerate (extent zero on any axis).
        hints = []
        if not vp.fade_enable:
            hints.append("fade is currently disabled — send "
                         "{\"enable\": true} to activate")
        extent = vp.fade_aoi_max - vp.fade_aoi_min
        if not bool((extent > 0.0).all()):
            hints.append("AoI bbox is degenerate (extent has a zero "
                         "axis) — send aoi_min + aoi_max to set it")
        if hints:
            out["hints"] = hints
        return out
    result = await bridge.invoke_on_qt(_apply)
    return web.json_response(result)


async def _set_render_mode(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    update = RenderModeUpdate(**data)
    bridge = dispatcher._bridge
    result = await bridge.invoke_on_qt(
        lambda: dispatcher._set_render_mode(update.mode, update.width, update.height)
    )
    return web.json_response(result)
