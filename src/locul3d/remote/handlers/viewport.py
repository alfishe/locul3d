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


async def _get_fade(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    bridge = dispatcher._bridge
    def _read():
        vp = dispatcher._viewport
        center = getattr(vp, "fade_aoi_center", None)
        return {
            "enable": bool(getattr(vp, "fade_enable", False)),
            "alpha_mul": float(getattr(vp, "fade_alpha_mul", 0.5)),
            "band": float(getattr(vp, "fade_band", 0.5)),
            "aoi_center": center.tolist() if center is not None else [0,0,0],
            "aoi_radius": float(getattr(vp, "fade_aoi_radius", 0.0)),
            "available": bool(
                getattr(vp, "_point_shader", None)
                and vp._point_shader.available
            ),
        }
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
        if "aoi_center" in data:
            vp.fade_aoi_center = _np.asarray(
                data["aoi_center"], dtype=_np.float64
            )
        if "aoi_radius" in data:
            vp.fade_aoi_radius = float(data["aoi_radius"])
        vp.update()
        return {
            "status": "ok",
            "enable": vp.fade_enable,
            "alpha_mul": vp.fade_alpha_mul,
            "band": vp.fade_band,
            "aoi_center": vp.fade_aoi_center.tolist(),
            "aoi_radius": vp.fade_aoi_radius,
            "available": bool(
                getattr(vp, "_point_shader", None)
                and vp._point_shader.available
            ),
        }
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
