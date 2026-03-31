"""Scene REST endpoints — layers, load, clear, bounds."""

from __future__ import annotations

from aiohttp import web

from ..dispatcher import CommandDispatcher
from ..schemas import LayerUpdate, SceneLoadRequest, FolderLoadRequest


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    app.router.add_get("/api/v1/scene/layers", _get_layers)
    app.router.add_put("/api/v1/scene/layers/{layer_id}", _update_layer)
    app.router.add_post("/api/v1/scene/load", _load)
    app.router.add_post("/api/v1/scene/load_folder", _load_folder)
    app.router.add_delete("/api/v1/scene/clear", _clear)
    app.router.add_get("/api/v1/scene/bounds", _bounds)


async def _get_layers(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layers = await dispatcher.get_layers()
    return web.json_response(layers)


async def _update_layer(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layer_id = request.match_info["layer_id"]
    data = await request.json()
    update = LayerUpdate(**data)
    result = await dispatcher.update_layer(layer_id, update)
    return web.json_response(result)


async def _load(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = SceneLoadRequest(**data)

    from ..bridge import QtBridge

    bridge = dispatcher._bridge

    def _do_load():
        window = dispatcher._window
        for path in req.paths:
            # Use the viewer's own load mechanism
            if hasattr(window, "_load_files"):
                window._load_files([path])
            elif hasattr(window, "load_files"):
                window.load_files([path])
        return {"status": "ok", "loaded": len(req.paths)}

    result = await bridge.invoke_on_qt(_do_load)
    return web.json_response(result)


async def _load_folder(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = FolderLoadRequest(**data)
    bridge = dispatcher._bridge

    def _do_load():
        window = dispatcher._window
        if hasattr(window, "_load_folder"):
            window._load_folder(req.path)
        elif hasattr(window, "load_folder"):
            window.load_folder(req.path)
        return {"status": "ok", "folder": req.path}

    result = await bridge.invoke_on_qt(_do_load)
    return web.json_response(result)


async def _clear(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    bridge = dispatcher._bridge

    def _do_clear():
        dispatcher._layer_manager.clear()
        dispatcher._viewport.update()
        return {"status": "ok"}

    result = await bridge.invoke_on_qt(_do_clear)
    return web.json_response(result)


async def _bounds(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    bridge = dispatcher._bridge

    def _get_bounds():
        center, radius = dispatcher._layer_manager.get_scene_bounds()
        return {
            "center": center.tolist(),
            "radius": float(radius),
            "x_min": float(center[0] - radius),
            "x_max": float(center[0] + radius),
            "y_min": float(center[1] - radius),
            "y_max": float(center[1] + radius),
            "z_min": float(center[2] - radius),
            "z_max": float(center[2] + radius),
        }

    result = await bridge.invoke_on_qt(_get_bounds)
    return web.json_response(result)
