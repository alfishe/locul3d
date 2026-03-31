"""Dynamic geometry layer REST endpoints — create, read, update, patch, delete.

Dynamic layers are first-class scene layers created via the API.
They get ``layer_id = f"dyn_{name}"``.  Duplicate names return HTTP 409.
"""

from __future__ import annotations

import logging

import numpy as np
from aiohttp import web

from ..dispatcher import CommandDispatcher
from ..schemas import DynamicLayerCreate, DynamicLayerInfo, DynamicLayerPatch

log = logging.getLogger(__name__)


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    app.router.add_get("/api/v1/scene/dynamic", _list_dynamic)
    app.router.add_post("/api/v1/scene/dynamic", _create_dynamic)
    app.router.add_get("/api/v1/scene/dynamic/{layer_id}", _get_dynamic)
    app.router.add_put("/api/v1/scene/dynamic/{layer_id}", _update_dynamic)
    app.router.add_patch("/api/v1/scene/dynamic/{layer_id}", _patch_dynamic)
    app.router.add_delete("/api/v1/scene/dynamic/{layer_id}", _delete_dynamic)
    app.router.add_delete("/api/v1/scene/dynamic", _clear_dynamic)


async def _list_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    result = await dispatcher.list_dynamic_layers()
    return web.json_response(result)


async def _create_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = DynamicLayerCreate(**data)
    try:
        result = await dispatcher.create_dynamic_layer(req)
        return web.json_response(result)
    except ValueError as exc:
        if "duplicate" in str(exc).lower() or "already exists" in str(exc).lower():
            return web.json_response(
                {"error": str(exc)}, status=409
            )
        raise


async def _get_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layer_id = request.match_info["layer_id"]
    try:
        result = await dispatcher.get_dynamic_layer(layer_id)
        return web.json_response(result)
    except ValueError:
        return web.json_response({"error": "Not found"}, status=404)


async def _update_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layer_id = request.match_info["layer_id"]
    data = await request.json()
    try:
        result = await dispatcher.update_dynamic_layer(layer_id, data)
        return web.json_response(result)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def _patch_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layer_id = request.match_info["layer_id"]
    data = await request.json()
    patch = DynamicLayerPatch(**data)
    try:
        result = await dispatcher.patch_dynamic_layer(layer_id, patch)
        return web.json_response(result)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def _delete_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    layer_id = request.match_info["layer_id"]
    try:
        result = await dispatcher.delete_dynamic_layer(layer_id)
        return web.json_response(result)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def _clear_dynamic(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    result = await dispatcher.clear_dynamic_layers()
    return web.json_response(result)
