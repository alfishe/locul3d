"""Shapes & Annotations REST endpoints — editor bbox and plane overlays.

These operate on the **Editor's** annotation list. They are editor-specific
and don't create scene layers.  When the active window is the Viewer
(no annotation support), endpoints return HTTP 400.
"""

from __future__ import annotations

import logging

from aiohttp import web

from ..dispatcher import CommandDispatcher
from ..schemas import BBoxCreate, BBoxUpdate, PlaneCreate

log = logging.getLogger(__name__)


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    # Bounding boxes
    app.router.add_get("/api/v1/shapes/bboxes", _list_bboxes)
    app.router.add_post("/api/v1/shapes/bboxes", _create_bbox)
    app.router.add_put("/api/v1/shapes/bboxes/{index}", _update_bbox)
    app.router.add_delete("/api/v1/shapes/bboxes/{index}", _delete_bbox)
    # Planes
    app.router.add_get("/api/v1/shapes/planes", _list_planes)
    app.router.add_post("/api/v1/shapes/planes", _create_plane)
    app.router.add_delete("/api/v1/shapes/planes/{index}", _delete_plane)


# ── Bounding Boxes ───────────────────────────────────────────────────


async def _list_bboxes(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    try:
        result = await dispatcher.list_bboxes()
        return web.json_response(result)
    except AttributeError:
        return web.json_response(
            {"error": "Annotations not available (viewer mode)"}, status=400
        )


async def _create_bbox(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = BBoxCreate(**data)
    try:
        result = await dispatcher.create_bbox(req)
        return web.json_response(result)
    except AttributeError:
        return web.json_response(
            {"error": "Annotations not available (viewer mode)"}, status=400
        )


async def _update_bbox(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    index = int(request.match_info["index"])
    data = await request.json()
    update = BBoxUpdate(**data)
    try:
        result = await dispatcher.update_bbox(index, update)
        return web.json_response(result)
    except (AttributeError, IndexError) as exc:
        status = 400 if isinstance(exc, AttributeError) else 404
        return web.json_response({"error": str(exc)}, status=status)


async def _delete_bbox(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    index = int(request.match_info["index"])
    try:
        result = await dispatcher.delete_bbox(index)
        return web.json_response(result)
    except (AttributeError, IndexError) as exc:
        status = 400 if isinstance(exc, AttributeError) else 404
        return web.json_response({"error": str(exc)}, status=status)


# ── Planes ───────────────────────────────────────────────────────────


async def _list_planes(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    try:
        result = await dispatcher.list_planes()
        return web.json_response(result)
    except AttributeError:
        return web.json_response(
            {"error": "Annotations not available (viewer mode)"}, status=400
        )


async def _create_plane(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = PlaneCreate(**data)
    try:
        result = await dispatcher.create_plane(req)
        return web.json_response(result)
    except AttributeError:
        return web.json_response(
            {"error": "Annotations not available (viewer mode)"}, status=400
        )


async def _delete_plane(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    index = int(request.match_info["index"])
    try:
        result = await dispatcher.delete_plane(index)
        return web.json_response(result)
    except (AttributeError, IndexError) as exc:
        status = 400 if isinstance(exc, AttributeError) else 404
        return web.json_response({"error": str(exc)}, status=status)
