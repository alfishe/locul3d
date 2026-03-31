"""System endpoints — ping, status, screenshot, openapi."""

from __future__ import annotations

from aiohttp import web

from ..dispatcher import CommandDispatcher


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    app.router.add_get("/api/v1/system/ping", _ping)
    app.router.add_get("/api/v1/system/status", _status)
    app.router.add_get("/api/v1/system/screenshot", _screenshot)


async def _ping(request: web.Request) -> web.Response:
    return web.json_response({"pong": True})


async def _status(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await dispatcher.get_status()
    return web.json_response(data)


async def _screenshot(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    png_bytes = await dispatcher.take_screenshot()
    return web.Response(body=png_bytes, content_type="image/png")
