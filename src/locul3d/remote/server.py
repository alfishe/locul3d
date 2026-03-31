"""aiohttp server — REST routes + WebSocket handler.

Runs on a background daemon thread.  All viewer mutations are dispatched
through :class:`CommandDispatcher` → :class:`QtBridge` to the Qt main thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Optional

from aiohttp import web

from .bridge import QtBridge
from .dispatcher import CommandDispatcher

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

log = logging.getLogger(__name__)


class RemoteServer:
    """Manages the aiohttp server lifecycle on a background thread."""

    def __init__(self, window: "QMainWindow", host: str, port: int) -> None:
        self._window = window
        self._host = host
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None

        # Bridge lives on the Qt main thread (created here, moved by Qt)
        self._bridge = QtBridge()
        self._dispatcher = CommandDispatcher(window, self._bridge, port)

        # Try to attach animation engine
        try:
            from locul3d.animation import create_engine

            engine = create_engine(self._dispatcher._viewport)
            self._dispatcher.set_animation_engine(engine)
        except ImportError:
            log.debug("Animation package not available — animation commands disabled")

    def start(self) -> None:
        """Start the server on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run_server,
            name="locul3d-remote-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._loop and self._runner:
            asyncio.run_coroutine_threadsafe(
                self._shutdown(), self._loop
            ).result(timeout=3.0)
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── Internal ──────────────────────────────────────────────────────

    def _run_server(self) -> None:
        """Thread target: create and run the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            log.exception("Remote API server crashed")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._dispatcher.set_event_loop(asyncio.get_event_loop())
        app = self._create_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        # Block until told to stop
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def _shutdown(self) -> None:
        """Called from the main thread to trigger graceful shutdown."""
        if self._runner:
            await self._runner.cleanup()
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def _create_app(self) -> web.Application:
        """Build the aiohttp Application with all routes."""
        app = web.Application()
        app["dispatcher"] = self._dispatcher

        # ── REST routes ───────────────────────────────────────────────

        from .handlers import system, camera, scene, viewport, dynamic, shapes

        system.setup_routes(app, self._dispatcher)
        camera.setup_routes(app, self._dispatcher)
        scene.setup_routes(app, self._dispatcher)
        viewport.setup_routes(app, self._dispatcher)
        dynamic.setup_routes(app, self._dispatcher)
        shapes.setup_routes(app, self._dispatcher)

        # ── WebSocket ─────────────────────────────────────────────────

        app.router.add_get("/ws", self._ws_handler)

        return app

    # ── WebSocket handler ─────────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=100 * 1024 * 1024)  # 100 MB
        await ws.prepare(request)

        dispatcher: CommandDispatcher = request.app["dispatcher"]
        dispatcher.register_ws(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({
                            "type": "error",
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON",
                        })
                        continue

                    msg_type = data.pop("type", "")
                    msg_id = data.pop("id", None)

                    try:
                        result = await dispatcher.handle_ws_command(msg_type, data)
                        response = {
                            "type": "result",
                            "status": "ok",
                            "data": result,
                        }
                        if msg_id is not None:
                            response["id"] = msg_id
                        await ws.send_json(response)
                    except ValueError as exc:
                        response = {
                            "type": "error",
                            "code": "INVALID_PARAM",
                            "message": str(exc),
                        }
                        if msg_id is not None:
                            response["id"] = msg_id
                        await ws.send_json(response)
                    except Exception as exc:
                        log.exception("WS command %s failed", msg_type)
                        response = {
                            "type": "error",
                            "code": "INTERNAL_ERROR",
                            "message": str(exc),
                        }
                        if msg_id is not None:
                            response["id"] = msg_id
                        await ws.send_json(response)

                elif msg.type == web.WSMsgType.BINARY:
                    try:
                        result = await dispatcher.handle_ws_binary(msg.data)
                        if result:
                            await ws.send_json({
                                "type": "result",
                                "status": "ok",
                                "data": result,
                            })
                    except Exception as exc:
                        log.exception("Binary WS message failed")
                        await ws.send_json({
                            "type": "error",
                            "code": "BINARY_ERROR",
                            "message": str(exc),
                        })

        finally:
            dispatcher.unregister_ws(ws)

        return ws
