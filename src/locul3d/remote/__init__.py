"""Locul3D Remote Control — optional HTTP REST + WebSocket API server.

This package is fully detachable: the viewer/editor work identically
without it.  Activate via ``start_server()`` or the ``--api-port`` CLI flag.

Usage::

    from locul3d.remote import start_server, stop_server

    server = start_server(window=viewer_window, port=8350)
    # ... later ...
    stop_server(server)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

log = logging.getLogger(__name__)

_server_handle: Optional[object] = None


def start_server(
    window: "QMainWindow",
    port: int = 8350,
    host: str = "127.0.0.1",
) -> object:
    """Start the aiohttp REST + WS server on a background daemon thread.

    Args:
        window: The active ViewerWindow or EditorWindow instance.
        port:   TCP port to bind (default 8350).
        host:   Interface to bind (default localhost-only).

    Returns:
        An opaque server handle that can be passed to ``stop_server()``.
    """
    global _server_handle
    from .server import RemoteServer

    srv = RemoteServer(window=window, host=host, port=port)
    srv.start()
    _server_handle = srv
    log.info("Remote API listening on http://%s:%d", host, port)
    return srv


def stop_server(handle: object = None) -> None:
    """Gracefully shut down the remote API server.

    Args:
        handle: Server handle returned by ``start_server()``.
                If *None*, stops the most-recently-started server.
    """
    global _server_handle
    srv = handle or _server_handle
    if srv is not None:
        from .server import RemoteServer

        assert isinstance(srv, RemoteServer)
        srv.stop()
        _server_handle = None
        log.info("Remote API server stopped")
