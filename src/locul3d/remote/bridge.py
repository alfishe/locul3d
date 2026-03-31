"""Qt ↔ asyncio thread bridge.

Provides ``QtBridge`` — the single safe mechanism for crossing from the
aiohttp server thread to the Qt main thread.  All viewer/editor mutations
*must* go through ``invoke_on_qt()`` to avoid OpenGL context violations
and widget thread-safety issues.

Implementation uses a custom Signal carrying a Python callable + Future
pair, which PySide6 marshals correctly across threads via QueuedConnection.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

from PySide6.QtCore import QObject, Qt, Signal, Slot

log = logging.getLogger(__name__)
T = TypeVar("T")


class _BridgeReceiver(QObject):
    """Lives on the Qt main thread — receives and executes callables.

    The ``execute`` signal is connected with ``Qt.QueuedConnection``
    so that emission from the aiohttp thread queues the work into the
    Qt event loop's queue, where it runs on the main thread.
    """

    execute = Signal(object, object)  # (fn, future)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Connect to our own slot with QueuedConnection
        self.execute.connect(self._on_execute, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object)
    def _on_execute(self, fn: Callable[[], Any], future: Future) -> None:
        """Run *fn()* on the Qt main thread and resolve *future*."""
        try:
            result = fn()
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)


class QtBridge:
    """Bridge between asyncio (aiohttp thread) and Qt main thread.

    Call :meth:`invoke_on_qt` from an ``async`` context on the server
    thread.  It emits a signal that Qt dispatches to the main thread,
    executes the callable there, and returns the result via a
    :class:`concurrent.futures.Future` wrapped as an asyncio awaitable.

    Thread-safety guarantees
    ------------------------
    * **No concurrent GL access** — every viewport mutation runs on the
      Qt event loop, serialised by Qt's internal queue.
    * **No deadlocks** — ``QueuedConnection`` is fully async; the aiohttp
      thread never blocks the Qt thread.
    * **Error propagation** — exceptions in Qt-side code propagate cleanly
      to the HTTP / WS response.
    """

    def __init__(self) -> None:
        # The receiver lives on the main thread (created from main thread
        # context during server.__init__, which runs before the bg thread).
        self._receiver = _BridgeReceiver()

    async def invoke_on_qt(self, fn: Callable[[], T]) -> T:
        """Call *fn()* on the Qt main thread, returning the result.

        This method is ``await``-ed from the asyncio event loop running on
        the server thread.  It emits a signal that Qt marshals to the main
        thread, then wraps the resulting :class:`concurrent.futures.Future`
        so that ``await`` returns once Qt has finished.

        Args:
            fn: A zero-argument callable to execute on the Qt main thread.

        Returns:
            Whatever *fn()* returns.

        Raises:
            Any exception raised inside *fn()* is re-raised here.
        """
        future: Future[T] = Future()
        # Emit from the server thread — Qt marshals via QueuedConnection
        self._receiver.execute.emit(fn, future)
        loop = asyncio.get_running_loop()
        return await asyncio.wrap_future(future, loop=loop)
