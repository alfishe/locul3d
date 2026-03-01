"""Rotate tool plugin."""

from PySide6.QtCore import Qt, QMouseEvent
from typing import Optional

from ..base import ToolPlugin


class RotateTool(ToolPlugin):
    """Tool for rotating selected objects around Z axis."""

    @property
    def name(self) -> str:
        return "Rotate"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def tool_id(self) -> str:
        return "rotate"

    @property
    def shortcut(self) -> Optional[str]:
        return "R"

    def activate(self, viewport):
        """Activate rotate tool."""
        viewport.setCursor(Qt.CursorShape.CrossCursor)

    def deactivate(self, viewport):
        """Deactivate rotate tool."""
        viewport.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent, viewport):
        """Handle mouse press - start rotation."""
        if event.button() == Qt.MouseButton.LeftButton:
            pass  # Start rotation

    def handle_mouse_move(self, event: QMouseEvent, viewport):
        """Handle mouse move - continue rotation."""
        pass  # Update rotation

    def handle_mouse_release(self, event: QMouseEvent, viewport):
        """Handle mouse release - end rotation."""
        pass
