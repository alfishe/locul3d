"""Move tool plugin."""

from PySide6.QtCore import Qt, QMouseEvent
from typing import Optional

from ..base import ToolPlugin


class MoveTool(ToolPlugin):
    """Tool for moving selected objects."""

    @property
    def name(self) -> str:
        return "Move"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def tool_id(self) -> str:
        return "move"

    @property
    def shortcut(self) -> Optional[str]:
        return "G"

    def activate(self, viewport):
        """Activate move tool."""
        viewport.setCursor(Qt.CursorShape.SizeAllCursor)

    def deactivate(self, viewport):
        """Deactivate move tool."""
        viewport.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent, viewport):
        """Handle mouse press - start drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            pass  # Start drag

    def handle_mouse_move(self, event: QMouseEvent, viewport):
        """Handle mouse move - continue drag."""
        pass  # Update position

    def handle_mouse_release(self, event: QMouseEvent, viewport):
        """Handle mouse release - end drag."""
        pass
