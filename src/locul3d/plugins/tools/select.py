"""Selection tool plugin."""

from PySide6.QtCore import Qt, QMouseEvent
from typing import Optional

from ..base import ToolPlugin


class SelectTool(ToolPlugin):
    """Tool for selecting objects in the viewport."""

    @property
    def name(self) -> str:
        return "Select"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def tool_id(self) -> str:
        return "select"

    @property
    def shortcut(self) -> Optional[str]:
        return "Q"

    def activate(self, viewport):
        """Activate select tool."""
        viewport.setCursor(Qt.CursorShape.ArrowCursor)

    def deactivate(self, viewport):
        """Deactivate select tool."""
        viewport.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent, viewport):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Perform selection logic
            pass

    def handle_mouse_move(self, event: QMouseEvent, viewport):
        """Handle mouse move."""
        pass

    def handle_mouse_release(self, event: QMouseEvent, viewport):
        """Handle mouse release."""
        pass
