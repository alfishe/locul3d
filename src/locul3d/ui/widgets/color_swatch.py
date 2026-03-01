"""Color swatch widget for color selection."""

from PySide6.QtWidgets import QWidget, QPushButton, QColorDialog, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QBrush, QPen
from typing import Optional

class ColorSwatch(QWidget):
    """Widget for displaying and selecting colors."""

    color_changed = Signal(QColor)

    def __init__(self, color: Optional[QColor] = None, parent=None):
        super().__init__(parent)
        self._color = color if color else QColor(255, 255, 255)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.button = QPushButton(self)
        self.button.setFixedSize(40, 40)
        self.button.clicked.connect(self.pick_color)
        self.update_button_color()

        layout.addWidget(self.button)

    def update_button_color(self):
        """Update button background to match current color."""
        self.button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name()};
                border: 2px solid #555;
                border-radius: 4px;
            }}
        """)

    def pick_color(self):
        """Open color dialog to pick a new color."""
        color = QColorDialog.getColor(self._color, self, "Select Color")
        if color.isValid():
            self.set_color(color)

    def set_color(self, color: QColor):
        """Set the current color."""
        self._color = color
        self.update_button_color()
        self.color_changed.emit(color)

    def color(self) -> QColor:
        """Return the current color."""
        return self._color


    def paintEvent(self, event):
        """Paint the colored rectangle."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Convert color to QColor if needed
        if isinstance(self._color, (list, tuple)):
            color = QColor(*(int(c * 255) for c in self._color[:3]))
        else:
            color = self._color
        
        from ...core.constants import COLORS
        p.setBrush(QBrush(color))
        p.setPen(QPen(QColor(COLORS['swatch_border']), 1))
        p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 3, 3)
        p.end()
