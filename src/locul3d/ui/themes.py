"""Theme management for the application."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

from ..core.constants import (
    COLORS, DARK_COLORS, LIGHT_COLORS,
    _detect_system_scheme
)


class ThemeManager:
    """Manages application theming (dark/light mode)."""

    def __init__(self):
        self._scheme = _detect_system_scheme()
        self._apply_theme()

    def _apply_theme(self):
        """Apply theme based on current system scheme."""
        scheme = self._scheme
        palette_src = DARK_COLORS if scheme == 'dark' else LIGHT_COLORS
        COLORS.clear()
        COLORS.update(palette_src)

        C = COLORS
        palette = QPalette()
        bg = QColor(C['bg'])
        card = QColor(C['card'])
        text = QColor(C['text'])
        accent = QColor(C['accent'])

        palette.setColor(QPalette.ColorRole.Window, bg)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, card)
        palette.setColor(QPalette.ColorRole.AlternateBase, bg)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, card)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipBase, card)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)

        app = QApplication.instance()
        if app:
            app.setPalette(palette)

    def get_stylesheet(self):
        """Return Qt stylesheet string for current theme."""
        C = COLORS
        return f"""
            QMainWindow {{ background: {C['bg']}; }}
            QToolBar {{ background: {C['card']}; border: none; border-bottom: 1px solid {C['border']}; padding: 4px; spacing: 4px; }}
            QToolBar QToolButton {{ color: {C['text']}; background: {C['button_bg']};
                padding: 5px 10px; border-radius: 4px; border: 1px solid {C['button_border']}; }}
            QToolBar QToolButton:hover {{ background: {C['hover']}; border-color: {C['accent']}; }}
            QToolBar QToolButton:checked {{ background: {C['accent']}; color: #ffffff; border-color: {C['accent']}; }}
            QToolBar::separator {{ background: transparent; width: 1px; height: 0px; margin: 4px 6px; }}
            QToolBar > QWidget {{ background: transparent; }}
            QStatusBar {{ background: {C['input_bg']}; color: {C['text_muted']}; font-size: 12px; }}
            QDockWidget {{ color: {C['text']}; font-size: 12px; }}
            QDockWidget::title {{ background: {C['card']}; padding: 6px; }}
            QGroupBox {{ border: 1px solid {C['border']}; border-radius: 6px; margin-top: 12px;
                        padding-top: 16px; color: {C['text_muted']}; font-size: 11px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
            QCheckBox {{ color: {C['text']}; spacing: 6px; font-size: 12px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px;
                                   border: 1px solid {C['checkbox_border']}; background: {C['checkbox_bg']}; }}
            QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
            QSlider::groove:horizontal {{ background: {C['slider_groove']}; height: 4px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {C['accent']}; width: 14px; height: 14px;
                                         margin: -5px 0; border-radius: 7px; }}
            QComboBox {{ background: {C['input_bg']}; color: {C['text']}; border: 1px solid {C['border']};
                        border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{ background: {C['input_bg']}; color: {C['text']};
                                          border: 1px solid {C['border']}; }}
            QPushButton {{ background: {C['button_bg']}; color: {C['text']}; border: 1px solid {C['button_border']};
                          border-radius: 4px; padding: 4px 10px; font-size: 11px; }}
            QPushButton:hover {{ background: {C['hover']}; }}
            QLabel {{ color: {C['text']}; font-size: 12px; }}
            QScrollArea {{ border: none; }}
        """

    def on_theme_changed(self):
        """Called when system theme changes."""
        self._scheme = _detect_system_scheme()
        self._apply_theme()

    @property
    def is_dark(self) -> bool:
        return self._scheme == 'dark'
