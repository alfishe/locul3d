"""Layer row widget for individual layer control."""

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QCheckBox, QSlider,
    QWidget, QPushButton, QScrollArea, QSizePolicy, QColorDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QColor, QMouseEvent

from ...core.layer import LayerData, LayerManager
from ...core.constants import COLORS


class _ClickableLabel(QLabel):
    """QLabel that emits clicked on mouse press."""
    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        event.accept()  # consume — don't propagate to parent row


class LayerRowWidget(QFrame):
    """Widget representing a single layer with visibility, opacity, and color controls."""

    visibility_changed = Signal()
    opacity_changed = Signal()
    opacity_adjusting = Signal(bool)   # True while slider is being dragged
    pano_requested = Signal(object)    # emits LayerData when Enter 360° clicked
    layer_selected = Signal(object)

    def __init__(self, layer: LayerData, parent=None):
        super().__init__(parent)
        self.layer = layer
        self._selected = False

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Visibility checkbox — styled to look distinct from color swatch
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(layer.visible)
        self.checkbox.setToolTip("Show/Hide")
        self.checkbox.setStyleSheet(f"""
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 2px solid {COLORS['border']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['accent']};
                border-color: {COLORS['accent']};
            }}
            QCheckBox::indicator:unchecked {{
                background: {COLORS['input_bg']};
            }}
        """)
        self.checkbox.toggled.connect(self._on_visibility)
        layout.addWidget(self.checkbox)

        # Color swatch — clickable to open color picker
        self._swatch = _ClickableLabel()
        self._swatch.setFixedSize(16, 12)
        self._swatch.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._swatch.setToolTip("Click to change layer color")
        self._swatch.clicked.connect(self._on_swatch_clicked)
        self._update_swatch()
        layout.addWidget(self._swatch)

        # Layer name
        self.name_label = QLabel(layer.name)
        self.name_label.setMinimumWidth(60)
        self.name_label.setToolTip(layer.name)
        self.name_label.setTextFormat(Qt.TextFormat.PlainText)
        from PySide6.QtWidgets import QSizePolicy
        self.name_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.name_label, stretch=1)

        # Panorama layers get an Enter button; regular layers get info + slider
        is_pano = (layer.layer_type == "panorama")

        if not is_pano:
            # Info label - point/triangle count
            if layer.load_error:
                info_text = "ERR"
            elif layer.tri_count > 0:
                info_text = f"{layer.tri_count:,}T"
            elif layer.point_count > 0:
                info_text = f"{layer.point_count:,}P"
            else:
                info_text = ""

            info = QLabel(info_text)
            info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
            info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(info)

        # Opacity slider (all layer types including panorama)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(layer.opacity * 100))
        self.slider.setFixedWidth(50)
        self.slider.setToolTip("Opacity")
        self.slider.valueChanged.connect(self._on_opacity)
        self.slider.sliderPressed.connect(lambda: self.opacity_adjusting.emit(True))
        self.slider.sliderReleased.connect(lambda: self.opacity_adjusting.emit(False))
        layout.addWidget(self.slider)

        if is_pano:
            # Panorama enter button
            pano_btn = QPushButton("360°")
            pano_btn.setFixedSize(40, 22)
            pano_btn.setToolTip("Enter 360° panorama view (Esc to exit)")
            pano_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['accent']}; color: white; "
                f"border-radius: 3px; font-size: 10px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: {COLORS['hover']}; }}"
            )
            pano_btn.clicked.connect(lambda: self.pano_requested.emit(self.layer))
            layout.addWidget(pano_btn)

    def _on_visibility(self, checked: bool):
        """Handle visibility checkbox toggle."""
        self.layer.visible = checked
        if not checked:
            self.layer.evict_byte_caches()
        self.visibility_changed.emit()

    def _on_opacity(self, value: int):
        """Handle opacity slider change."""
        self.layer.opacity = value / 100.0
        self.opacity_changed.emit()

    def _update_swatch(self):
        """Update swatch color from layer."""
        if self.layer.color:
            r, g, b = [int(c * 255) for c in self.layer.color[:3]]
            sc = f"rgb({r},{g},{b})"
        else:
            sc = "#b4b4c8"
        self._swatch.setStyleSheet(
            f"background: {sc}; border: 1px solid {COLORS['swatch_border']};"
            f" border-radius: 3px;"
        )

    def _on_swatch_clicked(self):
        """Open color picker dialog."""
        if self.layer.color:
            r, g, b = [int(c * 255) for c in self.layer.color[:3]]
            initial = QColor(r, g, b)
        else:
            initial = QColor(180, 180, 200)
        color = QColorDialog.getColor(initial, self, "Layer Color")
        if color.isValid():
            self.layer.color = [color.redF(), color.greenF(), color.blueF(), 1.0]
            self._update_swatch()
            self.layer.evict_byte_caches()
            self.visibility_changed.emit()  # triggers viewport redraw

    def sync_from_layer(self):
        """Update UI to match layer state (after programmatic changes)."""
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(self.layer.visible)
        self.checkbox.blockSignals(False)
        if self.slider is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(int(self.layer.opacity * 100))
            self.slider.blockSignals(False)

    def set_selected(self, selected: bool):
        """Highlight this row as selected."""
        self._selected = selected
        if selected:
            self.setStyleSheet(f"background: {COLORS['selected']}; border-radius: 4px;")
        else:
            self._apply_pano_style()

    def set_pano_active(self, active: bool):
        """Show a green border when this panorama is the active 360° view."""
        self._pano_active = active
        self._apply_pano_style()

    def _apply_pano_style(self):
        """Apply the active-panorama visual if applicable."""
        if getattr(self, '_pano_active', False):
            self.setStyleSheet(
                "outline: 2px solid #4CAF50; border-radius: 4px; "
                "background: rgba(76, 175, 80, 40);"
            )
        elif not self._selected:
            self.setStyleSheet("")

    def mousePressEvent(self, event):
        """Handle click on row body (not on checkbox/slider) to select."""
        # Only emit selection on left-click on row itself
        if event.button() == Qt.MouseButton.LeftButton:
            self.layer_selected.emit(self.layer)
        super().mousePressEvent(event)


class LayerPanel(QWidget):
    """Sidebar widget with grouped layer controls."""

    layer_changed = Signal()
    layer_selected = Signal(object)
    opacity_adjusting = Signal(bool)  # True while any slider is being dragged
    pano_requested = Signal(object)   # emits LayerData for panorama enter

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self._row_widgets: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_show = QPushButton("Show All")
        btn_show.setFixedHeight(26)
        btn_show.clicked.connect(self._on_show_all)
        btn_row.addWidget(btn_show)
        btn_hide = QPushButton("Hide All")
        btn_hide.setFixedHeight(26)
        btn_hide.clicked.connect(self._on_hide_all)
        btn_row.addWidget(btn_hide)
        layout.addLayout(btn_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet(
            "QScrollBar:vertical { width: 12px; }"
        )
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 4, 0, 0)
        self._scroll_layout.setSpacing(1)
        self._scroll.setWidget(self._scroll_content)
        layout.addWidget(self._scroll)

    def rebuild(self):
        """Rebuild layer list from layer manager."""
        self._row_widgets.clear()
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setVisible(False)
                w.deleteLater()

        grouped = self._categorize_layers()

        for group_name, layers in grouped.items():
            if not layers:
                continue

            header = QLabel(f"  {group_name}  ({len(layers)})")
            header.setStyleSheet(
                f"color: {COLORS['text']}; font-size: 11px; font-weight: bold; "
                f"padding: 6px 0 3px 0; border-bottom: 1px solid {COLORS['border']};"
            )
            self._scroll_layout.addWidget(header)

            for layer in layers:
                row = LayerRowWidget(layer)
                row.visibility_changed.connect(self._on_layer_changed)
                row.opacity_changed.connect(self._on_layer_changed)
                row.opacity_adjusting.connect(self.opacity_adjusting)
                row.pano_requested.connect(self.pano_requested)
                row.layer_selected.connect(self._on_layer_selected)
                self._scroll_layout.addWidget(row)
                self._row_widgets.append(row)

        self._scroll_layout.addStretch()

    def _categorize_layers(self):
        """Sort layers into display groups."""
        from ...core.constants import LAYER_GROUPS
        from collections import OrderedDict
        groups = OrderedDict((name, []) for name in LAYER_GROUPS)
        groups["Other"] = []

        for layer in self.layer_manager.layers:
            placed = False
            for group_name, classifier in LAYER_GROUPS.items():
                if classifier(layer):
                    groups[group_name].append(layer)
                    placed = True
                    break
            if not placed:
                groups["Other"].append(layer)

        return groups

    def highlight_active_pano(self, active_layer=None):
        """Highlight the row for the active panorama (green border).

        Call with ``None`` to clear all highlights.
        """
        for row in self._row_widgets:
            if row.layer.layer_type == "panorama":
                row.set_pano_active(row.layer is active_layer)

    def _on_layer_changed(self):
        self.layer_changed.emit()

    def _on_layer_selected(self, layer):
        """Handle a layer row being clicked - highlight and emit selection."""
        for row in self._row_widgets:
            row.set_selected(row.layer is layer)
        self.layer_selected.emit(layer)

    def _on_show_all(self):
        for layer in self.layer_manager.layers:
            layer.visible = True
        for row in self._row_widgets:
            row.sync_from_layer()
        self.layer_changed.emit()

    def _on_hide_all(self):
        for layer in self.layer_manager.layers:
            layer.visible = False
        for row in self._row_widgets:
            row.sync_from_layer()
        self.layer_changed.emit()

    def sync_all(self):
        """Sync all row widgets from layer state."""
        for row in self._row_widgets:
            row.sync_from_layer()

    def select_layer_by_data(self, layer, notify: bool = False):
        """Select a layer by its LayerData object (e.g., from marker click).

        Highlights the row and scrolls to make it visible.
        Pass ``None`` to clear selection.

        Parameters
        ----------
        notify : bool
            If True, also emit ``layer_selected`` to trigger info panel.
            Default is False (highlight + scroll only).
        """
        for row in self._row_widgets:
            is_match = layer is not None and row.layer is layer
            row.set_selected(is_match)
            if is_match:
                self._scroll.ensureWidgetVisible(row)
        if notify:
            self.layer_selected.emit(layer)
