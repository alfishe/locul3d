"""Scene dialog — non-modal dialog for scene bounds & ceiling clipping.

Features:
  - Pre-populated scene dimensions (X, Y, Z min/max) from cached AABB
  - Spinners with 0.1m precision and direct typing
  - "Hide Ceiling" button: reads pre-computed ceiling height
  - "Reset" button: remove all clipping
  - Live preview: changes are applied immediately to the viewport
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox,
)


class SceneDialog(QDialog):
    """Non-modal dialog for adjusting scene clip bounds.

    Uses LayerManager.scene_aabb (cached) for instant population.
    Emits *clip_changed* whenever bounds are adjusted.
    """

    clip_changed = Signal(float, float, float, float, float, float)

    def __init__(self, layer_manager, viewport, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene")
        self.setMinimumWidth(340)
        self.setModal(False)

        self._layer_manager = layer_manager
        self._viewport = viewport

        self._build_ui()
        self._populate_from_cache()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Scene Dimensions group ---
        dims_group = QGroupBox("Scene Bounds (metres)")
        dims_grid = QGridLayout()

        dims_grid.addWidget(QLabel(""), 0, 0)
        lbl_min = QLabel("Min")
        lbl_min.setStyleSheet("font-weight: bold;")
        dims_grid.addWidget(lbl_min, 0, 1)
        lbl_max = QLabel("Max")
        lbl_max.setStyleSheet("font-weight: bold;")
        dims_grid.addWidget(lbl_max, 0, 2)
        lbl_span = QLabel("Span")
        lbl_span.setStyleSheet("font-weight: bold; color: #888;")
        dims_grid.addWidget(lbl_span, 0, 3)

        self._spin_x_min = self._make_spin(-10000, 10000)
        self._spin_x_max = self._make_spin(-10000, 10000)
        self._lbl_x_span = QLabel("—")
        self._lbl_x_span.setStyleSheet("color: #888;")

        self._spin_y_min = self._make_spin(-10000, 10000)
        self._spin_y_max = self._make_spin(-10000, 10000)
        self._lbl_y_span = QLabel("—")
        self._lbl_y_span.setStyleSheet("color: #888;")

        self._spin_z_min = self._make_spin(-10000, 10000)
        self._spin_z_max = self._make_spin(-10000, 10000)
        self._lbl_z_span = QLabel("—")
        self._lbl_z_span.setStyleSheet("color: #888;")

        for row, (axis, s_min, s_max, lbl_span) in enumerate([
            ("X", self._spin_x_min, self._spin_x_max, self._lbl_x_span),
            ("Y", self._spin_y_min, self._spin_y_max, self._lbl_y_span),
            ("Z", self._spin_z_min, self._spin_z_max, self._lbl_z_span),
        ], start=1):
            lbl = QLabel(f"{axis}:")
            lbl.setFixedWidth(20)
            dims_grid.addWidget(lbl, row, 0)
            dims_grid.addWidget(s_min, row, 1)
            dims_grid.addWidget(s_max, row, 2)
            dims_grid.addWidget(lbl_span, row, 3)

        dims_group.setLayout(dims_grid)
        layout.addWidget(dims_group)

        # --- Actions ---
        actions = QHBoxLayout()

        btn_ceiling = QPushButton("Hide Ceiling")
        btn_ceiling.setToolTip("Auto-detect ceiling height and clip scene below it")
        btn_ceiling.clicked.connect(self._on_hide_ceiling)
        actions.addWidget(btn_ceiling)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Remove all clipping — show full scene")
        btn_reset.clicked.connect(self._on_reset)
        actions.addWidget(btn_reset)

        layout.addLayout(actions)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status)

    def _make_spin(self, lo: float, hi: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setSuffix(" m")
        spin.setKeyboardTracking(True)
        spin.valueChanged.connect(self._on_value_changed)
        return spin

    def _populate_from_cache(self):
        """Pre-populate spinners from cached scene AABB (instant)."""
        aabb = self._layer_manager.scene_aabb
        if aabb is None:
            return

        x0, x1, y0, y1, z0, z1 = aabb
        self._orig_bounds = aabb

        for spin in self._all_spins():
            spin.blockSignals(True)

        self._spin_x_min.setValue(x0)
        self._spin_x_max.setValue(x1)
        self._spin_y_min.setValue(y0)
        self._spin_y_max.setValue(y1)
        self._spin_z_min.setValue(z0)
        self._spin_z_max.setValue(z1)

        for spin in self._all_spins():
            spin.blockSignals(False)

        self._update_spans()

    def _all_spins(self):
        return (self._spin_x_min, self._spin_x_max,
                self._spin_y_min, self._spin_y_max,
                self._spin_z_min, self._spin_z_max)

    def _update_spans(self):
        self._lbl_x_span.setText(
            f"{self._spin_x_max.value() - self._spin_x_min.value():.1f} m")
        self._lbl_y_span.setText(
            f"{self._spin_y_max.value() - self._spin_y_min.value():.1f} m")
        self._lbl_z_span.setText(
            f"{self._spin_z_max.value() - self._spin_z_min.value():.1f} m")

    def _read_bounds(self):
        return (
            self._spin_x_min.value(), self._spin_x_max.value(),
            self._spin_y_min.value(), self._spin_y_max.value(),
            self._spin_z_min.value(), self._spin_z_max.value(),
        )

    def _on_value_changed(self):
        self._update_spans()
        x0, x1, y0, y1, z0, z1 = self._read_bounds()
        self.clip_changed.emit(x0, x1, y0, y1, z0, z1)

    def _on_hide_ceiling(self):
        """Set Z max just below the cached ceiling height.

        Ceiling is pre-computed in background after scene load.
        If no ceiling was detected, show a status message.
        """
        ceiling_z = self._layer_manager.ceiling_z

        if ceiling_z is None:
            self._status.setText("⚠ No clear ceiling surface detected")
            return

        # Set Z max to just below the ceiling (0.3m margin)
        clip_z = ceiling_z - 0.3
        self._spin_z_max.setValue(clip_z)
        self._status.setText(
            f"Ceiling at {ceiling_z:.1f} m — clipping at {clip_z:.1f} m")

    def _on_reset(self):
        """Remove all clipping — restore original scene bounds."""
        if hasattr(self, '_orig_bounds'):
            x0, x1, y0, y1, z0, z1 = self._orig_bounds
            for spin in self._all_spins():
                spin.blockSignals(True)
            self._spin_x_min.setValue(x0)
            self._spin_x_max.setValue(x1)
            self._spin_y_min.setValue(y0)
            self._spin_y_max.setValue(y1)
            self._spin_z_min.setValue(z0)
            self._spin_z_max.setValue(z1)
            for spin in self._all_spins():
                spin.blockSignals(False)
            self._update_spans()
            self.clip_changed.emit(x0, x1, y0, y1, z0, z1)
            self._viewport.scene_clip = None
            self._viewport.update()
            self._status.setText("Clipping reset — full scene visible")
