"""Full surface plane panel: list + property editor + color/opacity."""

from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QGridLayout,
    QFrame, QComboBox, QSlider, QAbstractItemView, QCheckBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ...core.geometry import PlaneItem
from ...core.constants import PLANE_COLORS, AXIS_COLORS


def _axis_qcolor(axis):
    """Return CSS color string for axis index."""
    r, g, b = AXIS_COLORS[axis]
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


class PlanePanel(QWidget):
    """Dock panel for managing reference surface planes."""

    plane_changed = Signal(int)
    selection_changed = Signal(int)
    delete_requested = Signal(int)
    create_requested = Signal()

    def __init__(self, planes, parent=None):
        super().__init__(parent)
        self.planes = planes
        self._updating = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("+ New Plane")
        self.btn_new.clicked.connect(lambda: self.create_requested.emit())
        btn_row.addWidget(self.btn_new)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_row.addWidget(self.btn_delete)
        layout.addLayout(btn_row)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_list_selection)
        self.list_widget.setMaximumHeight(140)
        layout.addWidget(self.list_widget)

        # Properties
        self.prop_frame = QFrame()
        self.prop_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prop = QGridLayout(self.prop_frame)
        prop.setContentsMargins(6, 6, 6, 6)
        prop.setSpacing(4)

        row = 0
        prop.addWidget(QLabel("Axis:"), row, 0)
        self.axis_combo = QComboBox()
        self.axis_combo.setEditable(False)
        self.axis_combo.addItems(["XY", "XZ", "YZ"])
        self.axis_combo.setStyleSheet("QComboBox { min-width: 60px; }")
        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)
        prop.addWidget(self.axis_combo, row, 1, 1, 3)

        row += 1
        self.global_check = QCheckBox("Global coordinates")
        self.global_check.setToolTip(
            "When checked, this plane is drawn in global (uncorrected) space — "
            "use as a fixed reference to align the scene correction")
        self.global_check.stateChanged.connect(self._on_global_changed)
        prop.addWidget(self.global_check, row, 0, 1, 4)

        # Center position
        row += 1
        lbl = QLabel("Center")
        lbl.setStyleSheet("font-weight: bold; padding-top: 4px;")
        prop.addWidget(lbl, row, 0, 1, 4)
        self.center_spins = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            row += 1
            al = QLabel(f"  {axis}:")
            al.setStyleSheet(f"color: {_axis_qcolor(i)};")
            prop.addWidget(al, row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.valueChanged.connect(self._on_prop_changed)
            prop.addWidget(sp, row, 1, 1, 3)
            self.center_spins[i] = sp

        # Size
        row += 1
        lbl = QLabel("Size")
        lbl.setStyleSheet("font-weight: bold; padding-top: 4px;")
        prop.addWidget(lbl, row, 0, 1, 4)
        self.size_spins = {}
        for i, label in enumerate(["W", "H"]):
            row += 1
            prop.addWidget(QLabel(f"  {label}:"), row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.5)
            sp.valueChanged.connect(self._on_prop_changed)
            prop.addWidget(sp, row, 1, 1, 3)
            self.size_spins[i] = sp

        # Color
        row += 1
        prop.addWidget(QLabel("Color:"), row, 0)
        color_row = QHBoxLayout()
        self.color_btns = []
        for ci, c in enumerate(PLANE_COLORS):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            qc = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            btn.setStyleSheet(f"background-color: {qc.name()}; border: 1px solid #555; border-radius: 3px;")
            btn.clicked.connect(lambda _, idx=ci: self._on_color_picked(idx))
            color_row.addWidget(btn)
            self.color_btns.append(btn)
        color_row.addStretch()
        prop.addLayout(color_row, row, 1, 1, 3)

        # Opacity
        row += 1
        prop.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 90)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        prop.addWidget(self.opacity_slider, row, 1, 1, 2)
        self.opacity_label = QLabel("30%")
        prop.addWidget(self.opacity_label, row, 3)

        layout.addWidget(self.prop_frame)
        layout.addStretch()
        self.prop_frame.setEnabled(False)

    def rebuild_list(self):
        self._updating = True
        current = self.list_widget.currentRow()
        self.list_widget.clear()
        for i, plane in enumerate(self.planes):
            item = QListWidgetItem(f"[{i}] {plane.axis.upper()} plane{' (global)' if plane.global_coords else ''}")
            qc = QColor(int(plane.color[0]*255), int(plane.color[1]*255), int(plane.color[2]*255))
            item.setForeground(qc)
            self.list_widget.addItem(item)
        if 0 <= current < len(self.planes):
            self.list_widget.setCurrentRow(current)
        self._updating = False

    def select_plane(self, idx):
        self._updating = True
        if 0 <= idx < self.list_widget.count():
            self.list_widget.setCurrentRow(idx)
        else:
            self.list_widget.clearSelection()
            self.list_widget.setCurrentRow(-1)
        self._updating = False
        self._populate_props(idx)

    def _populate_props(self, idx):
        if idx < 0 or idx >= len(self.planes):
            self.prop_frame.setEnabled(False)
            return
        self.prop_frame.setEnabled(True)
        plane = self.planes[idx]
        self._updating = True
        axis_map = {'xy': 0, 'xz': 1, 'yz': 2}
        self.axis_combo.setCurrentIndex(axis_map.get(plane.axis, 0))
        self.global_check.setChecked(plane.global_coords)
        for i in range(3):
            self.center_spins[i].setValue(plane.center[i])
        for i in range(2):
            self.size_spins[i].setValue(plane.size[i])
        self.opacity_slider.setValue(int(plane.opacity * 100))
        self.opacity_label.setText(f"{int(plane.opacity * 100)}%")
        self._updating = False

    def _on_list_selection(self, row):
        if not self._updating:
            self._populate_props(row)
            self.selection_changed.emit(row)

    def _on_prop_changed(self):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.planes):
            return
        plane = self.planes[idx]
        for i in range(3):
            plane.center[i] = self.center_spins[i].value()
        for i in range(2):
            plane.size[i] = self.size_spins[i].value()
        self.plane_changed.emit(idx)

    def _on_axis_changed(self, text):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes[idx].axis = text.lower()
        self.rebuild_list()
        self.list_widget.setCurrentRow(idx)
        self.plane_changed.emit(idx)

    def _on_global_changed(self, state):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes[idx].global_coords = bool(state)
        self.rebuild_list()
        self.list_widget.setCurrentRow(idx)
        self.plane_changed.emit(idx)

    def _on_color_picked(self, color_idx):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes[idx].color = list(PLANE_COLORS[color_idx])
        self.plane_changed.emit(idx)
        self.rebuild_list()
        self.list_widget.setCurrentRow(idx)

    def _on_opacity_changed(self, val):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes[idx].opacity = val / 100.0
        self.opacity_label.setText(f"{val}%")
        self.plane_changed.emit(idx)

    def _on_delete(self):
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.delete_requested.emit(idx)
