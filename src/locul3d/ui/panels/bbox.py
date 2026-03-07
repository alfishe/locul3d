"""Full BBox annotation panel: list + property editor + tool/axis switching."""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QGridLayout,
    QFrame, QComboBox, QAbstractItemView, QSlider,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ...core.geometry import BBoxItem
from ...core.constants import BBOX_COLORS, DEFAULT_SIZES, AXIS_COLORS


def _axis_qcolor(axis):
    """Return CSS color string for axis index."""
    r, g, b = AXIS_COLORS[axis]
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


class BBoxPanel(QWidget):
    """Non-blocking dock panel: bbox list + 3D-editor-style property panel."""

    bbox_changed = Signal(int)
    selection_changed = Signal(int)
    delete_requested = Signal(int)
    create_requested = Signal()
    tool_changed = Signal(str)
    axis_changed = Signal(object)  # int index or None
    pos_mode_changed = Signal(str)  # "center" or "corners"

    def __init__(self, annotations, parent=None):
        super().__init__(parent)
        self.annotations = annotations
        self._updating = False
        # Coordinate transform callbacks (set by window)
        self._world_to_display = None
        self._display_to_world = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Tool mode buttons ---
        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("Tool:"))
        self.tool_btns = {}
        for tool_id, label in [("select", "Q"), ("move", "G"), ("rotate", "R"), ("scale", "S")]:
            btn = QPushButton(f"{tool_id.capitalize()} ({label})")
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.setStyleSheet("QPushButton:checked { background: #3a6; color: white; border-radius: 3px; }")
            btn.clicked.connect(lambda checked, t=tool_id: self._on_tool_clicked(t))
            tool_row.addWidget(btn)
            self.tool_btns[tool_id] = btn
        self.tool_btns["select"].setChecked(True)
        layout.addLayout(tool_row)

        # --- Axis constraint buttons ---
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis:"))
        self.axis_btns = {}
        for i, axis_name in enumerate(["X", "Y", "Z"]):
            btn = QPushButton(axis_name)
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.setFixedWidth(32)
            btn.setStyleSheet(f"QPushButton {{ color: {_axis_qcolor(i)}; }}"
                              f"QPushButton:checked {{ background: {_axis_qcolor(i)}; color: white; border-radius: 3px; }}")
            btn.clicked.connect(lambda checked, a=i: self._on_axis_clicked(a, checked))
            axis_row.addWidget(btn)
            self.axis_btns[i] = btn
        axis_row.addStretch()
        layout.addLayout(axis_row)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("+ New")
        self.btn_new.setToolTip("Create new bbox at camera target (N)")
        self.btn_new.clicked.connect(lambda: self.create_requested.emit())
        btn_row.addWidget(self.btn_new)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip("Delete selected bbox (Del)")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_row.addWidget(self.btn_delete)

        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.setToolTip("Duplicate selected bbox (Ctrl+D)")
        self.btn_duplicate.clicked.connect(self._on_duplicate)
        btn_row.addWidget(self.btn_duplicate)
        layout.addLayout(btn_row)

        # --- List ---
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_list_selection)
        self.list_widget.setMaximumHeight(200)
        layout.addWidget(self.list_widget)

        # --- Property editor ---
        self.prop_frame = QFrame()
        self.prop_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prop = QGridLayout(self.prop_frame)
        prop.setContentsMargins(6, 6, 6, 6)
        prop.setSpacing(4)

        row = 0
        prop.addWidget(QLabel("Label:"), row, 0)
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems(list(DEFAULT_SIZES.keys()))
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        prop.addWidget(self.label_combo, row, 1, 1, 5)

        row += 1
        prop.addWidget(QLabel("Color:"), row, 0)
        color_row = QHBoxLayout()
        self.color_btns = []
        for ci, c in enumerate(BBOX_COLORS):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            qc = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            btn.setStyleSheet(f"background-color: {qc.name()}; border: 1px solid #555; border-radius: 3px;")
            btn.clicked.connect(lambda _, idx=ci: self._on_color_picked(idx))
            color_row.addWidget(btn)
            self.color_btns.append(btn)
        color_row.addStretch()
        prop.addLayout(color_row, row, 1, 1, 5)

        # Position: X Y Z  (with Center/Corners toggle)
        row += 1
        pos_header = QHBoxLayout()
        pos_header.addWidget(self._bold_label("Position"))
        pos_header.addStretch()
        self._pos_mode = "center"   # "center" or "corners"
        self._pos_mode_btn = QPushButton("Center")
        self._pos_mode_btn.setFixedWidth(70)
        self._pos_mode_btn.setToolTip("Toggle between Center+Size and Min/Max corners")
        self._pos_mode_btn.clicked.connect(self._toggle_pos_mode)
        pos_header.addWidget(self._pos_mode_btn)
        prop.addLayout(pos_header, row, 0, 1, 6)
        self.pos_spins = {}
        self._center_widgets = []  # labels + spinners to show/hide
        for i, axis in enumerate(["X", "Y", "Z"]):
            row += 1
            lbl = QLabel(f"  {axis}:")
            lbl.setStyleSheet(f"color: {_axis_qcolor(i)};")
            prop.addWidget(lbl, row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.valueChanged.connect(self._on_pos_changed)
            prop.addWidget(sp, row, 1, 1, 5)
            self.pos_spins[i] = sp
            self._center_widgets.extend([lbl, sp])

        # Corner Min/Max spinners (hidden by default, shown in "corners" mode)
        self._corner_min_label = self._bold_label("Min Corner")
        row += 1
        prop.addWidget(self._corner_min_label, row, 0, 1, 6)
        self.corner_min_spins = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            row += 1
            lbl = QLabel(f"  {axis}:")
            lbl.setStyleSheet(f"color: {_axis_qcolor(i)};")
            prop.addWidget(lbl, row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.valueChanged.connect(self._on_corner_changed)
            prop.addWidget(sp, row, 1, 1, 5)
            self.corner_min_spins[i] = sp
            self._corner_min_widgets = self._corner_min_widgets if hasattr(self, '_corner_min_widgets') else []
            self._corner_min_widgets.extend([lbl, sp])

        self._corner_max_label = self._bold_label("Max Corner")
        row += 1
        prop.addWidget(self._corner_max_label, row, 0, 1, 6)
        self.corner_max_spins = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            row += 1
            lbl = QLabel(f"  {axis}:")
            lbl.setStyleSheet(f"color: {_axis_qcolor(i)};")
            prop.addWidget(lbl, row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.valueChanged.connect(self._on_corner_changed)
            prop.addWidget(sp, row, 1, 1, 5)
            self.corner_max_spins[i] = sp
            self._corner_max_widgets = self._corner_max_widgets if hasattr(self, '_corner_max_widgets') else []
            self._corner_max_widgets.extend([lbl, sp])

        # Collect all corner widgets for show/hide
        self._corner_widgets = ([self._corner_min_label] +
                                self._corner_min_widgets +
                                [self._corner_max_label] +
                                self._corner_max_widgets)

        # Size: X Y Z (hidden in corners mode — redundant with min/max)
        row += 1
        self._size_label = self._bold_label("Size")
        prop.addWidget(self._size_label, row, 0, 1, 6)
        self.size_spins = {}
        self._size_widgets = [self._size_label]
        for i, axis in enumerate(["X", "Y", "Z"]):
            row += 1
            lbl = QLabel(f"  {axis}:")
            lbl.setStyleSheet(f"color: {_axis_qcolor(i)};")
            prop.addWidget(lbl, row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 10000)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.valueChanged.connect(self._on_size_changed)
            prop.addWidget(sp, row, 1, 1, 5)
            self.size_spins[i] = sp
            self._size_widgets.extend([lbl, sp])

        # Rotation Z
        row += 1
        prop.addWidget(self._bold_label("Rotation"), row, 0, 1, 6)
        row += 1
        lbl = QLabel("  Z:")
        lbl.setStyleSheet(f"color: {_axis_qcolor(2)};")
        prop.addWidget(lbl, row, 0)
        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-360, 360)
        self.rot_z_spin.setDecimals(1)
        self.rot_z_spin.setSingleStep(1.0)
        self.rot_z_spin.setSuffix(" deg")
        self.rot_z_spin.valueChanged.connect(self._on_rot_changed)
        prop.addWidget(self.rot_z_spin, row, 1, 1, 5)

        # Fill opacity slider
        row += 1
        prop.addWidget(self._bold_label("Fill"), row, 0, 1, 6)
        row += 1
        fill_row = QHBoxLayout()
        self.fill_slider = QSlider(Qt.Orientation.Horizontal)
        self.fill_slider.setRange(0, 100)
        self.fill_slider.setValue(0)
        self.fill_slider.setToolTip("Fill surface opacity (0% = wireframe only)")
        self.fill_slider.valueChanged.connect(self._on_fill_changed)
        self.fill_label = QLabel("0%")
        self.fill_label.setFixedWidth(36)
        fill_row.addWidget(self.fill_slider)
        fill_row.addWidget(self.fill_label)
        prop.addLayout(fill_row, row, 0, 1, 6)

        # Preset
        row += 1
        prop.addWidget(QLabel("Preset:"), row, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(DEFAULT_SIZES.keys()))
        prop.addWidget(self.preset_combo, row, 1, 1, 3)
        btn_apply = QPushButton("Apply Size")
        btn_apply.clicked.connect(self._on_apply_preset)
        prop.addWidget(btn_apply, row, 4, 1, 2)

        layout.addWidget(self.prop_frame)
        layout.addStretch()
        self.prop_frame.setEnabled(False)

        # Initially show center mode, hide corners
        self._apply_pos_mode()

    @staticmethod
    def _bold_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; padding-top: 4px;")
        return lbl

    # --- Position mode: Center vs Corners ---

    def _toggle_pos_mode(self):
        """Toggle between Center+Size and Min/Max corners editing modes."""
        if self._pos_mode == "center":
            self._pos_mode = "corners"
        else:
            self._pos_mode = "center"
        self._apply_pos_mode()
        self.pos_mode_changed.emit(self._pos_mode)
        # Update the selected bbox's save format
        idx = self.list_widget.currentRow()
        if 0 <= idx < len(self.annotations):
            self.annotations[idx].save_format = self._pos_mode
            self._populate_props(idx)

    def _apply_pos_mode(self):
        """Show/hide widgets based on current position mode."""
        is_center = (self._pos_mode == "center")
        self._pos_mode_btn.setText("Center" if is_center else "Corners")

        # Center mode: show center XYZ + size XYZ, hide corner min/max
        for w in self._center_widgets:
            w.setVisible(is_center)
        for w in self._size_widgets:
            w.setVisible(is_center)
        for w in self._corner_widgets:
            w.setVisible(not is_center)

    def _on_corner_changed(self):
        """Handle corner min/max spinner edits — recompute center and size."""
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        bbox = self.annotations[idx]

        display_min = np.array([self.corner_min_spins[i].value() for i in range(3)],
                               dtype=np.float64)
        display_max = np.array([self.corner_max_spins[i].value() for i in range(3)],
                               dtype=np.float64)

        # Convert from display to world coords if needed
        if self._display_to_world is not None:
            world_min = self._display_to_world(display_min)
            world_max = self._display_to_world(display_max)
        else:
            world_min = display_min
            world_max = display_max

        # Ensure min <= max per axis
        for i in range(3):
            if world_min[i] > world_max[i]:
                world_min[i], world_max[i] = world_max[i], world_min[i]

        bbox.center_pos[:] = (world_min + world_max) / 2.0
        bbox.size[:] = world_max - world_min
        # Clamp size minimum
        for i in range(3):
            if bbox.size[i] < 0.01:
                bbox.size[i] = 0.01

        self.bbox_changed.emit(idx)

    # --- Tool / axis ---

    def set_tool(self, tool_id: str):
        """Programmatic tool change (no signal emitted)."""
        self._updating = True
        for tid, btn in self.tool_btns.items():
            btn.setChecked(tid == tool_id)
        self._updating = False

    def set_axis(self, axis):
        """Programmatic axis change (no signal emitted). axis=int or None."""
        self._updating = True
        for i, btn in self.axis_btns.items():
            btn.setChecked(i == axis)
        self._updating = False

    def _on_tool_clicked(self, tool_id):
        if self._updating:
            return
        for tid, btn in self.tool_btns.items():
            btn.setChecked(tid == tool_id)
        self.tool_changed.emit(tool_id)

    def _on_axis_clicked(self, axis, checked):
        if self._updating:
            return
        for i, btn in self.axis_btns.items():
            if i != axis:
                btn.setChecked(False)
        self.axis_changed.emit(axis if checked else None)

    # --- List management ---

    def rebuild_list(self):
        self._updating = True
        current = self.list_widget.currentRow()
        self.list_widget.clear()
        for i, bbox in enumerate(self.annotations):
            item = QListWidgetItem(f"[{i}] {bbox.label}")
            qc = QColor(int(bbox.color[0]*255), int(bbox.color[1]*255), int(bbox.color[2]*255))
            item.setForeground(qc)
            self.list_widget.addItem(item)
        if 0 <= current < len(self.annotations):
            self.list_widget.setCurrentRow(current)
        self._updating = False

    def select_bbox(self, idx):
        self._updating = True
        if 0 <= idx < self.list_widget.count():
            self.list_widget.setCurrentRow(idx)
        else:
            self.list_widget.clearSelection()
            self.list_widget.setCurrentRow(-1)
        self._updating = False
        self._populate_props(idx)

    def update_values(self, idx):
        if 0 <= idx < len(self.annotations):
            self._populate_props(idx)

    def _populate_props(self, idx):
        """Populate all property spinners from the bbox at the given index."""
        if idx < 0 or idx >= len(self.annotations):
            self.prop_frame.setEnabled(False)
            return
        self.prop_frame.setEnabled(True)
        bbox = self.annotations[idx]

        # Restore this item's position mode
        item_mode = getattr(bbox, 'save_format', 'center')
        if item_mode != self._pos_mode:
            self._pos_mode = item_mode
            self._apply_pos_mode()
            self.pos_mode_changed.emit(self._pos_mode)

        self._updating = True
        self.label_combo.setCurrentText(bbox.label)
        display_pos = bbox.center_pos.copy()
        if self._world_to_display is not None:
            display_pos = self._world_to_display(display_pos)
        for i in range(3):
            self.pos_spins[i].setValue(display_pos[i])
            self.size_spins[i].setValue(bbox.size[i])

        # Populate corner spinners
        bb_min = bbox.bb_min
        bb_max = bbox.bb_max
        if self._world_to_display is not None:
            bb_min = self._world_to_display(bb_min)
            bb_max = self._world_to_display(bb_max)
        for i in range(3):
            self.corner_min_spins[i].setValue(bb_min[i])
            self.corner_max_spins[i].setValue(bb_max[i])

        self.rot_z_spin.setValue(bbox.rotation_z)
        fill_pct = int(round(bbox.fill_opacity * 100))
        self.fill_slider.setValue(fill_pct)
        self.fill_label.setText(f"{fill_pct}%")
        self._updating = False

    # --- Property change handlers ---

    def _on_list_selection(self, row):
        if not self._updating:
            self._populate_props(row)
            self.selection_changed.emit(row)

    def _on_pos_changed(self):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        bbox = self.annotations[idx]
        display_pos = np.array([self.pos_spins[i].value() for i in range(3)],
                               dtype=np.float64)
        if self._display_to_world is not None:
            world_pos = self._display_to_world(display_pos)
        else:
            world_pos = display_pos
        bbox.center_pos[:] = world_pos
        self.bbox_changed.emit(idx)

    def _on_size_changed(self):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        bbox = self.annotations[idx]
        for i in range(3):
            bbox.size[i] = self.size_spins[i].value()
        self.bbox_changed.emit(idx)

    def _on_rot_changed(self):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        self.annotations[idx].rotation_z = self.rot_z_spin.value()
        self.bbox_changed.emit(idx)

    def _on_fill_changed(self, value):
        """Handle fill opacity slider change (0–100 → 0.0–1.0)."""
        self.fill_label.setText(f"{value}%")
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        self.annotations[idx].fill_opacity = value / 100.0
        self.bbox_changed.emit(idx)

    def _on_label_changed(self, text):
        if self._updating:
            return
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        self.annotations[idx].label = text
        item = self.list_widget.item(idx)
        if item:
            item.setText(f"[{idx}] {text}")

    def _on_color_picked(self, color_idx):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        self.annotations[idx].color = list(BBOX_COLORS[color_idx])
        self.bbox_changed.emit(idx)
        self.rebuild_list()
        self.list_widget.setCurrentRow(idx)

    def _on_apply_preset(self):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        size = DEFAULT_SIZES.get(self.preset_combo.currentText(), DEFAULT_SIZES["custom"])
        self.annotations[idx].size = size.copy()
        self._populate_props(idx)
        self.bbox_changed.emit(idx)

    def _on_delete(self):
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.delete_requested.emit(idx)

    def _on_duplicate(self):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.annotations):
            return
        src = self.annotations[idx]
        new = BBoxItem(
            label=src.label,
            center=src.center_pos + np.array([0, 0.05, 0]),
            size=src.size.copy(),
            rotation_z=src.rotation_z,
            color=list(src.color),
        )
        self.annotations.append(new)
        self.rebuild_list()
        self.list_widget.setCurrentRow(len(self.annotations) - 1)
        self.selection_changed.emit(len(self.annotations) - 1)
        self.bbox_changed.emit(len(self.annotations) - 1)
