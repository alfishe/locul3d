"""Scene Correction dialog — live adjustment of rotation + shift for axis alignment."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox,
    QFileDialog, QMessageBox,
)

from ...core.correction import SceneCorrection


class CorrectionDialog(QDialog):
    """Global dialog for editing scene correction (rotations + shifts).

    Emits *correction_changed* whenever any value changes, enabling
    live preview in the viewport.
    """

    correction_changed = Signal(SceneCorrection)

    def __init__(self, correction: SceneCorrection, scene_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene Correction")
        self.setMinimumWidth(360)
        self._scene_dir = scene_dir
        self._correction = correction

        self._build_ui()
        self._load_values(correction)

    # ---- UI construction ----

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Rotation group ---
        rot_group = QGroupBox("Rotation (degrees)")
        rot_grid = QGridLayout()
        self._spin_rx = self._make_spin(-180.0, 180.0, 0.1)
        self._spin_ry = self._make_spin(-180.0, 180.0, 0.1)
        self._spin_rz = self._make_spin(-180.0, 180.0, 0.1)
        for row, (label, spin) in enumerate([
            ("X", self._spin_rx), ("Y", self._spin_ry), ("Z", self._spin_rz),
        ]):
            lbl = QLabel(f"Rotate {label}:")
            lbl.setFixedWidth(70)
            rot_grid.addWidget(lbl, row, 0)
            rot_grid.addWidget(spin, row, 1)
        rot_group.setLayout(rot_grid)
        layout.addWidget(rot_group)

        # --- Shift group ---
        shift_group = QGroupBox("Shift (scene units)")
        shift_grid = QGridLayout()
        self._spin_sx = self._make_spin(-10000.0, 10000.0, 0.01)
        self._spin_sy = self._make_spin(-10000.0, 10000.0, 0.01)
        self._spin_sz = self._make_spin(-10000.0, 10000.0, 0.01)
        for row, (label, spin) in enumerate([
            ("X", self._spin_sx), ("Y", self._spin_sy), ("Z", self._spin_sz),
        ]):
            lbl = QLabel(f"Shift {label}:")
            lbl.setFixedWidth(70)
            shift_grid.addWidget(lbl, row, 0)
            shift_grid.addWidget(spin, row, 1)
        shift_group.setLayout(shift_grid)
        layout.addWidget(shift_group)

        # --- Buttons ---
        btn_row = QHBoxLayout()

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(btn_reset)

        btn_row.addStretch()

        btn_load = QPushButton("Load YAML…")
        btn_load.clicked.connect(self._on_load)
        btn_row.addWidget(btn_load)

        btn_save = QPushButton("Save YAML…")
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_save)

        layout.addLayout(btn_row)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

    def _make_spin(self, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setKeyboardTracking(False)  # update only on Enter / focus-out
        spin.valueChanged.connect(self._on_value_changed)
        return spin

    # ---- value sync ----

    def _load_values(self, c: SceneCorrection):
        for spin in (self._spin_rx, self._spin_ry, self._spin_rz,
                     self._spin_sx, self._spin_sy, self._spin_sz):
            spin.blockSignals(True)
        self._spin_rx.setValue(c.rotate_x)
        self._spin_ry.setValue(c.rotate_y)
        self._spin_rz.setValue(c.rotate_z)
        self._spin_sx.setValue(c.shift_x)
        self._spin_sy.setValue(c.shift_y)
        self._spin_sz.setValue(c.shift_z)
        for spin in (self._spin_rx, self._spin_ry, self._spin_rz,
                     self._spin_sx, self._spin_sy, self._spin_sz):
            spin.blockSignals(False)

    def _read_values(self) -> SceneCorrection:
        return SceneCorrection(
            rotate_x=self._spin_rx.value(),
            rotate_y=self._spin_ry.value(),
            rotate_z=self._spin_rz.value(),
            shift_x=self._spin_sx.value(),
            shift_y=self._spin_sy.value(),
            shift_z=self._spin_sz.value(),
        )

    def _on_value_changed(self):
        c = self._read_values()
        self._correction = c
        self.correction_changed.emit(c)

    # ---- actions ----

    def _on_reset(self):
        c = SceneCorrection()
        self._load_values(c)
        self._correction = c
        self.correction_changed.emit(c)

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Correction YAML", self._scene_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return
        try:
            c = SceneCorrection.load_yaml(path)
            self._load_values(c)
            self._correction = c
            self.correction_changed.emit(c)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Correction YAML", self._scene_dir + "/correction.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return
        try:
            self._correction.save_yaml(path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    @property
    def correction(self) -> SceneCorrection:
        return self._correction
