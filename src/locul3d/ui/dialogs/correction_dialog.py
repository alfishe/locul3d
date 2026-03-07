"""Scene Correction dialog — non-modal, live preview, auto-detect, reset."""

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar,
)

import numpy as np

from ...core.correction import SceneCorrection


class _AutoDetectWorker(QThread):
    """Run scene correction analysis in a background thread."""

    finished = Signal(SceneCorrection, object)  # (correction, diagnostics)
    error = Signal(str)

    def __init__(self, points: np.ndarray, parent=None):
        super().__init__(parent)
        self._points = points

    def run(self):
        try:
            from ...analysis.scene_correction import auto_detect_correction
            corr, diag = auto_detect_correction(self._points)
            self.finished.emit(corr, diag)
        except Exception as e:
            self.error.emit(str(e))


class CorrectionDialog(QDialog):
    """Non-modal dialog for editing scene correction (rotations + shifts).

    Features:
      - Live preview: all spinner changes immediately update the viewport
      - Reset: restore the correction that was active when the dialog opened
      - Auto-Detect: analyze the loaded point cloud to find floor/wall alignment
      - Save to Project: writes correction to the unified YAML project file

    Emits *correction_changed* whenever any value changes.
    Emits *save_requested* when the user clicks Save to Project.
    """

    correction_changed = Signal(SceneCorrection)
    save_requested = Signal()
    diagnostics_ready = Signal(object)  # CorrectionDiagnostics

    def __init__(self, correction: SceneCorrection, scene_dir: str = "",
                 parent=None, point_source=None):
        """
        Args:
            correction: Current scene correction values.
            scene_dir: Directory of the scene file (for file dialogs).
            parent: Parent widget.
            point_source: Callable returning (N,3) ndarray of scene points
                          for auto-detection, or None to disable the button.
        """
        super().__init__(parent)
        self.setWindowTitle("Scene Correction")
        self.setMinimumWidth(380)
        # Non-modal: user can interact with viewport while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._scene_dir = scene_dir
        self._correction = correction
        self._initial_correction = SceneCorrection(
            rotate_x=correction.rotate_x, rotate_y=correction.rotate_y,
            rotate_z=correction.rotate_z, shift_x=correction.shift_x,
            shift_y=correction.shift_y, shift_z=correction.shift_z,
        )
        self._point_source = point_source
        self._worker = None

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

        # --- Auto-detect ---
        auto_row = QHBoxLayout()
        self._btn_auto = QPushButton("⚡ Auto-Detect")
        self._btn_auto.setToolTip(
            "Analyze point cloud to find floor plane (→Z=0) "
            "and wall alignment (→axis-aligned X/Y)")
        self._btn_auto.clicked.connect(self._on_auto_detect)
        self._btn_auto.setEnabled(self._point_source is not None)
        auto_row.addWidget(self._btn_auto)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(18)
        self._progress.hide()
        auto_row.addWidget(self._progress)
        layout.addLayout(auto_row)

        # --- Action buttons ---
        btn_row = QHBoxLayout()

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Restore the correction from when the dialog was opened")
        btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(btn_reset)

        btn_zero = QPushButton("Zero")
        btn_zero.setToolTip("Set all values to zero (identity correction)")
        btn_zero.clicked.connect(self._on_zero)
        btn_row.addWidget(btn_zero)

        btn_row.addStretch()

        btn_load = QPushButton("Load…")
        btn_load.clicked.connect(self._on_load)
        btn_row.addWidget(btn_load)

        btn_save_project = QPushButton("💾 Save to Project")
        btn_save_project.setToolTip("Save correction into the project YAML file")
        btn_save_project.clicked.connect(self._on_save_project)
        btn_row.addWidget(btn_save_project)

        layout.addLayout(btn_row)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

    def _make_spin(self, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(4)
        spin.setKeyboardTracking(False)
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
        """Restore the correction from when the dialog opened."""
        self._load_values(self._initial_correction)
        self._correction = SceneCorrection(
            rotate_x=self._initial_correction.rotate_x,
            rotate_y=self._initial_correction.rotate_y,
            rotate_z=self._initial_correction.rotate_z,
            shift_x=self._initial_correction.shift_x,
            shift_y=self._initial_correction.shift_y,
            shift_z=self._initial_correction.shift_z,
        )
        self.correction_changed.emit(self._correction)

    def _on_zero(self):
        """Set all values to zero."""
        c = SceneCorrection()
        self._load_values(c)
        self._correction = c
        self.correction_changed.emit(c)

    def _on_auto_detect(self):
        """Run auto-detection in a background thread."""
        if self._point_source is None:
            return
        points = self._point_source()
        if points is None or len(points) < 100:
            QMessageBox.warning(
                self, "Auto-Detect",
                "Not enough points loaded for auto-detection (need ≥100).")
            return

        self._btn_auto.setEnabled(False)
        self._btn_auto.setText("Analyzing…")
        self._progress.show()

        self._worker = _AutoDetectWorker(points, self)
        self._worker.finished.connect(self._on_auto_detect_done)
        self._worker.error.connect(self._on_auto_detect_error)
        self._worker.start()

    def _on_auto_detect_done(self, corr: SceneCorrection, diag):
        """Apply auto-detected correction values and emit diagnostics."""
        self._progress.hide()
        self._btn_auto.setEnabled(True)
        self._btn_auto.setText("⚡ Auto-Detect")
        self._worker = None

        self._load_values(corr)
        self._correction = corr
        self.correction_changed.emit(corr)
        self.diagnostics_ready.emit(diag)

    def _on_auto_detect_error(self, msg: str):
        self._progress.hide()
        self._btn_auto.setEnabled(True)
        self._btn_auto.setText("⚡ Auto-Detect")
        self._worker = None
        QMessageBox.warning(self, "Auto-Detect Error", msg)

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

    def _on_save_project(self):
        """Request save to the unified project file."""
        self.save_requested.emit()

    @property
    def correction(self) -> SceneCorrection:
        return self._correction
