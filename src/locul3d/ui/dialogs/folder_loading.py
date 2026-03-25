"""Folder loading dialog with per-file progress — mirrors E57ProgressDialog style."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QTextEdit, QScrollArea, QWidget,
)

from ...core.constants import COLORS, AUTO_LAYER_COLORS
from ...core.layer import LayerData
from ...utils.io import load_geometry


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class FolderLoadWorker(QThread):
    """Background thread that loads geometry files one-by-one."""

    file_started = Signal(str, int, int)   # filename, index, total
    file_done = Signal(str, int, int)      # filename, index, total
    log_message = Signal(str)
    finished_ok = Signal(list)             # list[LayerData]
    finished_err = Signal(str)

    def __init__(self, file_paths: list, parent=None):
        super().__init__(parent)
        self._file_paths = file_paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        layers: list = []
        total = len(self._file_paths)
        for i, path_str in enumerate(self._file_paths):
            if self._cancelled:
                self.finished_err.emit("Cancelled by user")
                return

            p = Path(path_str)
            name = p.name
            self.file_started.emit(name, i, total)

            try:
                ext = p.suffix.lower()
                layer_type = "mesh" if ext in (".obj", ".stl") else "pointcloud"
                color_idx = i % len(AUTO_LAYER_COLORS)
                auto_color = AUTO_LAYER_COLORS[color_idx] + [1.0]

                layer_def = {
                    "id": f"file_{i}",
                    "name": name,
                    "type": layer_type,
                    "file": name,
                    "visible": True,
                    "opacity": 1.0,
                    "color": auto_color,
                }
                layer = LayerData(layer_def, str(p.parent))
                load_geometry(path_str, layer)
                layer.loaded = True

                if (layer.layer_type == "wireframe"
                        and layer.colors is not None
                        and len(layer.colors) > 0):
                    median_rgb = np.median(layer.colors[:, :3], axis=0)
                    layer.color = median_rgb.tolist() + [1.0]

                layers.append(layer)
                pts = layer.point_count
                self.log_message.emit(f"  {name}  ({pts:,} pts)")
            except Exception as e:
                self.log_message.emit(f"  Error: {name} — {e}")

            self.file_done.emit(name, i, total)

        self.finished_ok.emit(layers)


# ---------------------------------------------------------------------------
# Modal progress dialog
# ---------------------------------------------------------------------------

class FolderProgressDialog(QDialog):
    """Modal dialog with per-file progress for folder loading."""

    def __init__(self, folder_path: str, file_names: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Folder")
        self.setMinimumSize(560, 480)
        self.setModal(True)
        self._result: Optional[list] = None
        self._worker: Optional[FolderLoadWorker] = None
        self._start_time = 0.0
        self._file_names = file_names
        self._setup_ui(folder_path, file_names)

    # ---- UI setup ---------------------------------------------------------

    def _setup_ui(self, folder_path: str, file_names: list):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel(f"Loading: {Path(folder_path).name}")
        header.setStyleSheet(
            f"font-size: 14px; font-weight: bold; "
            f"color: {COLORS.get('text', '#eee')};")
        layout.addWidget(header)

        # --- Per-file stage rows (scrollable) ---
        self._file_widgets = {}

        self._scroll = QScrollArea()
        scroll = self._scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        stages_container = QWidget()
        stages_container.setStyleSheet(
            f"background: {COLORS.get('input_bg', '#222')}; "
            f"border-radius: 6px;")
        stages_layout = QVBoxLayout(stages_container)
        stages_layout.setContentsMargins(8, 8, 8, 8)
        stages_layout.setSpacing(6)

        for fname in file_names:
            row = QHBoxLayout()
            row.setSpacing(8)

            icon_label = QLabel("  ")
            icon_label.setFixedWidth(20)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet(
                f"color: {COLORS.get('hover', '#555')}; font-size: 12px;")
            row.addWidget(icon_label)

            name_label = QLabel(fname)
            name_label.setFixedWidth(220)
            name_label.setStyleSheet(
                f"color: {COLORS.get('text_muted', '#888')}; font-size: 12px;")
            row.addWidget(name_label)

            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFixedHeight(14)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {COLORS.get('hover', '#555')}; "
                f"border-radius: 3px; }}")
            row.addWidget(pbar, stretch=1)

            stages_layout.addLayout(row)
            self._file_widgets[fname] = (icon_label, name_label, pbar)

        scroll.setWidget(stages_container)
        # Cap height so the scroll area doesn't dominate the dialog
        scroll.setMaximumHeight(220)
        layout.addWidget(scroll)

        # --- Elapsed time ---
        self._time_label = QLabel("Elapsed: 0.0s")
        self._time_label.setStyleSheet(
            f"color: {COLORS.get('text_muted', '#888')}; font-size: 11px;")
        layout.addWidget(self._time_label)

        # --- Log view ---
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(160)
        self._log_view.setStyleSheet(
            f"QTextEdit {{ background: {COLORS.get('bg', '#111')}; "
            f"color: {COLORS.get('text_secondary', '#aaa')}; "
            f"font-family: monospace; font-size: 11px; "
            f"border: 1px solid {COLORS.get('border', '#444')}; "
            f"border-radius: 4px; padding: 4px; }}")
        layout.addWidget(self._log_view)

        # --- Cancel / Close ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        # --- Elapsed timer ---
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(100)

        self._current_file: Optional[str] = None

    # ---- Public API -------------------------------------------------------

    def start(self, worker: FolderLoadWorker):
        self._worker = worker
        self._start_time = time.time()

        worker.file_started.connect(self._on_file_started)
        worker.file_done.connect(self._on_file_done)
        worker.log_message.connect(self._on_log)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.finished_err.connect(self._on_finished_err)

        self._on_log(f"Loading {len(self._file_names)} files...")
        QTimer.singleShot(50, worker.start)

    def get_result(self) -> Optional[list]:
        return self._result

    # ---- Signal handlers --------------------------------------------------

    def _on_file_started(self, filename: str, index: int, total: int):
        # Mark previous file done
        if self._current_file and self._current_file in self._file_widgets:
            self._mark_file_done(self._current_file)

        self._current_file = filename
        if filename in self._file_widgets:
            icon, name, pbar = self._file_widgets[filename]
            icon.setText(">>")
            icon.setStyleSheet(
                f"color: {COLORS.get('accent', '#36f')}; "
                f"font-size: 11px; font-weight: bold;")
            name.setStyleSheet(
                f"color: {COLORS.get('text', '#eee')}; "
                f"font-size: 12px; font-weight: bold;")
            pbar.setValue(0)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {COLORS.get('accent', '#36f')}; "
                f"border-radius: 3px; }}")
            # Auto-scroll to keep current file visible
            self._scroll.ensureWidgetVisible(icon)

    def _on_file_done(self, filename: str, index: int, total: int):
        if filename in self._file_widgets:
            self._mark_file_done(filename)

    def _mark_file_done(self, filename: str):
        if filename in self._file_widgets:
            icon, name, pbar = self._file_widgets[filename]
            icon.setText("OK")
            icon.setStyleSheet(
                "color: #40c040; font-size: 11px; font-weight: bold;")
            name.setStyleSheet("color: #80c080; font-size: 12px;")
            pbar.setValue(100)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: #40c040; "
                f"border-radius: 3px; }}")

    def _on_log(self, msg: str):
        self._log_view.append(msg)
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_view.setTextCursor(cursor)

    def _on_finished_ok(self, layers: list):
        if self._current_file:
            self._mark_file_done(self._current_file)
        elapsed = time.time() - self._start_time
        total_pts = sum(l.point_count for l in layers)
        self._time_label.setText(f"Completed in {elapsed:.1f}s")
        self._on_log(
            f"\nLoaded {len(layers)} files "
            f"({total_pts:,} points) in {elapsed:.1f}s")
        self._result = layers
        self._timer.stop()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)
        QTimer.singleShot(600, self.accept)

    def _on_finished_err(self, error: str):
        elapsed = time.time() - self._start_time
        self._time_label.setText(f"Failed after {elapsed:.1f}s")
        if self._current_file and self._current_file in self._file_widgets:
            icon, name, pbar = self._file_widgets[self._current_file]
            icon.setText("!!")
            icon.setStyleSheet(
                "color: #ff4040; font-size: 11px; font-weight: bold;")
            name.setStyleSheet("color: #ff6060; font-size: 12px;")
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: #ff4040; "
                f"border-radius: 3px; }}")
        self._on_log(f"\nERROR: {error}")
        self._timer.stop()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._on_log("Cancelling...")
            self._worker.cancel()
            self._worker.wait(3000)
        self.reject()

    def _update_elapsed(self):
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            self._time_label.setText(f"Elapsed: {elapsed:.1f}s")
