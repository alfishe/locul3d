"""Locul3D Viewer — main window with full 3D viewing workflow."""

import sys
import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QStatusBar,
    QLabel, QSlider, QComboBox, QDockWidget, QFileDialog,
    QApplication, QMessageBox,
)
from PySide6.QtGui import QAction, QKeyEvent

from ..core.layer import LayerManager, LayerData
from ..core.constants import COLORS
from ..core.correction import SceneCorrection
from ..rendering.gl.viewport import BaseGLViewport
from ..ui.themes import ThemeManager
from ..ui.widgets.layers import LayerPanel
from ..ui.widgets.info import InfoPanel
from ..ui.dialogs.correction_dialog import CorrectionDialog
from ..ui.dialogs.scene_dialog import SceneDialog


class ViewerWindow(QMainWindow):
    """Locul3D Viewer — point cloud / mesh / wireframe / E57 viewer."""

    def __init__(self, files=None, correction_angles=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Locul3D — Viewer")
        self.resize(1280, 800)

        # Theme
        self.theme = ThemeManager()
        self.setStyleSheet(self.theme.get_stylesheet())

        # Core data
        self.layer_manager = LayerManager()
        self.viewport = BaseGLViewport(self.layer_manager, self)

        # Apply CLI correction angles if provided
        if correction_angles:
            sc = self.viewport.scene_correction
            sc.rotate_x = correction_angles.get('rotate_x', 0.0)
            sc.rotate_y = correction_angles.get('rotate_y', 0.0)
            sc.rotate_z = correction_angles.get('rotate_z', 0.0)
            sc.shift_x = correction_angles.get('shift_x', 0.0)
            sc.shift_y = correction_angles.get('shift_y', 0.0)
            sc.shift_z = correction_angles.get('shift_z', 0.0)
        self._cli_correction = correction_angles or {}

        self._setup_ui()
        self._setup_toolbar()
        self._setup_sidebar()
        self._setup_statusbar()

        # File-watch timer
        self._file_watch_timer = QTimer(self)
        self._file_watch_timer.timeout.connect(self._check_file_changes)
        self._file_watch_timer.start(2000)

        # Wire signals
        self.viewport.fps_updated.connect(self._on_fps_updated)
        self.layer_panel.layer_changed.connect(self._on_layer_changed)
        self.layer_panel.layer_selected.connect(self._on_layer_selected)
        self.layer_panel.pano_requested.connect(self._on_pano_requested)

        # Marker click in viewport → select in layer panel (no info panel)
        self.viewport.marker_selected.connect(
            self.layer_panel.select_layer_by_data)
        # Marker double-click → select + open info panel
        self.viewport.marker_activated.connect(
            lambda layer: self.layer_panel.select_layer_by_data(layer, notify=True))

        self._selected_layer = None

        # Deferred file loading
        if files:
            self._deferred_files = files
            QTimer.singleShot(100, self._deferred_load)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewport)
        self.setCentralWidget(central)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File actions
        act_open = QAction("Open File", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_file)
        toolbar.addAction(act_open)

        act_open_folder = QAction("Open Folder", self)
        act_open_folder.setShortcut("Ctrl+Shift+O")
        act_open_folder.triggered.connect(self._on_open_folder)
        toolbar.addAction(act_open_folder)

        # E57 Import (first-class)
        act_e57 = QAction("Import E57", self)
        act_e57.setToolTip("Import E57 point cloud with processing pipeline")
        act_e57.triggered.connect(self._on_import_e57)
        toolbar.addAction(act_e57)

        toolbar.addSeparator()

        # View toggles
        self.act_layer_colors = QAction("Layer Colors", self, checkable=True, checked=True)
        self.act_layer_colors.triggered.connect(self._on_toggle_layer_colors)
        toolbar.addAction(self.act_layer_colors)

        self.act_axes = QAction("Axes", self, checkable=True, checked=True)
        self.act_axes.triggered.connect(lambda c: self._toggle_view("show_axes", c))
        toolbar.addAction(self.act_axes)

        self.act_grid = QAction("Grid", self, checkable=True, checked=True)
        self.act_grid.triggered.connect(lambda c: self._toggle_view("show_grid", c))
        toolbar.addAction(self.act_grid)

        toolbar.addSeparator()

        # Point size
        toolbar.addWidget(QLabel("  Pt:"))
        self.pt_slider = QSlider(Qt.Orientation.Horizontal)
        self.pt_slider.setRange(1, 20)
        self.pt_slider.setValue(2)
        self.pt_slider.setFixedWidth(80)
        self.pt_slider.valueChanged.connect(self._on_point_size)
        toolbar.addWidget(self.pt_slider)

        toolbar.addSeparator()

        # Camera presets
        cam_combo = QComboBox()
        cam_combo.addItems(["Perspective", "Top", "Front", "Right", "Isometric"])
        cam_combo.currentTextChanged.connect(self._on_camera_preset)
        toolbar.addWidget(cam_combo)

        act_fit = QAction("Fit All", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.viewport.fit_to_scene)
        toolbar.addAction(act_fit)

        toolbar.addSeparator()

        act_screenshot = QAction("Screenshot", self)
        act_screenshot.setToolTip("Save viewport screenshot")
        act_screenshot.triggered.connect(self._on_screenshot)
        toolbar.addAction(act_screenshot)

        act_scene = QAction("Scene", self)
        act_scene.setToolTip("Scene bounds, ceiling clipping, dimensions")
        act_scene.triggered.connect(self._on_scene)
        toolbar.addAction(act_scene)

        act_correction = QAction("Scene Correction", self)
        act_correction.setToolTip("Adjust scene rotation and shift for axis alignment")
        act_correction.triggered.connect(self._on_scene_correction)
        toolbar.addAction(act_correction)

    def _setup_sidebar(self):
        # Layers dock
        self._layers_dock = QDockWidget("Layers", self)
        self._layers_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._layers_dock.setMinimumWidth(280)
        self.layer_panel = LayerPanel(self.layer_manager)
        self._layers_dock.setWidget(self.layer_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layers_dock)

        # Info dock
        self._info_dock = QDockWidget("Info", self)
        self._info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._info_dock.setMinimumWidth(280)
        self.info_panel = InfoPanel()
        self._info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

        self.tabifyDockWidget(self._layers_dock, self._info_dock)
        self._layers_dock.raise_()

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.cam_label = QLabel("Cam: --")
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.cam_label)
        self.status_bar.addPermanentWidget(self.fps_label)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _deferred_load(self):
        files = getattr(self, '_deferred_files', [])
        for arg in files:
            p = Path(arg)
            if p.is_dir():
                self._load_folder(str(p))
            elif p.is_file():
                if p.suffix.lower() == '.e57':
                    self._import_e57_file(str(p))
                else:
                    self._load_file(str(p), fit_camera=False)
        self.viewport.fit_to_scene()
        # Compute ceiling height silently in background (cached, not applied)
        QTimer.singleShot(0, self.layer_manager.compute_ceiling_background)
        # Auto-detect sidecar correction YAML
        if files:
            self._try_load_sidecar(files[0])

    def _on_open_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Files", "",
            "3D Files (*.ply *.obj *.stl *.e57);;All Files (*)")
        for p in paths:
            if Path(p).suffix.lower() == '.e57':
                self._import_e57_file(p)
            else:
                self._load_file(p)
        # Auto-detect sidecar for first opened file
        if paths:
            self._try_load_sidecar(paths[0])

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self._load_folder(folder)

    def _on_import_e57(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import E57 File", "",
            "E57 Files (*.e57);;All Files (*)")
        if path:
            self._import_e57_file(path)

    def _import_e57_file(self, path: str):
        """Import E57 file through the full processing pipeline."""
        # --- Correction sidecar detection ---
        from ..core.correction import SceneCorrection
        scene_p = Path(path).resolve()
        print(f"[E57] Searching for correction sidecar in: {scene_p.parent}/")
        sidecar = SceneCorrection.find_sidecar(str(scene_p))
        correction = None
        if sidecar:
            try:
                correction = SceneCorrection.load_yaml(sidecar)
                print(f"[E57] Correction LOADED from: {sidecar}")
                print(f"[E57]   rot=({correction.rotate_x}, {correction.rotate_y}, {correction.rotate_z})°  "
                      f"shift=({correction.shift_x}, {correction.shift_y}, {correction.shift_z})")
            except Exception as e:
                print(f"[E57] WARNING: failed to load correction sidecar: {e}")
                correction = None
        else:
            print(f"[E57] No correction sidecar found for: {scene_p.name}")

        try:
            from ..plugins.importers.e57 import (
                E57ImportWorker, E57ProgressDialog, E57Importer,
            )
        except ImportError as e:
            QMessageBox.warning(self, "E57 Import",
                                f"E57 import dependencies not available:\n{e}")
            return

        importer = E57Importer()
        if not importer.is_available():
            QMessageBox.warning(self, "E57 Import",
                                "E57 import requires pye57 and open3d.\n"
                                "Install: pip install pye57 open3d")
            return

        worker = importer.create_worker(path, self)
        dialog = importer.create_dialog(path, self)
        dialog.start(worker)
        result_code = dialog.exec()

        result = dialog.get_result()
        if result and result.layers:
            # Bake correction into point data (scene coords → global coords)
            if correction and not correction.is_identity:
                import numpy as np
                for layer in result.layers:
                    if layer.points is not None and len(layer.points) > 0:
                        layer.points = correction.bake_points(layer.points).astype(np.float32)
                    if hasattr(layer, 'pano_position') and layer.pano_position is not None:
                        pos = correction.transform_point(layer.pano_position)
                        layer.pano_position = pos.tolist()

            for layer in result.layers:
                self.layer_manager.layers.append(layer)
            self.layer_panel.rebuild()
            self.viewport.fit_to_scene()

            if result.metadata or result.stats:
                self.info_panel.populate(result.metadata, result.stats)
                self._info_dock.raise_()

            n_layers = len(result.layers)
            total_pts = sum(l.point_count for l in result.layers)
            self.status_label.setText(
                f"Imported E57: {n_layers} layers, {total_pts:,} points")
            self.setWindowTitle(f"Locul3D Viewer — {Path(path).name}")

    def _load_file(self, path: str, fit_camera: bool = True):
        self.status_label.setText(f"Loading {Path(path).name}...")
        QApplication.processEvents()
        try:
            self.layer_manager.load_single_file(path)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            return
        self.layer_panel.rebuild()
        if fit_camera:
            self.viewport.fit_to_scene()
        last = self.layer_manager.layers[-1] if self.layer_manager.layers else None
        if last:
            self.info_panel.show_layer_info(last)
        self.status_label.setText(f"Loaded {Path(path).name}")
        # Auto-detect correction sidecar
        self._try_load_sidecar(path)

    def _load_folder(self, folder: str):
        folder_path = Path(folder)
        for ext in ("*.ply", "*.obj", "*.stl"):
            for p in sorted(folder_path.glob(ext)):
                self._load_file(str(p), fit_camera=False)
        # E57 files auto-routed through pipeline
        for p in sorted(folder_path.glob("*.e57")):
            self._import_e57_file(str(p))

        # layers.json manifest
        manifest_path = folder_path / "layers.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as mf:
                    manifest_entries = json.load(mf)
                manifest_map = {e["file"]: e for e in manifest_entries}
                for layer in self.layer_manager.layers:
                    entry = manifest_map.get(layer.name)
                    if entry:
                        layer.color = entry.get("color")
                        if "name" in entry:
                            layer.name = entry["name"]
                        if "visible" in entry:
                            layer.visible = entry["visible"]
                        layer.evict_byte_caches()
                self.viewport.use_layer_colors = True
                self.act_layer_colors.setChecked(True)
                self.layer_panel.rebuild()
            except Exception:
                pass
        self.viewport.fit_to_scene()
        self.status_label.setText(f"Loaded folder: {folder_path.name}")

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------

    def _on_toggle_layer_colors(self, checked):
        self.viewport.use_layer_colors = checked
        for l in self.layer_manager.layers:
            l.evict_byte_caches()
            self.viewport.delete_vbos_for_layer(l.id)
        self.viewport.update()

    def _toggle_view(self, attr, checked):
        setattr(self.viewport, attr, checked)
        self.viewport.update()

    def _on_point_size(self, val):
        self.viewport.point_size = val
        self.viewport.update()

    def _on_camera_preset(self, name):
        vp = self.viewport
        presets = {"Top": (0, 89), "Front": (0, 0), "Right": (90, 0), "Isometric": (45, 30)}
        if name in presets:
            vp.cam_azimuth, vp.cam_elevation = presets[name]
            vp.update()

    def _on_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "Images (*.png *.jpg *.bmp);;All Files (*)")
        if path:
            pixmap = self.viewport.grab()
            pixmap.save(path)
            self.status_label.setText(f"Screenshot saved: {Path(path).name}")

    def _on_scene(self):
        """Open the non-modal Scene dialog for bounds & ceiling clipping."""
        dlg = SceneDialog(self.layer_manager, self.viewport, self)
        dlg.clip_changed.connect(self._apply_scene_clip)
        dlg.show()  # non-modal
        self._scene_dialog = dlg  # prevent GC

    def _apply_scene_clip(self, x0, x1, y0, y1, z0, z1):
        """Apply scene clip bounds from the Scene dialog."""
        self.viewport.scene_clip = (x0, x1, y0, y1, z0, z1)
        self.viewport.update()
        self.status_label.setText(
            f"Scene clip: X=[{x0:.1f},{x1:.1f}] Y=[{y0:.1f},{y1:.1f}] Z=[{z0:.1f},{z1:.1f}]")

    def _on_scene_correction(self):
        """Open the Scene Correction dialog for live rotation/shift adjustment."""
        scene_dir = self.layer_manager.base_dir or ""
        dlg = CorrectionDialog(self.viewport.scene_correction, scene_dir, self)
        dlg.correction_changed.connect(self._apply_correction)
        dlg.exec()

    def _apply_correction(self, c: SceneCorrection):
        """Apply correction values from dialog to viewport (live preview)."""
        self.viewport.scene_correction = c
        self.viewport.update()
        if not c.is_identity:
            self.status_label.setText(
                f"Correction: rot=({c.rotate_x:.1f}, {c.rotate_y:.1f}, {c.rotate_z:.1f})°  "
                f"shift=({c.shift_x:.2f}, {c.shift_y:.2f}, {c.shift_z:.2f})")
        else:
            self.status_label.setText("Correction reset")

    def _try_load_sidecar(self, scene_path: str):
        """Auto-detect and load a correction YAML sidecar next to a scene file."""
        p = Path(scene_path).resolve()  # resolve to absolute path
        sidecar = SceneCorrection.find_sidecar(str(p))
        if sidecar is None:
            return
        try:
            c = SceneCorrection.load_yaml(sidecar)
            # CLI args override sidecar values (non-zero CLI wins)
            cli = self._cli_correction
            if cli.get('rotate_x', 0): c.rotate_x = cli['rotate_x']
            if cli.get('rotate_y', 0): c.rotate_y = cli['rotate_y']
            if cli.get('rotate_z', 0): c.rotate_z = cli['rotate_z']
            if cli.get('shift_x', 0): c.shift_x = cli['shift_x']
            if cli.get('shift_y', 0): c.shift_y = cli['shift_y']
            if cli.get('shift_z', 0): c.shift_z = cli['shift_z']
            self.viewport.scene_correction = c
            self.viewport.update()
            print(f"Scene correction loaded from: {sidecar}")
            print(f"  rot=({c.rotate_x}, {c.rotate_y}, {c.rotate_z})°  "
                  f"shift=({c.shift_x}, {c.shift_y}, {c.shift_z})")
            self.status_label.setText(f"Correction loaded: {Path(sidecar).name}")
        except Exception as e:
            print(f"Warning: failed to load correction sidecar: {e}")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_pano_requested(self, layer):
        """Enter 360° panorama mode and highlight the active row."""
        self.viewport.enter_panorama(layer)
        self.layer_panel.highlight_active_pano(layer)

    def _on_layer_changed(self):
        self.viewport.update()

    def _on_layer_selected(self, layer_data):
        self._selected_layer = layer_data
        self.info_panel.show_layer_info(layer_data)
        self._info_dock.raise_()

    def _on_fps_updated(self, fps):
        vp = self.viewport
        self.cam_label.setText(
            f"Az:{vp.cam_azimuth%360:.0f} El:{vp.cam_elevation:.0f} D:{vp.cam_distance:.1f}")
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def _check_file_changes(self):
        any_changed = False
        for layer in self.layer_manager.layers:
            if layer.file_changed_on_disk():
                try:
                    layer.reload()
                    self.viewport.delete_vbos_for_layer(layer.id)
                    any_changed = True
                except Exception:
                    pass
        if any_changed:
            self.layer_panel.rebuild()
            self.viewport.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if (hasattr(self.viewport, '_panorama')
                    and self.viewport._panorama
                    and self.viewport._panorama.is_active):
                self.viewport.exit_panorama()
                self.layer_panel.highlight_active_pano(None)
                return
        super().keyPressEvent(event)
