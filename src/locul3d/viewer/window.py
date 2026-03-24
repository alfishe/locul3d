"""Locul3D Viewer — main window with full 3D viewing workflow."""

import sys
import json
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QStatusBar,
    QLabel,
    QSlider,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QApplication,
    QMessageBox,
    QToolButton,
    QMenu,
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
    """Locul3D Viewer — read-only 3D scene viewer.

    Supports point clouds (.ply), meshes (.obj, .stl), and E57 scan files.
    Provides panorama viewing, layer management, scene correction, and
    scene clipping via the Scene dialog.

    Architecture:
        - viewport:       BaseGLViewport — OpenGL rendering widget
        - layer_manager:  LayerManager   — owns all loaded LayerData
        - layer_panel:    LayerPanel     — sidebar for visibility/opacity
        - info_panel:     InfoPanel      — metadata display for selected layer
        - theme:          ThemeManager   — dark/light stylesheet generation

    Lifecycle:
        1. __init__ creates UI, wires signals, optionally defers file loading
        2. _deferred_load runs after event loop starts (100ms delay)
        3. _post_load triggers background AABB/ceiling caching after any load
        4. File-watch timer polls every 2s for on-disk changes (hot reload)
    """

    def __init__(self, files=None, correction_angles=None, parent=None):
        """Initialise the Viewer window.

        Args:
            files:             List of file/folder paths to load on startup.
                               Loading is deferred 100ms to let the event loop start.
            correction_angles: Dict with rotate_x/y/z and shift_x/y/z overrides
                               (applied to viewport.scene_correction on startup).
            parent:            Optional parent QWidget.
        """
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
            sc.rotate_x = correction_angles.get("rotate_x", 0.0)
            sc.rotate_y = correction_angles.get("rotate_y", 0.0)
            sc.rotate_z = correction_angles.get("rotate_z", 0.0)
            sc.shift_x = correction_angles.get("shift_x", 0.0)
            sc.shift_y = correction_angles.get("shift_y", 0.0)
            sc.shift_z = correction_angles.get("shift_z", 0.0)
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
        self.viewport.marker_selected.connect(self.layer_panel.select_layer_by_data)
        # Marker double-click → select + open info panel
        self.viewport.marker_activated.connect(
            lambda layer: self.layer_panel.select_layer_by_data(layer, notify=True)
        )

        self._selected_layer = None

        # Deferred file loading
        if files:
            self._deferred_files = files
            QTimer.singleShot(100, self._deferred_load)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Create the central widget layout with the GL viewport filling it."""
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewport)
        self.setCentralWidget(central)

    def _setup_toolbar(self):
        """Build the main toolbar with file, view, camera, and tool actions.

        Toolbar items (left to right):
          File:   Open File, Open Folder, Import E57
          View:   Layer Colors (checkable), Axes, Grid
          Size:   Point size slider (1-20)
          Camera: Preset dropdown, Fit All, Screenshot
          Tools:  Scene, Scene Correction
        """
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

        act_clear = QAction("Clear Scene", self)
        act_clear.setToolTip("Remove all layers from the scene")
        act_clear.triggered.connect(self._on_clear_scene)
        toolbar.addAction(act_clear)

        # E57 Import (first-class)
        act_e57 = QAction("Import E57", self)
        act_e57.setToolTip("Import E57 point cloud with processing pipeline")
        act_e57.triggered.connect(self._on_import_e57)
        toolbar.addAction(act_e57)

        toolbar.addSeparator()

        # View toggles
        self.act_layer_colors = QAction(
            "Layer Colors", self, checkable=True, checked=True
        )
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
        self.pt_slider.setRange(1, 10)
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

        toolbar.addSeparator()
        exp_btn = QToolButton()
        exp_btn.setText("Experimental")
        exp_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        exp_menu = QMenu(self)
        self.act_fps_movement = QAction("FPS Movement", self, checkable=True)
        self.act_fps_movement.setToolTip(
            "WASD/QE moves camera instead of scene correction"
        )
        self.act_fps_movement.triggered.connect(
            lambda c: self._toggle_view("fps_movement", c)
        )
        exp_menu.addAction(self.act_fps_movement)
        self.act_fps_camera = QAction("FPS Camera", self, checkable=True)
        self.act_fps_camera.setToolTip(
            "First-person camera: mouselook + WASD walking (collapses orbit distance)"
        )
        self.act_fps_camera.triggered.connect(self._on_fps_camera_toggled)
        exp_menu.addAction(self.act_fps_camera)
        self.act_pt_attenuation = QAction("Perspective Point Attenuation", self, checkable=True)
        self.act_pt_attenuation.setToolTip(
            "Scale point size by 1/distance so points shrink naturally as camera moves away"
        )
        self.act_pt_attenuation.triggered.connect(
            lambda c: self._toggle_view("point_attenuation", c)
        )
        exp_menu.addAction(self.act_pt_attenuation)
        self.act_auto_scale = QAction(
            "Auto-Scale Small Points", self, checkable=True, checked=True
        )
        self.act_auto_scale.setToolTip(
            "Automatically enlarge points for small/sparse layers (disable for uniform size)"
        )
        self.act_auto_scale.triggered.connect(
            lambda c: self._toggle_view("auto_scale_small_points", c)
        )
        exp_menu.addAction(self.act_auto_scale)
        exp_btn.setMenu(exp_menu)
        toolbar.addWidget(exp_btn)

    def _setup_sidebar(self):
        """Create right-side dock widgets: Layers panel and Info panel.

        Layers and Info are tabified; Layers tab is raised by default.
        """
        # Layers dock
        self._layers_dock = QDockWidget("Layers", self)
        self._layers_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._layers_dock.setMinimumWidth(280)
        self.layer_panel = LayerPanel(self.layer_manager)
        self._layers_dock.setWidget(self.layer_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layers_dock)

        # Info dock
        self._info_dock = QDockWidget("Info", self)
        self._info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._info_dock.setMinimumWidth(280)
        self.info_panel = InfoPanel()
        self._info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

        self.tabifyDockWidget(self._layers_dock, self._info_dock)
        self._layers_dock.raise_()

    def _setup_statusbar(self):
        """Create status bar with status text, camera info, and FPS counter."""
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

    def _post_load(self):
        """Called after any geometry load — refresh caches in background."""
        self.layer_manager.invalidate_scene_aabb()
        QTimer.singleShot(0, self.layer_manager.compute_ceiling_background)

    def _deferred_load(self):
        """Load files passed via __init__ after the event loop starts.

        Directories are loaded via _load_folder, .e57 via _import_e57_file,
        all other files via _load_file.  After loading, fits the camera,
        triggers _post_load, and auto-detects correction sidecars.
        """
        files = getattr(self, "_deferred_files", [])
        for arg in files:
            p = Path(arg)
            if p.is_dir():
                self._load_folder(str(p))
            elif p.is_file():
                if p.suffix.lower() == ".e57":
                    self._import_e57_file(str(p))
                else:
                    self._load_file(str(p), fit_camera=False)
        self.viewport.fit_to_scene()
        self._post_load()
        # Auto-detect sidecar correction YAML
        if files:
            self._try_load_sidecar(files[0])

    def _on_open_file(self):
        """Show file dialog to open one or more 3D files (.ply/.obj/.stl/.e57).

        E57 files are routed through the full import pipeline with progress.
        Other files are loaded directly. Auto-detects correction sidecar for
        the first opened file.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Files", "", "3D Files (*.ply *.obj *.stl *.e57);;All Files (*)"
        )
        for p in paths:
            if Path(p).suffix.lower() == ".e57":
                self._import_e57_file(p)
            else:
                self._load_file(p)
        # Auto-detect sidecar for first opened file
        if paths:
            self._try_load_sidecar(paths[0])

    def _on_open_folder(self):
        """Show folder dialog and load all supported files from selected directory."""
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self._load_folder(folder)

    def _on_import_e57(self):
        """Show file dialog specifically for E57 import with full pipeline."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import E57 File", "", "E57 Files (*.e57);;All Files (*)"
        )
        if path:
            self._import_e57_file(path)

    def _import_e57_file(self, path: str):
        """Import E57 file through the full processing pipeline.

        Correction is handled via the unified project YAML (loaded by
        _try_load_sidecar after import).
        """
        try:
            from ..plugins.importers.e57 import (
                E57ImportWorker,
                E57ProgressDialog,
                E57Importer,
            )
        except ImportError as e:
            QMessageBox.warning(
                self, "E57 Import", f"E57 import dependencies not available:\n{e}"
            )
            return

        importer = E57Importer()
        if not importer.is_available():
            QMessageBox.warning(
                self,
                "E57 Import",
                "E57 import requires pye57 and open3d.\n"
                "Install: pip install pye57 open3d",
            )
            return

        worker = importer.create_worker(path, self)
        dialog = importer.create_dialog(path, self)
        dialog.start(worker)
        result_code = dialog.exec()

        result = dialog.get_result()
        if result and result.layers:
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
                f"Imported E57: {n_layers} layers, {total_pts:,} points"
            )
            self.setWindowTitle(f"Locul3D Viewer — {Path(path).name}")
            self._post_load()

    def _load_file(self, path: str, fit_camera: bool = True):
        """Load a single geometry file (.ply/.obj/.stl) as a new layer.

        Args:
            path:       Absolute path to the file.
            fit_camera: If True, auto-fit camera after loading (disabled
                        during batch loads to avoid repeated re-fitting).

        After load, triggers _post_load() and auto-detects correction sidecar.
        """
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
        self._post_load()
        # Auto-detect correction sidecar
        self._try_load_sidecar(path)

    def _load_folder(self, folder: str):
        """Load all geometry files from a folder.

        Fully resets internal state before loading so the new folder opens
        from scratch (not appending to current layers).

        Scans for .ply, .obj, .stl files and loads them without fitting camera
        per file.  E57 files are routed through the import pipeline.
        If a layers.json manifest exists, layer colors/names/visibility are
        applied from it.  Camera is fitted once after all files load.
        """
        # --- Full reset ---
        self.viewport.delete_all_vbos()
        self.layer_manager.layers.clear()
        self.layer_manager.invalidate_scene_aabb()
        self.viewport.scene_correction = SceneCorrection()
        self.viewport.scene_clip = None
        self._selected_layer = None
        self.info_panel.clear()

        # Exit panorama mode if active
        if (hasattr(self.viewport, '_panorama')
                and self.viewport._panorama
                and self.viewport._panorama.is_active):
            self.viewport.exit_panorama()
            self.layer_panel.highlight_active_pano(None)

        # --- Load new folder ---
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
        self._post_load()
        self.status_label.setText(f"Loaded folder: {folder_path.name}")
        self.setWindowTitle(f"Locul3D Viewer — {folder_path.name}")

    def _on_clear_scene(self):
        """Remove all layers from the scene."""
        self.viewport.delete_all_vbos()
        for layer in self.layer_manager.layers:
            layer.release_source_data()
        self.layer_manager.layers.clear()
        self.layer_manager.invalidate_scene_aabb()
        self.viewport.scene_correction = SceneCorrection()
        self.viewport.scene_clip = None
        self._selected_layer = None
        self.info_panel.clear()

        if (hasattr(self.viewport, '_panorama')
                and self.viewport._panorama
                and self.viewport._panorama.is_active):
            self.viewport.exit_panorama()
            self.layer_panel.highlight_active_pano(None)

        self.layer_panel.rebuild()
        self.viewport.update()
        self.status_label.setText("Scene cleared")

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------

    def _on_toggle_layer_colors(self, checked):
        """Toggle per-layer color tinting in the viewport.

        When enabled, each layer is rendered with its assigned color.
        Evicts all byte caches and VBOs to force re-upload with new colors.
        """
        self.viewport.use_layer_colors = checked
        for l in self.layer_manager.layers:
            l.evict_byte_caches()
            self.viewport.delete_vbos_for_layer(l.id)
        self.viewport.update()

    def _on_fps_camera_toggled(self, checked):
        self.viewport.set_fps_camera(checked)
        self.act_fps_movement.setChecked(self.viewport.fps_movement)

    def _toggle_view(self, attr, checked):
        """Generic toggle for viewport boolean attributes (show_axes, show_grid, etc.)."""
        setattr(self.viewport, attr, checked)
        self.viewport.update()

    def _on_point_size(self, val):
        """Update GL point size from the toolbar slider (range 1-20)."""
        self.viewport.point_size = val
        self.viewport.update()

    def _on_camera_preset(self, name):
        """Set camera to a named preset (Top/Front/Right/Isometric).

        Does not change distance — only azimuth and elevation.
        "Perspective" entry in the combo is a no-op (default orbital view).
        """
        vp = self.viewport
        presets = {
            "Top": (0, 89),
            "Front": (0, 0),
            "Right": (90, 0),
            "Isometric": (45, 30),
        }
        if name in presets:
            vp.cam_azimuth, vp.cam_elevation = presets[name]
            vp.update()

    def _on_screenshot(self):
        """Save the current viewport contents to an image file.

        Uses QWidget.grab() to capture the GL widget at screen resolution.
        Supported formats depend on Qt image plugins (PNG, JPG, BMP).
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "Images (*.png *.jpg *.bmp);;All Files (*)",
        )
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
            f"Scene clip: X=[{x0:.1f},{x1:.1f}] Y=[{y0:.1f},{y1:.1f}] Z=[{z0:.1f},{z1:.1f}]"
        )

    def _on_scene_correction(self):
        """Open or raise the non-modal Scene Correction dialog."""
        if hasattr(self, "_correction_dlg") and self._correction_dlg is not None:
            self._correction_dlg.raise_()
            self._correction_dlg.activateWindow()
            return

        scene_dir = self.layer_manager.base_dir or ""
        corr = self.viewport.scene_correction or SceneCorrection()

        dlg = CorrectionDialog(
            corr,
            scene_dir,
            parent=self,
            point_source=self._collect_all_points,
        )
        dlg.correction_changed.connect(self._apply_correction)
        dlg.destroyed.connect(lambda: setattr(self, "_correction_dlg", None))
        self._correction_dlg = dlg
        dlg.show()

    def _apply_correction(self, c: SceneCorrection):
        """Apply correction values from dialog to viewport (live preview)."""
        self.viewport.scene_correction = c
        self.viewport.update()
        if not c.is_identity:
            self.status_label.setText(
                f"Correction: rot=({c.rotate_x:.1f}, {c.rotate_y:.1f}, {c.rotate_z:.1f})°  "
                f"shift=({c.shift_x:.2f}, {c.shift_y:.2f}, {c.shift_z:.2f})"
            )
        else:
            self.status_label.setText("Correction reset")

    def _collect_all_points(self):
        """Collect all visible point cloud data for auto-detection."""
        import numpy as np

        all_pts = []
        for layer in self.layer_manager.layers:
            if not layer.visible:
                continue
            if hasattr(layer, "points") and layer.points is not None:
                all_pts.append(np.asarray(layer.points, dtype=np.float64))
        if not all_pts:
            return None
        return np.vstack(all_pts)

    def _try_load_sidecar(self, scene_path: str):
        """Auto-detect and load a unified project file or correction sidecar.

        Search order:
          1. ``<stem>.yaml`` / ``<stem>.yml``   — unified project file
          2. ``<stem>.correction.yaml``         — correction-only sidecar
          3. ``correction.yaml``                — directory-level correction
        """
        p = Path(scene_path).resolve()
        parent = p.parent
        stem = p.name  # e.g. "google_test.e57"

        # --- Unified project file ---
        for ext in (".yaml", ".yml"):
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        data = yaml.safe_load(f) if HAS_YAML else {}
                    if "correction" in data:
                        c_data = data["correction"]
                        c = SceneCorrection(
                            rotate_x=float(c_data.get("rotate_x", 0)),
                            rotate_y=float(c_data.get("rotate_y", 0)),
                            rotate_z=float(c_data.get("rotate_z", 0)),
                            shift_x=float(c_data.get("shift_x", 0)),
                            shift_y=float(c_data.get("shift_y", 0)),
                            shift_z=float(c_data.get("shift_z", 0)),
                        )
                        cli = self._cli_correction
                        if cli.get("rotate_x", 0):
                            c.rotate_x = cli["rotate_x"]
                        if cli.get("rotate_y", 0):
                            c.rotate_y = cli["rotate_y"]
                        if cli.get("rotate_z", 0):
                            c.rotate_z = cli["rotate_z"]
                        if cli.get("shift_x", 0):
                            c.shift_x = cli["shift_x"]
                        if cli.get("shift_y", 0):
                            c.shift_y = cli["shift_y"]
                        if cli.get("shift_z", 0):
                            c.shift_z = cli["shift_z"]
                        self.viewport.scene_correction = c
                        self.viewport.update()
                        print(f"Correction loaded from project: {candidate.name}")
                        self.status_label.setText(f"Project loaded: {candidate.name}")
                except Exception as e:
                    print(
                        f"Warning: failed to parse project file {candidate.name}: {e}"
                    )
                return
        print(f"  No project file found for: {p.name}")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_pano_requested(self, layer):
        """Enter 360° panorama mode and highlight the active row."""
        self.viewport.enter_panorama(layer)
        self.layer_panel.highlight_active_pano(layer)

    def _on_layer_changed(self):
        """Called when any layer's visibility or properties change — trigger redraw."""
        self.viewport.update()

    def _on_layer_selected(self, layer_data):
        """Handle layer selection in the sidebar — update info panel and raise its dock."""
        self._selected_layer = layer_data
        self.info_panel.show_layer_info(layer_data)
        self._info_dock.raise_()

    def _on_fps_updated(self, fps):
        """Update status bar with camera azimuth/elevation/distance and FPS.

        Called by the viewport's fps_updated signal after each render frame.
        """
        vp = self.viewport
        self.cam_label.setText(
            f"Az:{vp.cam_azimuth % 360:.0f} El:{vp.cam_elevation:.0f} D:{vp.cam_distance:.1f}"
        )
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def _check_file_changes(self):
        """Poll loaded files for on-disk changes (hot-reload).

        Runs every 2 seconds via _file_watch_timer.  If any layer's source
        file has been modified, reloads the geometry and rebuilds VBOs.
        """
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
        """Handle global keyboard shortcuts.

        Escape: exit panorama mode if active.
        All other keys: forwarded to the default QMainWindow handler.
        """
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if (
                hasattr(self.viewport, "_panorama")
                and self.viewport._panorama
                and self.viewport._panorama.is_active
            ):
                self.viewport.exit_panorama()
                self.layer_panel.highlight_active_pano(None)
                return
        super().keyPressEvent(event)
