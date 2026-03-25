"""Locul3D Editor — main window with full annotation workflow."""

import sys
import copy
import json
from pathlib import Path
from typing import Optional, List

import numpy as np

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
    QToolButton,
    QMenu,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QKeyEvent

from ..core.layer import LayerManager, LayerData
from ..core.geometry import BBoxItem, GapItem, PlaneItem
from ..core.constants import (
    COLORS,
    BBOX_COLORS,
    PLANE_COLORS,
    DEFAULT_SIZES,
    TOOL_SELECT,
    TOOL_MOVE,
    TOOL_ROTATE,
    TOOL_SCALE,
    AXIS_NAMES,
)
from ..core.correction import SceneCorrection
from ..ui.themes import ThemeManager
from ..ui.widgets.layers import LayerPanel
from ..ui.widgets.info import InfoPanel
from ..ui.panels.bbox import BBoxPanel
from ..ui.panels.plane import PlanePanel
from ..ui.panels.reference import ReferencePanel
from ..ui.dialogs.correction_dialog import CorrectionDialog
from ..ui.dialogs.scene_dialog import SceneDialog
from ..rendering.gl.viewport import BaseGLViewport
from .viewport import EditorViewport

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _make_project_dumper():
    """YAML dumper: dicts block-style, scalar lists inline, dict lists block."""

    class ProjectDumper(yaml.SafeDumper):
        pass

    def _represent_list(dumper, data):
        # Lists of simple scalars (numbers, strings) → inline [0.8, 0.6, 2.5]
        # Lists containing dicts or nested structures → block style
        has_complex = any(isinstance(item, (dict, list)) for item in data)
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=not has_complex
        )

    ProjectDumper.add_representer(list, _represent_list)

    return ProjectDumper


class EditorWindow(QMainWindow):
    """Locul3D Editor — annotation editor for 3D point cloud scenes.

    Extends the viewer with BBox annotation, surface plane, reference point,
    and scene correction tools.  Supports undo, YAML/JSON save/load, and
    Blender-like keyboard shortcuts.

    Architecture:
        - gl_viewport:    EditorViewport — OpenGL widget with picking/gizmo
        - layer_manager:  LayerManager   — owns all loaded LayerData
        - layer_panel:    LayerPanel     — sidebar for visibility/opacity
        - bbox_panel:     BBoxPanel      — annotation list and property editing
        - plane_panel:    PlanePanel     — surface plane list and editing
        - ref_panel:      ReferencePanel — reference point and coord mode
        - info_panel:     InfoPanel      — metadata display for selected layer
        - theme:          ThemeManager   — dark/light stylesheet generation

    Lifecycle:
        1. __init__ creates UI, wires signals, optionally defers file loading
        2. _deferred_load runs after event loop starts (100ms delay)
        3. _post_load triggers background AABB/ceiling caching after any load
        4. File-watch timer polls every 2s for on-disk changes (hot reload)
    """

    def __init__(
        self, files=None, annotations_path=None, correction_angles=None, parent=None
    ):
        """Initialise the Editor window.

        Args:
            files:             List of file/folder paths to load on startup.
                               Loading is deferred 100ms to let the event loop start.
            annotations_path:  Optional path to a YAML/JSON annotation file to load.
            correction_angles: Dict with rotate_x/y/z and shift_x/y/z overrides
                               (applied to gl_viewport.scene_correction on startup).
            parent:            Optional parent QWidget.
        """
        super().__init__(parent)
        self.setWindowTitle("Locul3D — Editor")
        self.resize(1500, 900)

        self._yaml_path: Optional[str] = annotations_path
        self.annotations: List[BBoxItem] = []
        self.planes: List[PlaneItem] = []
        self.gap_items: List[GapItem] = []
        self._color_idx = 0
        self._plane_color_idx = 0
        self._undo_stack = []

        # Coordinate system
        self._coord_mode = "scene"  # "scene" = absolute, "relative" = from ref point
        self._ref_point: Optional[np.ndarray] = None

        # Theme
        self.theme = ThemeManager()
        self.setStyleSheet(self.theme.get_stylesheet())

        # Core data
        self.layer_manager = LayerManager()
        self.gl_viewport = EditorViewport(self.layer_manager)
        self.gl_viewport.annotations = self.annotations
        self.gl_viewport.planes = self.planes

        # Apply CLI correction angles if provided
        if correction_angles:
            sc = self.gl_viewport.scene_correction
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
        self.gl_viewport.fps_updated.connect(self._on_fps_updated)
        self.layer_panel.layer_changed.connect(self._on_layer_changed)
        self.layer_panel.layer_selected.connect(self._on_layer_selected)
        self.layer_panel.opacity_adjusting.connect(self._on_opacity_adjusting)
        self.layer_panel.pano_requested.connect(self._on_pano_requested)
        self.layer_panel.annotation_changed.connect(self.gl_viewport.update)

        self.gl_viewport.point_picked.connect(self._on_point_picked)
        self.gl_viewport.bbox_selected.connect(self._on_bbox_selected)
        self.gl_viewport.bbox_moved.connect(self._on_bbox_moved)
        self.gl_viewport.transform_committed.connect(self._on_transform_committed)

        # Marker click in viewport → select in layer panel (no info panel)
        self.gl_viewport.marker_selected.connect(self.layer_panel.select_layer_by_data)
        # Marker double-click → select + open info panel
        self.gl_viewport.marker_activated.connect(
            lambda layer: self.layer_panel.select_layer_by_data(layer, notify=True)
        )

        self.bbox_panel.bbox_changed.connect(self._on_bbox_panel_changed)
        self.bbox_panel.selection_changed.connect(self._on_bbox_panel_selection)
        self.bbox_panel.delete_requested.connect(self._delete_bbox)
        self.bbox_panel.create_requested.connect(self._create_bbox_at_target)
        self.bbox_panel.tool_changed.connect(self._set_tool)
        self.bbox_panel.axis_changed.connect(self._set_axis)
        self.bbox_panel.pos_mode_changed.connect(self._on_pos_mode_changed)

        self.plane_panel.plane_changed.connect(self._on_plane_changed)
        self.plane_panel.delete_requested.connect(self._delete_plane)
        self.plane_panel.create_requested.connect(self._create_plane)

        self.ref_panel.set_ref_requested.connect(self._on_set_ref_point)
        self.ref_panel.clear_ref_requested.connect(self._on_clear_ref_point)
        self.ref_panel.coord_mode_changed.connect(self._on_coord_mode_changed)

        # Wire coordinate transforms to bbox panel
        self.bbox_panel._world_to_display = self._world_to_display
        self.bbox_panel._display_to_world = self._display_to_world

        self._selected_layer = None

        # Pre-warm file dialog platform plugin (avoids delay on first open)
        QTimer.singleShot(200, self._prewarm_file_dialog)

        # Deferred file loading
        if files:
            self._deferred_files = files
            self._deferred_yaml = annotations_path
            QTimer.singleShot(100, self._deferred_load)

    def _prewarm_file_dialog(self):
        """Touch QFileDialog to initialize the platform dialog plugin early."""
        d = QFileDialog(self)
        d.deleteLater()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Create the central widget layout with the GL viewport filling it."""
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_viewport)
        self.setCentralWidget(central)

    def _setup_toolbar(self):
        """Build the main toolbar with file, view, annotation, and tool actions.

        Toolbar items (left to right):
          File:   Open File, Open Folder
          YAML:   Save, Save As, Load
          View:   Layer Colors (checkable), Axes, Grid
          Size:   Point size slider (1–20)
          Camera: Preset dropdown, Fit All, Reset View
          Tools:  Ref/Coords, Ground Z=0, Scene, Scene Correction
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
        act_clear.setToolTip("Remove all layers and annotations from the scene")
        act_clear.triggered.connect(self._on_clear_scene)
        toolbar.addAction(act_clear)

        toolbar.addSeparator()

        # YAML save/load
        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._on_save_yaml)
        toolbar.addAction(act_save)

        act_save_as = QAction("Save As...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._on_save_yaml_as)
        toolbar.addAction(act_save_as)

        act_load = QAction("Load...", self)
        act_load.setShortcut("Ctrl+L")
        act_load.triggered.connect(self._on_load_yaml)
        toolbar.addAction(act_load)

        toolbar.addSeparator()

        # View toggles
        self.act_layer_colors = QAction(
            "Layer Colors", self, checkable=True, checked=False
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
        act_fit.triggered.connect(self.gl_viewport.fit_to_scene)
        toolbar.addAction(act_fit)

        act_reset = QAction("Reset View", self)
        act_reset.setShortcut("Home")
        act_reset.setToolTip(
            "Fit view to selected bbox (or scene) with current projection"
        )
        act_reset.triggered.connect(self._on_reset_view)
        toolbar.addAction(act_reset)

        toolbar.addSeparator()

        self.act_ref_panel = QAction("Ref/Coords", self, checkable=True)
        self.act_ref_panel.setToolTip("Toggle Reference & Coordinates panel")
        self.act_ref_panel.triggered.connect(lambda c: self._ref_dock.setVisible(c))
        toolbar.addAction(self.act_ref_panel)

        act_ground = QAction("Ground Z=0", self, checkable=True)
        act_ground.setToolTip("Show global reference plane at Z=0 (cyan wireframe)")
        act_ground.triggered.connect(
            lambda c: self._toggle_view("show_ground_plane", c)
        )
        toolbar.addAction(act_ground)

        act_scene = QAction("Scene", self)
        act_scene.setToolTip("Scene bounds, ceiling clipping, dimensions")
        act_scene.triggered.connect(self._on_scene)
        toolbar.addAction(act_scene)

        act_correction = QAction("Scene Correction", self)
        act_correction.setToolTip("Adjust scene rotation and shift for axis alignment")
        act_correction.triggered.connect(self._on_scene_correction)
        toolbar.addAction(act_correction)

        act_pipeline = QAction("Load Pipeline Context", self)
        act_pipeline.setToolTip("Load pipeline_context.yaml to display gap annotations between racks")
        act_pipeline.triggered.connect(self._on_load_pipeline_context)
        toolbar.addAction(act_pipeline)

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
        """Create dock widgets: Layers, BBox, Planes, Info, Reference.

        Right-side docks are tabified (Layers → BBox → Planes → Info).
        Reference dock is floating and hidden by default.
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

        # BBox dock
        self._bbox_dock = QDockWidget("BBox Annotations", self)
        self._bbox_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._bbox_dock.setMinimumWidth(320)
        self.bbox_panel = BBoxPanel(self.annotations)
        self._bbox_dock.setWidget(self.bbox_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._bbox_dock)

        # Planes dock
        self._planes_dock = QDockWidget("Surface Planes", self)
        self._planes_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._planes_dock.setMinimumWidth(300)
        self.plane_panel = PlanePanel(self.planes)
        self._planes_dock.setWidget(self.plane_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._planes_dock)

        # Info dock
        self._info_dock = QDockWidget("Info", self)
        self._info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._info_dock.setMinimumWidth(280)
        self.info_panel = InfoPanel()
        self._info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

        # Reference & Coordinates dock (floating by default)
        self._ref_dock = QDockWidget("Reference & Coordinates", self)
        self._ref_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._ref_dock.setMinimumWidth(240)
        self.ref_panel = ReferencePanel()
        self._ref_dock.setWidget(self.ref_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._ref_dock)
        self._ref_dock.setFloating(True)
        self._ref_dock.resize(260, 300)
        self._ref_dock.hide()  # hidden by default
        self._ref_dock.visibilityChanged.connect(
            lambda vis: self.act_ref_panel.setChecked(vis)
        )

        # Tabify docks on the right
        self.tabifyDockWidget(self._layers_dock, self._bbox_dock)
        self.tabifyDockWidget(self._bbox_dock, self._planes_dock)
        self.tabifyDockWidget(self._planes_dock, self._info_dock)
        self._layers_dock.raise_()

    def _setup_statusbar(self):
        """Create status bar with tool/axis info, camera readout, and FPS."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready — Ctrl+Click to place bbox")
        self.cam_label = QLabel("Cam: --")
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.cam_label)
        self.status_bar.addPermanentWidget(self.fps_label)

    # ------------------------------------------------------------------
    # Tool mode
    # ------------------------------------------------------------------

    def _on_pos_mode_changed(self, mode: str):
        """Sync viewport scale behaviour with BBox panel position mode.

        In 'corners' mode, scale handles anchor the opposite face.
        In 'center' mode, scaling is symmetric around center.
        """
        self.gl_viewport.scale_from_corner = mode == "corners"

    def _set_tool(self, tool):
        """Switch the active gizmo tool (select/move/rotate/scale).

        Updates the viewport, the BBox panel button states, and the status bar.
        """
        self.gl_viewport.tool = tool
        self.bbox_panel.set_tool(tool)
        self._update_status()
        self.gl_viewport.update()

    def _set_axis(self, axis):
        """Set or clear the axis constraint for move/rotate/scale gizmos.

        Args:
            axis: 0=X, 1=Y, 2=Z, or None for unconstrained.
        """
        self.gl_viewport.axis_constraint = axis
        self.bbox_panel.set_axis(axis)
        self._update_status()
        self.gl_viewport.update()

    def _update_status(self):
        """Refresh the status bar text with current tool and axis info."""
        tool = self.gl_viewport.tool
        axis = self.gl_viewport.axis_constraint
        parts = [f"Tool: {tool.capitalize()}"]
        if axis is not None:
            parts.append(f"Axis: {AXIS_NAMES[axis]}")
        parts.append("Ctrl+Click to place")
        self.status_label.setText(" | ".join(parts))

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _post_load(self):
        """Called after any geometry load — refresh caches in background."""
        self.layer_manager.invalidate_scene_aabb()
        QTimer.singleShot(0, self.layer_manager.compute_ceiling_background)

    def _deferred_load(self):
        """Load files passed via constructor after the event loop starts."""
        files = getattr(self, "_deferred_files", [])
        yaml_path = getattr(self, "_deferred_yaml", None)
        for arg in files:
            p = Path(arg)
            if p.is_dir():
                self._load_folder(str(p))
            elif p.is_file():
                if p.suffix.lower() == ".e57":
                    self._import_e57_file(str(p))
                else:
                    self._load_file(str(p), fit_camera=False)
        self.gl_viewport.fit_to_scene()
        self._post_load()
        if yaml_path and Path(yaml_path).exists():
            self._load_yaml(yaml_path)
        self._update_status()

    def _on_open_file(self):
        """Show file dialog to open one or more 3D files.

        E57 files are routed through the full import pipeline with progress.
        Sidecar detection is handled by _import_e57_file / _load_file.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Files", "", "3D Files (*.ply *.obj *.stl *.e57);;All Files (*)"
        )
        if not paths:
            return

        # Separate E57 files from geometry files
        e57_paths = [p for p in paths if Path(p).suffix.lower() == ".e57"]
        geo_paths = [p for p in paths if Path(p).suffix.lower() != ".e57"]

        # Load geometry files with progress dialog
        if geo_paths:
            from ..ui.dialogs.folder_loading import (
                FolderLoadWorker, FolderProgressDialog,
            )
            file_names = [Path(p).name for p in geo_paths]
            folder = str(Path(geo_paths[0]).parent)
            worker = FolderLoadWorker(geo_paths, self)
            dialog = FolderProgressDialog(folder, file_names, self)
            dialog.start(worker)
            dialog.exec()

            layers = dialog.get_result()
            if layers:
                self.layer_manager.base_dir = str(Path(geo_paths[0]).parent)
                for layer in layers:
                    self.layer_manager.layers.append(layer)
                self.layer_manager.invalidate_scene_aabb()
                self.layer_panel.rebuild()
                self.gl_viewport.fit_to_scene()
                self._post_load()

        # E57 files through their own pipeline
        for p in e57_paths:
            self._import_e57_file(p)

    def _on_open_folder(self):
        """Show folder dialog and load all supported files from selected directory."""
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self._load_folder(folder)

    def _import_e57_file(self, path: str):
        """Import E57 file through the full processing pipeline.

        Correction and annotations are handled separately via the unified
        project YAML file (loaded by _try_load_sidecar after import).
        """
        # --- E57 import ---
        try:
            from ..plugins.importers.e57 import (
                E57ImportWorker,
                E57ProgressDialog,
                E57Importer,
            )
        except ImportError as e:
            self.status_label.setText(f"E57 import not available: {e}")
            return

        importer = E57Importer()
        if not importer.is_available():
            msg = importer.missing_deps_message()
            self.status_label.setText(msg)
            self._log(msg)
            return

        worker = importer.create_worker(path, self)
        dialog = importer.create_dialog(path, self)
        dialog.start(worker)
        dialog.exec()

        result = dialog.get_result()
        if result and result.layers:
            for layer in result.layers:
                self.layer_manager.layers.append(layer)
            self.layer_panel.rebuild()
            self.gl_viewport.fit_to_scene()
            if result.metadata or result.stats:
                self.info_panel.populate(result.metadata, result.stats)
            n_layers = len(result.layers)
            total_pts = sum(l.point_count for l in result.layers)
            self.status_label.setText(
                f"Imported E57: {n_layers} layers, {total_pts:,} points"
            )
            self._post_load()
            # Auto-detect project YAML / correction sidecar next to the E57
            self._try_load_sidecar(path)

    def _load_file(self, path: str, fit_camera: bool = True):
        """Load a single geometry file as a new layer.

        Args:
            path:       Absolute path to the file.
            fit_camera: If True, auto-fit camera after loading.
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
            self.gl_viewport.fit_to_scene()
        last = self.layer_manager.layers[-1] if self.layer_manager.layers else None
        if last:
            self.info_panel.show_layer_info(last)
        self.status_label.setText(f"Loaded {Path(path).name}")
        self._post_load()
        # Auto-detect correction sidecar
        self._try_load_sidecar(path)

    def _load_folder(self, folder: str):
        """Load all .ply files from a folder, applying layers.json manifest if present.

        Fully resets internal state before loading so the new folder opens
        from scratch (not appending to current layers).
        Camera is fitted once after all files load.
        """
        # --- Full reset ---
        self.gl_viewport.delete_all_vbos()
        self.layer_manager.layers.clear()
        self.layer_manager.invalidate_scene_aabb()
        self.annotations.clear()
        self.planes.clear()
        self._undo_stack.clear()
        self._yaml_path = None
        self._color_idx = 0
        self._plane_color_idx = 0
        self.gl_viewport.selected_idx = -1
        self.gl_viewport.scene_correction = SceneCorrection()
        self.gl_viewport.scene_clip = None
        self.gl_viewport.set_correction_diagnostics(None)
        self._ref_point = None
        self.gl_viewport.ref_point = None
        self.bbox_panel.rebuild_list()
        self.plane_panel.rebuild_list()
        self.info_panel.clear()

        # --- Load new folder ---
        folder_path = Path(folder)
        ply_files = sorted(folder_path.glob("*.ply"))

        if ply_files:
            from ..ui.dialogs.folder_loading import (
                FolderLoadWorker, FolderProgressDialog,
            )
            file_paths = [str(p) for p in ply_files]
            file_names = [p.name for p in ply_files]
            worker = FolderLoadWorker(file_paths, self)
            dialog = FolderProgressDialog(folder, file_names, self)
            dialog.start(worker)
            dialog.exec()

            layers = dialog.get_result()
            if layers:
                self.layer_manager.base_dir = str(folder_path)
                for layer in layers:
                    self.layer_manager.layers.append(layer)
                self.layer_manager.invalidate_scene_aabb()
                self.layer_panel.rebuild()

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
                self.gl_viewport.use_layer_colors = True
                self.act_layer_colors.setChecked(True)
                self.layer_panel.rebuild()
            except Exception:
                pass
        self.gl_viewport.fit_to_scene()
        self._post_load()
        self.status_label.setText(f"Loaded folder: {folder_path.name}")
        self.setWindowTitle(f"Locul3D Editor — {folder_path.name}")

    # ------------------------------------------------------------------
    # Pipeline context (gap annotations)
    # ------------------------------------------------------------------

    def _on_load_pipeline_context(self):
        """Open file dialog to load pipeline_context.yaml and display gap annotations."""
        if not HAS_YAML:
            self.status_label.setText("PyYAML not installed — cannot load pipeline context")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline Context", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            bboxes, gaps = self._parse_pipeline_context(data)
            # Store pipeline bboxes in scene-space list (post-correction rendering)
            self.gl_viewport.scene_bboxes = bboxes
            # Also add to annotations so they appear in the bbox panel list
            for bbox in bboxes:
                bbox.scene_coords = True  # flag: drawn in scene space, skip in _draw_annotations
                self.annotations.append(bbox)
            self.gap_items = gaps
            self.gl_viewport.gaps = gaps
            self.bbox_panel.rebuild_list()

            # Build annotation groups for layer panel toggles
            rack_bboxes = [b for b in bboxes if b.label == "rack"]
            es_bboxes = [b for b in bboxes if b.label == "empty_space"]
            rack_gaps = [g for g in gaps if g.color == self._RACK_GAP_ANNOT]
            es_gaps = [g for g in gaps if g.color == self._EMPTY_GAP_ANNOT]
            groups = []
            if rack_bboxes or rack_gaps:
                groups.append({"name": "Racks", "color": self._RACK_COLOR,
                               "items": rack_bboxes + rack_gaps})
            if es_bboxes or es_gaps:
                groups.append({"name": "Empty Spaces", "color": self._EMPTY_SPACE_COLOR,
                               "items": es_bboxes + es_gaps})
            self.layer_panel.annotation_groups = groups
            self.layer_panel.rebuild()

            self.gl_viewport.update()
            n_racks = len(rack_bboxes)
            n_es = len(es_bboxes)
            self.status_label.setText(
                f"Pipeline: {n_racks} racks, {n_es} empty spaces, {len(gaps)} gap annotations")
        except Exception as exc:
            self.status_label.setText(f"Failed to load pipeline context: {exc}")

    # Bbox + gap annotation colors (known, so text contrast is deterministic)
    _RACK_COLOR = [1.0, 0.5, 0.0]       # orange
    _EMPTY_SPACE_COLOR = [1.0, 0.2, 0.2] # red
    _RACK_GAP_ANNOT = (0.0, 0.85, 0.85)  # cyan — contrasts orange
    _EMPTY_GAP_ANNOT = (0.2, 0.9, 0.2)   # green — contrasts red

    def _parse_pipeline_context(self, data):
        """Parse pipeline_context.yaml → (list[BBoxItem], list[GapItem])."""
        racks = data.get("stage4", {}).get("racks", [])
        stage5 = data.get("stage5", {})
        corridor_axis = stage5.get("corridor_axis", "X")
        axis = 0 if corridor_axis == "X" else 1
        cross_axis = 1 - axis

        bboxes = []
        gaps = []

        # --- Rack bboxes (orange) ---
        for r in racks:
            try:
                bboxes.append(BBoxItem(
                    label="rack", center=r["center"], size=r["size"],
                    color=self._RACK_COLOR))
            except (KeyError, TypeError):
                continue

        # --- Empty space bboxes (red) ---
        for es in stage5.get("empty_spaces", []):
            try:
                bboxes.append(BBoxItem(
                    label="empty_space", center=es["center"], size=es["size"],
                    color=self._EMPTY_SPACE_COLOR))
            except (KeyError, TypeError):
                continue

        # --- Rack gaps (bracket above racks, ticks down to rack tops) ---
        for gap_info in stage5.get("rack_gaps", []):
            try:
                a_idx = gap_info["rack_a_index"]
                b_idx = gap_info["rack_b_index"]
            except (KeyError, TypeError):
                continue
            if not (0 <= a_idx < len(racks) and 0 <= b_idx < len(racks)):
                continue

            try:
                a = racks[a_idx]
                b = racks[b_idx]
                ac, asz = a["center"], a["size"]
                bc, bsz = b["center"], b["size"]

                a_right = ac[axis] + asz[axis] / 2
                b_left = bc[axis] - bsz[axis] / 2
                cross = (ac[cross_axis] + bc[cross_axis]) / 2
                a_top_z = ac[2] + asz[2] / 2
                b_top_z = bc[2] + bsz[2] / 2
                arrow_z = max(a_top_z, b_top_z) + 0.05

                if axis == 1:
                    edge_a = [cross, a_right, arrow_z]
                    edge_b = [cross, b_left, arrow_z]
                    anchor_a = [cross, a_right, a_top_z]
                    anchor_b = [cross, b_left, b_top_z]
                else:
                    edge_a = [a_right, cross, arrow_z]
                    edge_b = [b_left, cross, arrow_z]
                    anchor_a = [a_right, cross, a_top_z]
                    anchor_b = [b_left, cross, b_top_z]

                gaps.append(GapItem(edge_a, edge_b, gap_info["gap_mm"], axis, True,
                                    anchor_a=anchor_a, anchor_b=anchor_b,
                                    tick_dir=[0, 0, 0.03],
                                    color=self._RACK_GAP_ANNOT))
            except (KeyError, IndexError, TypeError):
                continue

        # --- Empty space gaps (bracket at front face) ---
        all_rack_rows = []
        for rr in data.get("stage3_rack", {}).get("rack_regions", []):
            try:
                bbox = rr["bbox"]
                rr_min = bbox["min"]
                rr_max = bbox["max"]
                side = rr.get("side", "right")
                cross_center = (rr_min[cross_axis] + rr_max[cross_axis]) / 2
                depth = rr_max[cross_axis] - rr_min[cross_axis]
                if side == "right":
                    front_x = cross_center - depth / 2
                    sign = -1
                else:
                    front_x = cross_center + depth / 2
                    sign = 1
                all_rack_rows.append((cross_center, front_x, sign))
            except (KeyError, TypeError):
                continue

        for es in stage5.get("empty_spaces", []):
            try:
                center = es["center"]
                along_min = es["along_min"]
                along_max = es["along_max"]
                length_mm = es["length_mm"]

                if not all_rack_rows:
                    continue
                es_cross = center[cross_axis]
                _, front_x, sign = min(all_rack_rows, key=lambda r: abs(r[0] - es_cross))
                offset = 0.08
                mid_z = center[2]

                if axis == 1:
                    bracket_x = front_x + sign * offset
                    edge_a = [bracket_x, along_min, mid_z]
                    edge_b = [bracket_x, along_max, mid_z]
                    anchor_a = [front_x, along_min, mid_z]
                    anchor_b = [front_x, along_max, mid_z]
                    tick_dir = [sign * 0.03, 0, 0]
                else:
                    bracket_y = front_x + sign * offset
                    edge_a = [along_min, bracket_y, mid_z]
                    edge_b = [along_max, bracket_y, mid_z]
                    anchor_a = [along_min, front_x, mid_z]
                    anchor_b = [along_max, front_x, mid_z]
                    tick_dir = [0, sign * 0.03, 0]

                gaps.append(GapItem(edge_a, edge_b, length_mm, axis, True,
                                    anchor_a=anchor_a, anchor_b=anchor_b,
                                    tick_dir=tick_dir,
                                    color=self._EMPTY_GAP_ANNOT))
            except (KeyError, IndexError, TypeError):
                continue

        return bboxes, gaps

    def _on_clear_scene(self):
        """Remove all layers and annotations from the scene."""
        self.gl_viewport.delete_all_vbos()
        for layer in self.layer_manager.layers:
            layer.release_source_data()
        self.layer_manager.layers.clear()
        self.layer_manager.invalidate_scene_aabb()
        self.annotations.clear()
        self.planes.clear()
        self.gap_items.clear()
        self.gl_viewport.gaps = self.gap_items
        self.layer_panel.annotation_groups = []
        self._undo_stack.clear()
        self._yaml_path = None
        self._color_idx = 0
        self._plane_color_idx = 0
        self.gl_viewport.selected_idx = -1
        self.gl_viewport.scene_correction = SceneCorrection()
        self.gl_viewport.scene_clip = None
        self.gl_viewport.set_correction_diagnostics(None)
        self._ref_point = None
        self.gl_viewport.ref_point = None
        self.bbox_panel.rebuild_list()
        self.plane_panel.rebuild_list()
        self.info_panel.clear()
        self.layer_panel.rebuild()
        self.gl_viewport.update()
        self.status_label.setText("Scene cleared")

    # ------------------------------------------------------------------
    # BBox operations
    # ------------------------------------------------------------------

    def _create_bbox_at_position(self, x, y, z):
        """Create a new BBox annotation at the given world coordinates.

        Uses the current label from bbox_panel's combo box and default size.
        Bottom-center is placed at the picked point. Auto-selects the new bbox.
        """
        label = self.bbox_panel.label_combo.currentText() or "mts_column"
        size = DEFAULT_SIZES.get(label, DEFAULT_SIZES["custom"]).copy()
        # Place bottom-center at picked point
        center = np.array([x, y, z + size[2] / 2.0], dtype=np.float64)
        color = list(BBOX_COLORS[self._color_idx % len(BBOX_COLORS)])
        self._color_idx += 1
        bbox = BBoxItem(label=label, center=center, size=size, color=color)
        self.annotations.append(bbox)
        self._push_undo("create", {"idx": len(self.annotations) - 1})
        self.bbox_panel.rebuild_list()
        idx = len(self.annotations) - 1
        self.bbox_panel.select_bbox(idx)
        self.gl_viewport.selected_idx = idx
        self.gl_viewport.update()
        self.status_label.setText(
            f"Created [{idx}] {label} at ({x:.2f}, {y:.2f}, {z:.2f})"
        )

    def _create_bbox_at_target(self):
        """Create a new BBox at the camera target point (keyboard shortcut N)."""
        t = self.gl_viewport.cam_target
        self._create_bbox_at_position(float(t[0]), float(t[1]), float(t[2]))

    def _delete_bbox(self, idx):
        """Delete a BBox annotation by index, pushing the action to the undo stack."""
        if idx < 0 or idx >= len(self.annotations):
            return
        self._push_undo(
            "delete", {"idx": idx, "bbox": copy.deepcopy(self.annotations[idx])}
        )
        self.annotations.pop(idx)
        self.gl_viewport.selected_idx = -1
        self.bbox_panel.rebuild_list()
        self.bbox_panel.select_bbox(-1)
        self.gl_viewport.update()
        self.status_label.setText(f"Deleted bbox [{idx}]")

    def _undo(self):
        """Pop and reverse the last action from the undo stack.

        Supports undo for create, delete, and transform operations.
        """
        action = self._pop_undo()
        if action is None:
            return
        act_type, data = action
        if act_type == "create":
            idx = data["idx"]
            if idx < len(self.annotations):
                self.annotations.pop(idx)
        elif act_type == "delete":
            self.annotations.insert(data["idx"], data["bbox"])
        elif act_type == "transform":
            idx = data["idx"]
            if idx < len(self.annotations):
                b = self.annotations[idx]
                b.center_pos = data["center"]
                b.size = data["size"]
                b.rotation_z = data["rotation_z"]
        self.gl_viewport.selected_idx = -1
        self.bbox_panel.rebuild_list()
        self.bbox_panel.select_bbox(-1)
        self.gl_viewport.update()
        self.status_label.setText("Undo")

    def _push_undo(self, action_type, data):
        """Push an action onto the undo stack for later reversal."""
        self._undo_stack.append((action_type, data))

    def _pop_undo(self):
        """Pop and return the most recent undo action, or None if stack is empty."""
        if not self._undo_stack:
            return None
        return self._undo_stack.pop()

    def _on_transform_committed(self, idx, snapshot):
        """Called before a gizmo drag starts — push pre-drag state for undo."""
        self._push_undo(
            "transform",
            {
                "idx": idx,
                "center": snapshot["center"],
                "size": snapshot["size"],
                "rotation_z": snapshot["rotation_z"],
            },
        )

    # ------------------------------------------------------------------
    # Plane operations
    # ------------------------------------------------------------------

    def _create_plane(self):
        """Create a new surface plane at the camera target, corrected to Z=0.

        Color is auto-cycled from PLANE_COLORS palette.
        Raises the Planes dock tab.
        """
        t = self.gl_viewport.cam_target
        color = list(PLANE_COLORS[self._plane_color_idx % len(PLANE_COLORS)])
        self._plane_color_idx += 1
        # Transform center through correction so X,Y match the visual scene
        sc = self.gl_viewport.scene_correction
        center = sc.transform_point([float(t[0]), float(t[1]), 0.0])
        center[2] = 0.0  # keep Z=0 for floor-level reference
        plane = PlaneItem(
            axis="xy",
            center=center.tolist(),
            size=[5.0, 5.0],
            color=color,
            opacity=0.25,
        )
        self.planes.append(plane)
        self.plane_panel.rebuild_list()
        idx = len(self.planes) - 1
        self.plane_panel.select_plane(idx)
        self.gl_viewport.update()
        self._planes_dock.raise_()
        self.status_label.setText(f"Created plane [{idx}] at target")

    def _delete_plane(self, idx):
        """Delete a surface plane by index."""
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes.pop(idx)
        self.plane_panel.rebuild_list()
        self.plane_panel.select_plane(-1)
        self.gl_viewport.update()
        self.status_label.setText(f"Deleted plane [{idx}]")

    def _on_plane_changed(self, idx):
        """Redraw viewport when a plane's properties are edited in the panel."""
        self.gl_viewport.update()

    # ------------------------------------------------------------------
    # Reference point & Coordinates
    # ------------------------------------------------------------------

    def _on_set_ref_point(self):
        """Enter reference point picking mode — next point-cloud click sets the origin."""
        self.gl_viewport._picking_ref_point = True
        self.status_label.setText("Click on point cloud to set reference origin...")

    def _on_clear_ref_point(self):
        """Clear the reference point and revert to scene coordinates."""
        self._ref_point = None
        self.gl_viewport.ref_point = None
        self.ref_panel.clear_ref_point()
        self.gl_viewport.update()
        self._refresh_bbox_panel_coords()
        self.status_label.setText("Reference point cleared")

    def _on_ref_point_picked(self, x, y, z):
        """Handle a ref-point pick — store the point, update panel and viewport."""
        self._ref_point = np.array([x, y, z], dtype=np.float64)
        self.gl_viewport.ref_point = self._ref_point
        self.ref_panel.set_ref_point(x, y, z)
        self.gl_viewport.update()
        self._refresh_bbox_panel_coords()
        self.status_label.setText(f"Reference point set to ({x:.2f}, {y:.2f}, {z:.2f})")

    def _on_coord_mode_changed(self, index):
        """Switch between scene (absolute) and relative coordinate display."""
        self._coord_mode = "scene" if index == 0 else "relative"
        self._refresh_bbox_panel_coords()

    def _refresh_bbox_panel_coords(self):
        """Refresh the BBox panel's coordinate display after ref-point or mode change."""
        idx = self.gl_viewport.selected_idx
        if idx >= 0 and idx < len(self.annotations):
            self.bbox_panel.update_values(idx)

    def _world_to_display(self, pos):
        """Convert world position to display coordinates."""
        if self._coord_mode == "relative" and self._ref_point is not None:
            return pos - self._ref_point
        return pos

    def _display_to_world(self, display_pos):
        """Convert display coordinates back to world position."""
        if self._coord_mode == "relative" and self._ref_point is not None:
            return display_pos + self._ref_point
        return display_pos

    # ------------------------------------------------------------------
    # YAML save/load
    # ------------------------------------------------------------------

    def _save_yaml(self, path: str):
        """Serialise the unified project file (annotations, planes, correction).

        All sections are optional — only non-empty sections are written.
        Format depends on whether PyYAML is installed (HAS_YAML).
        Updates window title and status bar on success.
        """
        data = {
            "default_column_size": DEFAULT_SIZES["mts_column"].tolist(),
            "default_box_size": DEFAULT_SIZES["mts_box"].tolist(),
        }
        # Scene correction
        corr = self.gl_viewport.scene_correction
        if corr is not None and not corr.is_identity:
            data["correction"] = {
                "rotate_x": round(corr.rotate_x, 4),
                "rotate_y": round(corr.rotate_y, 4),
                "rotate_z": round(corr.rotate_z, 4),
                "shift_x": round(corr.shift_x, 4),
                "shift_y": round(corr.shift_y, 4),
                "shift_z": round(corr.shift_z, 4),
            }
        # Annotations
        # Bboxes are stored internally in world (global) coordinates.
        # For YAML persistence, inverse-transform back to scene (scanner-local)
        # coordinates so the file format stays in the original coordinate system.
        if self.annotations:
            save_bboxes = []
            for b in self.annotations:
                if corr is not None and not corr.is_identity:
                    import copy
                    b_copy = copy.deepcopy(b)
                    b_copy.center_pos = corr.inverse_transform_point(b.center_pos)
                    save_bboxes.append(b_copy.to_dict())
                else:
                    save_bboxes.append(b.to_dict())
            data["bboxes"] = save_bboxes
        if self.planes:
            data["planes"] = [p.to_dict() for p in self.planes]
        if self._ref_point is not None:
            data["reference_point"] = [round(float(v), 4) for v in self._ref_point]
        if HAS_YAML:
            with open(path, "w") as f:
                dumper = _make_project_dumper()
                yaml.dump(data, f, Dumper=dumper, sort_keys=False, allow_unicode=True)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        self._yaml_path = path
        parts = []
        if self.annotations:
            parts.append(f"{len(self.annotations)} bboxes")
        if self.planes:
            parts.append(f"{len(self.planes)} planes")
        if corr and not corr.is_identity:
            parts.append("correction")
        summary = ", ".join(parts) if parts else "empty"
        self.status_label.setText(f"Saved {summary} to {Path(path).name}")
        self.setWindowTitle(f"Locul3D Editor — {Path(path).name}")

    def _load_yaml(self, path: str):
        """Load annotations, planes, correction, and reference point from YAML/JSON.

        This is the unified project file loader.  Supports all sections:
          - ``bboxes``          — annotation boxes (center+size or min+max)
          - ``planes``          — reference planes
          - ``reference_point`` — coordinate reference point
          - ``correction``      — scene rotation/shift correction

        Replaces all existing annotations and planes.
        Updates window title and status bar.
        """
        with open(path) as f:
            if path.endswith((".yaml", ".yml")) and HAS_YAML:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        # --- Annotations ---
        self.annotations.clear()
        for entry in data.get("bboxes", []):
            self.annotations.append(BBoxItem.from_dict(entry))
        self.planes.clear()
        for entry in data.get("planes", []):
            self.planes.append(PlaneItem.from_dict(entry))
        rp = data.get("reference_point")
        if rp is not None:
            self._ref_point = np.array(rp, dtype=np.float64)
            self.gl_viewport.ref_point = self._ref_point
            self.ref_panel.set_ref_point(rp[0], rp[1], rp[2])
        # --- Scene correction (optional) ---
        if "correction" in data:
            c = data["correction"]
            corr = SceneCorrection(
                rotate_x=float(c.get("rotate_x", 0)),
                rotate_y=float(c.get("rotate_y", 0)),
                rotate_z=float(c.get("rotate_z", 0)),
                shift_x=float(c.get("shift_x", 0)),
                shift_y=float(c.get("shift_y", 0)),
                shift_z=float(c.get("shift_z", 0)),
            )
            # CLI overrides
            cli = self._cli_correction
            if cli.get("rotate_x", 0):
                corr.rotate_x = cli["rotate_x"]
            if cli.get("rotate_y", 0):
                corr.rotate_y = cli["rotate_y"]
            if cli.get("rotate_z", 0):
                corr.rotate_z = cli["rotate_z"]
            if cli.get("shift_x", 0):
                corr.shift_x = cli["shift_x"]
            if cli.get("shift_y", 0):
                corr.shift_y = cli["shift_y"]
            if cli.get("shift_z", 0):
                corr.shift_z = cli["shift_z"]
            self.gl_viewport.scene_correction = corr
            self.gl_viewport.update()
        # --- Transform bboxes from scene → world coordinates ---
        # Project YAML stores bboxes in original (scene/scanner-local) coords.
        # Transform them to the target (world/global) coordinate system so
        # they are ALWAYS rendered and interacted with in world space.
        corr = self.gl_viewport.scene_correction
        if corr is not None and not corr.is_identity and self.annotations:
            for bbox in self.annotations:
                bbox.center_pos = corr.transform_point(bbox.center_pos)
        # --- UI update ---
        self._yaml_path = path
        self.bbox_panel.rebuild_list()
        self.plane_panel.rebuild_list()
        self.gl_viewport.selected_idx = -1
        self.gl_viewport.update()
        n_bbox = len(self.annotations)
        n_plane = len(self.planes)
        has_corr = "correction" in data
        parts = []
        if n_bbox:
            parts.append(f"{n_bbox} bboxes")
        if n_plane:
            parts.append(f"{n_plane} planes")
        if has_corr:
            parts.append("correction")
        summary = ", ".join(parts) if parts else "empty"
        self.status_label.setText(f"Loaded {summary} from {Path(path).name}")
        self.setWindowTitle(f"Locul3D Editor — {Path(path).name}")

    def _on_save_yaml(self):
        """Save to the current YAML path, or prompt Save As if no path set."""
        if self._yaml_path:
            self._save_yaml(self._yaml_path)
        else:
            self._on_save_yaml_as()

    def _on_save_yaml_as(self):
        """Prompt for a new file path and save annotations."""
        ext = "YAML (*.yaml *.yml)" if HAS_YAML else "JSON (*.json)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", "", f"{ext};;All Files (*)"
        )
        if path:
            self._save_yaml(path)

    def _on_load_yaml(self):
        """Prompt for a YAML/JSON file and load annotations from it."""
        ext = "YAML/JSON (*.yaml *.yml *.json)" if HAS_YAML else "JSON (*.json)"
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", "", f"{ext};;All Files (*)"
        )
        if path:
            self._load_yaml(path)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_point_picked(self, x, y, z):
        """Handle a point-cloud pick: route to ref-point handler or create bbox.

        If in ref-point picking mode, sets the reference origin.
        Otherwise creates a new BBox at the picked position.
        """
        # Route to ref-point handler if in picking mode
        if getattr(self.gl_viewport, "_picking_ref_point", False):
            self.gl_viewport._picking_ref_point = False
            self._on_ref_point_picked(x, y, z)
            return
        self._create_bbox_at_position(x, y, z)

    def _on_bbox_selected(self, idx):
        """Handle bbox selection from viewport click — raise BBox dock."""
        self.bbox_panel.select_bbox(idx)
        self._bbox_dock.raise_()

    def _on_bbox_moved(self, idx):
        """Refresh BBox panel values after a gizmo drag moves a bbox."""
        self.bbox_panel.update_values(idx)

    def _on_bbox_panel_changed(self, idx):
        """Redraw viewport when bbox properties are edited in the panel."""
        self.gl_viewport.update()

    def _on_bbox_panel_selection(self, idx):
        """Sync viewport selection with the BBox panel's list selection."""
        self.gl_viewport.selected_idx = idx
        self.gl_viewport.update()

    # ------------------------------------------------------------------
    # Layer handlers
    # ------------------------------------------------------------------

    def _on_layer_changed(self):
        """Called when any layer's visibility/properties change — evict VBOs and redraw."""
        for layer in self.layer_manager.layers:
            if not layer.visible:
                self.gl_viewport.delete_vbos_for_layer(layer.id)
        self.gl_viewport.update()

    def _on_pano_requested(self, layer):
        """Enter 360° panorama mode and highlight the active row."""
        self.gl_viewport.enter_panorama(layer)
        self.layer_panel.highlight_active_pano(layer)

    def _on_opacity_adjusting(self, adjusting: bool):
        """Stride-based preview while opacity slider is dragged."""
        self.gl_viewport._adjusting_opacity = adjusting
        if not adjusting:
            self.gl_viewport.update()  # full-quality redraw on release

    def _on_layer_selected(self, layer_data):
        """Handle layer selection in sidebar — update info panel and raise its dock."""
        self._selected_layer = layer_data
        self.info_panel.show_layer_info(layer_data)
        self._info_dock.raise_()

    # ------------------------------------------------------------------
    # View controls
    # ------------------------------------------------------------------

    def _on_scene(self):
        """Open the non-modal Scene dialog for bounds & ceiling clipping."""
        dlg = SceneDialog(self.layer_manager, self.gl_viewport, self)
        dlg.clip_changed.connect(self._apply_scene_clip)
        dlg.show()  # non-modal
        self._scene_dialog = dlg  # prevent GC

    def _apply_scene_clip(self, x0, x1, y0, y1, z0, z1):
        """Apply scene clip bounds from the Scene dialog."""
        self.gl_viewport.scene_clip = (x0, x1, y0, y1, z0, z1)
        self.gl_viewport.update()
        self.status_label.setText(
            f"Scene clip: X=[{x0:.1f},{x1:.1f}] Y=[{y0:.1f},{y1:.1f}] Z=[{z0:.1f},{z1:.1f}]"
        )

    def _on_scene_correction(self):
        """Open or raise the non-modal Scene Correction dialog."""
        # Reuse existing dialog if already open
        if hasattr(self, "_correction_dlg") and self._correction_dlg is not None:
            self._correction_dlg.show()
            self._correction_dlg.raise_()
            self._correction_dlg.activateWindow()
            return

        scene_dir = self.layer_manager.base_dir or ""
        corr = self.gl_viewport.scene_correction or SceneCorrection()

        dlg = CorrectionDialog(
            corr,
            scene_dir,
            parent=self,
            point_source=self._collect_all_points,
        )
        dlg.correction_changed.connect(self._apply_correction)
        dlg.save_requested.connect(self._on_save_correction_to_project)
        dlg.diagnostics_ready.connect(self._on_diag_ready)
        dlg.destroyed.connect(self._on_correction_dlg_closed)
        self._correction_dlg = dlg
        dlg.show()

    def _apply_correction(self, c: SceneCorrection):
        """Apply correction values from dialog to viewport (live preview)."""
        self.gl_viewport.scene_correction = c
        self.gl_viewport.update()

    def _on_diag_ready(self, diag):
        """Show auto-detect debug overlays in the viewport."""
        self.gl_viewport.set_correction_diagnostics(diag)

    def _on_correction_dlg_closed(self):
        """Clear diagnostics and dialog reference when dialog closes."""
        self._correction_dlg = None
        self.gl_viewport.set_correction_diagnostics(None)

    def closeEvent(self, event):
        """Ensure background threads are stopped before window destruction."""
        if hasattr(self, "_correction_dlg") and self._correction_dlg is not None:
            self._correction_dlg.close()  # triggers its closeEvent → worker shutdown
            self._correction_dlg = None
        super().closeEvent(event)

    def _on_save_correction_to_project(self):
        """Save current state (including correction) to the project YAML."""
        if not self._yaml_path:
            # No YAML path yet — trigger Save As
            self._on_save_yaml_as()
        else:
            self._save_yaml(self._yaml_path)
            self.status_label.setText(
                f"Correction saved to {Path(self._yaml_path).name}"
            )

    def _collect_all_points(self):
        """Collect visible point cloud data for auto-detection.

        Returns a subsampled (~2M pts) float64 array.  Avoids the
        30-second stall that occurred when converting the full 348M-point
        raw_scan from F-order float32 to float64.
        """
        import numpy as np

        # Prefer the mid-res layer (already ~5M pts) over raw_scan (348M)
        target = 2_000_000
        all_pts = []
        seen_ids = set()
        for layer in self.layer_manager.layers:
            if not layer.visible:
                continue
            if not hasattr(layer, "points") or layer.points is None:
                continue
            # Skip raw_scan if we already have mid-res (same data, fewer pts)
            if layer.id == "raw_scan" and "midres" in seen_ids:
                continue
            if layer.id == "midres":
                seen_ids.add("midres")
            pts = layer.points
            # Stride-subsample large layers to ~target pts total
            if len(pts) > target:
                stride = max(1, len(pts) // target)
                pts = pts[::stride]
            all_pts.append(np.ascontiguousarray(pts, dtype=np.float64))
        if not all_pts:
            return None
        return np.vstack(all_pts) if len(all_pts) > 1 else all_pts[0]

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
                print(f"Project file found: {candidate.name}")
                self._load_yaml(str(candidate))
                return
        print(f"  No project file found for: {p.name}")

    def _on_toggle_layer_colors(self, checked):
        """Toggle per-layer color tinting — evicts caches and VBOs for full re-upload."""
        self.gl_viewport.use_layer_colors = checked
        for l in self.layer_manager.layers:
            l.evict_byte_caches()
            self.gl_viewport.delete_vbos_for_layer(l.id)
        self.gl_viewport.update()

    def _on_fps_camera_toggled(self, checked):
        self.gl_viewport.set_fps_camera(checked)
        self.act_fps_movement.setChecked(self.gl_viewport.fps_movement)

    def _toggle_view(self, attr, checked):
        """Generic toggle for viewport boolean attributes (show_axes, show_grid, etc.)."""
        setattr(self.gl_viewport, attr, checked)
        self.gl_viewport.update()

    def _on_point_size(self, val):
        """Update GL point size from the toolbar slider (range 1–20)."""
        self.gl_viewport.point_size = val
        self.gl_viewport.update()

    def _on_camera_preset(self, name):
        """Set camera to a named preset (Top/Front/Right/Isometric). Distance unchanged."""
        vp = self.gl_viewport
        presets = {
            "Top": (0, 89),
            "Front": (0, 0),
            "Right": (90, 0),
            "Isometric": (45, 30),
        }
        if name in presets:
            vp.cam_azimuth, vp.cam_elevation = presets[name]
            vp.update()

    def _on_reset_view(self):
        """Fit view to selected bbox or entire scene."""
        vp = self.gl_viewport
        idx = vp.selected_idx
        if idx >= 0 and idx < len(self.annotations):
            bbox = self.annotations[idx]
            vp.cam_target = bbox.center_pos.copy()
            extent = np.linalg.norm(bbox.size) * 1.5
            vp.cam_distance = max(extent, 1.0)
        else:
            center, radius = self.layer_manager.get_scene_bounds()
            vp.cam_target = center.copy()
            vp.cam_distance = radius * 2.5
        vp.update()

    def _on_fps_updated(self, fps):
        """Update status bar with camera azimuth/elevation/distance and FPS."""
        vp = self.gl_viewport
        self.cam_label.setText(
            f"Az:{vp.cam_azimuth % 360:.0f} El:{vp.cam_elevation:.0f} D:{vp.cam_distance:.1f}"
        )
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def _check_file_changes(self):
        """Poll loaded files for on-disk changes (hot-reload, every 2s via timer)."""
        any_changed = False
        for layer in self.layer_manager.layers:
            if layer.file_changed_on_disk():
                try:
                    layer.reload()
                    self.gl_viewport.delete_vbos_for_layer(layer.id)
                    any_changed = True
                except Exception:
                    pass
        if any_changed:
            self.layer_panel.rebuild()
            self.gl_viewport.update()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts.

        Scene correction:  WASD/QE/Arrows → forwarded to viewport
        Navigation:        Escape → exit panorama or deselect bbox
        Annotation:        Delete/Backspace → delete selected bbox
                           N → create bbox at target
                           Ctrl+Z → undo, Ctrl+D → duplicate
        Blender tools:     Q=Select, G=Move, R=Rotate, S=Scale
        Axis constraints:  X/Y/Z → toggle axis constraint for gizmo
        """
        key = event.key()
        mods = event.modifiers()

        # Scene correction keys: WASD+QE+arrows → route to viewport first
        _CORRECTION_KEYS = {
            Qt.Key.Key_W,
            Qt.Key.Key_A,
            Qt.Key.Key_S,
            Qt.Key.Key_D,
            Qt.Key.Key_Q,
            Qt.Key.Key_E,
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }
        if key in _CORRECTION_KEYS:
            self.gl_viewport.keyPressEvent(event)
            return

        if key == Qt.Key.Key_Escape:
            # Exit panorama mode first if active
            if (
                hasattr(self.gl_viewport, "_panorama")
                and self.gl_viewport._panorama
                and self.gl_viewport._panorama.is_active
            ):
                self.gl_viewport.exit_panorama()
                self.layer_panel.highlight_active_pano(None)
                return
            if self.gl_viewport.selected_idx >= 0:
                self.gl_viewport.selected_idx = -1
                self.gl_viewport._hovered_gizmo = None
                self.gl_viewport.setCursor(Qt.CursorShape.ArrowCursor)
                self.gl_viewport.bbox_selected.emit(-1)
                self.gl_viewport.update()
                return

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.gl_viewport.selected_idx >= 0:
                self._delete_bbox(self.gl_viewport.selected_idx)
                return

        # Tool shortcuts (Blender-like)
        if key == Qt.Key.Key_Q and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._set_tool(TOOL_SELECT)
            return
        if key == Qt.Key.Key_G and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._set_tool(TOOL_MOVE)
            return
        if key == Qt.Key.Key_R and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._set_tool(TOOL_ROTATE)
            return
        if key == Qt.Key.Key_S and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._set_tool(TOOL_SCALE)
            return

        # Axis constraints (toggle)
        if key == Qt.Key.Key_X and not (mods & Qt.KeyboardModifier.ControlModifier):
            cur = self.gl_viewport.axis_constraint
            self._set_axis(0 if cur != 0 else None)
            return
        if key == Qt.Key.Key_Y and not (mods & Qt.KeyboardModifier.ControlModifier):
            cur = self.gl_viewport.axis_constraint
            self._set_axis(1 if cur != 1 else None)
            return
        if key == Qt.Key.Key_Z and not (mods & Qt.KeyboardModifier.ControlModifier):
            cur = self.gl_viewport.axis_constraint
            self._set_axis(2 if cur != 2 else None)
            return

        if key == Qt.Key.Key_N and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._create_bbox_at_target()
            return
        if key == Qt.Key.Key_Z and (mods & Qt.KeyboardModifier.ControlModifier):
            self._undo()
            return
        if key == Qt.Key.Key_D and (mods & Qt.KeyboardModifier.ControlModifier):
            self.bbox_panel._on_duplicate()
            return

        super().keyPressEvent(event)
