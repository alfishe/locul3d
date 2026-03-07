"""Locul3D Editor — main window with full annotation workflow."""

import sys
import copy
import json
from pathlib import Path
from typing import Optional, List

import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QStatusBar,
    QLabel, QSlider, QComboBox, QDockWidget, QFileDialog,
    QApplication,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QKeyEvent

from ..core.layer import LayerManager, LayerData
from ..core.geometry import BBoxItem, PlaneItem
from ..core.constants import (
    COLORS, BBOX_COLORS, PLANE_COLORS, DEFAULT_SIZES,
    TOOL_SELECT, TOOL_MOVE, TOOL_ROTATE, TOOL_SCALE,
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


class EditorWindow(QMainWindow):
    """BBox Annotation Editor — full-featured main window."""

    def __init__(self, files=None, annotations_path=None, correction_angles=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Locul3D — Editor")
        self.resize(1500, 900)

        self._yaml_path: Optional[str] = annotations_path
        self.annotations: List[BBoxItem] = []
        self.planes: List[PlaneItem] = []
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
        self.gl_viewport.fps_updated.connect(self._on_fps_updated)
        self.layer_panel.layer_changed.connect(self._on_layer_changed)
        self.layer_panel.layer_selected.connect(self._on_layer_selected)
        self.layer_panel.opacity_adjusting.connect(
            self._on_opacity_adjusting)
        self.layer_panel.pano_requested.connect(self._on_pano_requested)

        self.gl_viewport.point_picked.connect(self._on_point_picked)
        self.gl_viewport.bbox_selected.connect(self._on_bbox_selected)
        self.gl_viewport.bbox_moved.connect(self._on_bbox_moved)
        self.gl_viewport.transform_committed.connect(self._on_transform_committed)

        # Marker click in viewport → select in layer panel (no info panel)
        self.gl_viewport.marker_selected.connect(
            self.layer_panel.select_layer_by_data)
        # Marker double-click → select + open info panel
        self.gl_viewport.marker_activated.connect(
            lambda layer: self.layer_panel.select_layer_by_data(layer, notify=True))

        self.bbox_panel.bbox_changed.connect(self._on_bbox_panel_changed)
        self.bbox_panel.selection_changed.connect(self._on_bbox_panel_selection)
        self.bbox_panel.delete_requested.connect(self._delete_bbox)
        self.bbox_panel.create_requested.connect(self._create_bbox_at_target)
        self.bbox_panel.tool_changed.connect(self._set_tool)
        self.bbox_panel.axis_changed.connect(self._set_axis)

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

        # Deferred file loading
        if files:
            self._deferred_files = files
            self._deferred_yaml = annotations_path
            QTimer.singleShot(100, self._deferred_load)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_viewport)
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
        act_fit.triggered.connect(self.gl_viewport.fit_to_scene)
        toolbar.addAction(act_fit)

        act_reset = QAction("Reset View", self)
        act_reset.setShortcut("Home")
        act_reset.setToolTip("Fit view to selected bbox (or scene) with current projection")
        act_reset.triggered.connect(self._on_reset_view)
        toolbar.addAction(act_reset)

        toolbar.addSeparator()

        self.act_ref_panel = QAction("Ref/Coords", self, checkable=True)
        self.act_ref_panel.setToolTip("Toggle Reference & Coordinates panel")
        self.act_ref_panel.triggered.connect(
            lambda c: self._ref_dock.setVisible(c))
        toolbar.addAction(self.act_ref_panel)

        act_ground = QAction("Ground Z=0", self, checkable=True)
        act_ground.setToolTip("Show global reference plane at Z=0 (cyan wireframe)")
        act_ground.triggered.connect(
            lambda c: self._toggle_view('show_ground_plane', c))
        toolbar.addAction(act_ground)

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

        # BBox dock
        self._bbox_dock = QDockWidget("BBox Annotations", self)
        self._bbox_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._bbox_dock.setMinimumWidth(320)
        self.bbox_panel = BBoxPanel(self.annotations)
        self._bbox_dock.setWidget(self.bbox_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._bbox_dock)

        # Planes dock
        self._planes_dock = QDockWidget("Surface Planes", self)
        self._planes_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._planes_dock.setMinimumWidth(300)
        self.plane_panel = PlanePanel(self.planes)
        self._planes_dock.setWidget(self.plane_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._planes_dock)

        # Info dock
        self._info_dock = QDockWidget("Info", self)
        self._info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._info_dock.setMinimumWidth(280)
        self.info_panel = InfoPanel()
        self._info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

        # Reference & Coordinates dock (floating by default)
        self._ref_dock = QDockWidget("Reference & Coordinates", self)
        self._ref_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._ref_dock.setMinimumWidth(240)
        self.ref_panel = ReferencePanel()
        self._ref_dock.setWidget(self.ref_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._ref_dock)
        self._ref_dock.setFloating(True)
        self._ref_dock.resize(260, 300)
        self._ref_dock.hide()  # hidden by default
        self._ref_dock.visibilityChanged.connect(
            lambda vis: self.act_ref_panel.setChecked(vis))

        # Tabify docks on the right
        self.tabifyDockWidget(self._layers_dock, self._bbox_dock)
        self.tabifyDockWidget(self._bbox_dock, self._planes_dock)
        self.tabifyDockWidget(self._planes_dock, self._info_dock)
        self._layers_dock.raise_()

    def _setup_statusbar(self):
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

    def _set_tool(self, tool):
        self.gl_viewport.tool = tool
        self.bbox_panel.set_tool(tool)
        self._update_status()
        self.gl_viewport.update()

    def _set_axis(self, axis):
        self.gl_viewport.axis_constraint = axis
        self.bbox_panel.set_axis(axis)
        self._update_status()
        self.gl_viewport.update()

    def _update_status(self):
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

    def _deferred_load(self):
        """Load files passed via constructor after the event loop starts."""
        files = getattr(self, '_deferred_files', [])
        yaml_path = getattr(self, '_deferred_yaml', None)
        for arg in files:
            p = Path(arg)
            if p.is_dir():
                self._load_folder(str(p))
            elif p.is_file():
                if p.suffix.lower() == '.e57':
                    self._import_e57_file(str(p))
                else:
                    self._load_file(str(p), fit_camera=False)
        self.gl_viewport.fit_to_scene()
        # Compute ceiling height silently in background (cached, not applied)
        QTimer.singleShot(0, self.layer_manager.compute_ceiling_background)
        if yaml_path and Path(yaml_path).exists():
            self._load_yaml(yaml_path)
        self._update_status()

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

    def _import_e57_file(self, path: str):
        """Import E57 file through the full processing pipeline."""

        # --- Correction sidecar detection ---
        scene_p = Path(path).resolve()
        print(f"[E57] Searching for correction sidecar in: {scene_p.parent}/")
        sidecar = SceneCorrection.find_sidecar(str(scene_p))
        correction = None
        if sidecar:
            try:
                correction = SceneCorrection.load_yaml(sidecar)
                cli = self._cli_correction
                if cli.get('rotate_x', 0): correction.rotate_x = cli['rotate_x']
                if cli.get('rotate_y', 0): correction.rotate_y = cli['rotate_y']
                if cli.get('rotate_z', 0): correction.rotate_z = cli['rotate_z']
                if cli.get('shift_x', 0): correction.shift_x = cli['shift_x']
                if cli.get('shift_y', 0): correction.shift_y = cli['shift_y']
                if cli.get('shift_z', 0): correction.shift_z = cli['shift_z']
                print(f"[E57] Correction LOADED from: {sidecar}")
                print(f"[E57]   rot=({correction.rotate_x}, {correction.rotate_y}, {correction.rotate_z})°  "
                      f"shift=({correction.shift_x}, {correction.shift_y}, {correction.shift_z})")
                self.status_label.setText(f"Correction loaded: {Path(sidecar).name}")
            except Exception as e:
                print(f"[E57] WARNING: failed to load correction sidecar: {e}")
                correction = None
        else:
            print(f"[E57] No correction sidecar found for: {scene_p.name}")

        # --- E57 import ---
        try:
            from ..plugins.importers.e57 import (
                E57ImportWorker, E57ProgressDialog, E57Importer,
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
            self.gl_viewport.fit_to_scene()
            if result.metadata or result.stats:
                self.info_panel.populate(result.metadata, result.stats)
            n_layers = len(result.layers)
            total_pts = sum(l.point_count for l in result.layers)
            self.status_label.setText(
                f"Imported E57: {n_layers} layers, {total_pts:,} points")

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
            self.gl_viewport.fit_to_scene()
        last = self.layer_manager.layers[-1] if self.layer_manager.layers else None
        if last:
            self.info_panel.show_layer_info(last)
        self.status_label.setText(f"Loaded {Path(path).name}")
        # Auto-detect correction sidecar
        self._try_load_sidecar(path)

    def _load_folder(self, folder: str):
        folder_path = Path(folder)
        ply_files = sorted(folder_path.glob("*.ply"))
        for p in ply_files:
            self._load_file(str(p), fit_camera=False)
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
        self.status_label.setText(f"Loaded folder: {folder_path.name}")

    # ------------------------------------------------------------------
    # BBox operations
    # ------------------------------------------------------------------

    def _create_bbox_at_position(self, x, y, z):
        label = self.bbox_panel.label_combo.currentText() or "mts_column"
        size = DEFAULT_SIZES.get(label, DEFAULT_SIZES["custom"]).copy()
        # Place bottom-center at picked point
        center = np.array([x, y, z + size[2] / 2.0], dtype=np.float64)
        color = list(BBOX_COLORS[self._color_idx % len(BBOX_COLORS)])
        self._color_idx += 1
        bbox = BBoxItem(label=label, center=center, size=size, color=color)
        self.annotations.append(bbox)
        self._push_undo('create', {'idx': len(self.annotations) - 1})
        self.bbox_panel.rebuild_list()
        idx = len(self.annotations) - 1
        self.bbox_panel.select_bbox(idx)
        self.gl_viewport.selected_idx = idx
        self.gl_viewport.update()
        self.status_label.setText(f"Created [{idx}] {label} at ({x:.2f}, {y:.2f}, {z:.2f})")

    def _create_bbox_at_target(self):
        t = self.gl_viewport.cam_target
        self._create_bbox_at_position(float(t[0]), float(t[1]), float(t[2]))

    def _delete_bbox(self, idx):
        if idx < 0 or idx >= len(self.annotations):
            return
        self._push_undo('delete', {
            'idx': idx, 'bbox': copy.deepcopy(self.annotations[idx])})
        self.annotations.pop(idx)
        self.gl_viewport.selected_idx = -1
        self.bbox_panel.rebuild_list()
        self.bbox_panel.select_bbox(-1)
        self.gl_viewport.update()
        self.status_label.setText(f"Deleted bbox [{idx}]")

    def _undo(self):
        action = self._pop_undo()
        if action is None:
            return
        act_type, data = action
        if act_type == 'create':
            idx = data['idx']
            if idx < len(self.annotations):
                self.annotations.pop(idx)
        elif act_type == 'delete':
            self.annotations.insert(data['idx'], data['bbox'])
        elif act_type == 'transform':
            idx = data['idx']
            if idx < len(self.annotations):
                b = self.annotations[idx]
                b.center_pos = data['center']
                b.size = data['size']
                b.rotation_z = data['rotation_z']
        self.gl_viewport.selected_idx = -1
        self.bbox_panel.rebuild_list()
        self.bbox_panel.select_bbox(-1)
        self.gl_viewport.update()
        self.status_label.setText("Undo")

    def _push_undo(self, action_type, data):
        self._undo_stack.append((action_type, data))

    def _pop_undo(self):
        if not self._undo_stack:
            return None
        return self._undo_stack.pop()

    def _on_transform_committed(self, idx, snapshot):
        """Called before a gizmo drag starts — push pre-drag state for undo."""
        self._push_undo('transform', {
            'idx': idx,
            'center': snapshot['center'],
            'size': snapshot['size'],
            'rotation_z': snapshot['rotation_z'],
        })

    # ------------------------------------------------------------------
    # Plane operations
    # ------------------------------------------------------------------

    def _create_plane(self):
        t = self.gl_viewport.cam_target
        color = list(PLANE_COLORS[self._plane_color_idx % len(PLANE_COLORS)])
        self._plane_color_idx += 1
        # Transform center through correction so X,Y match the visual scene
        sc = self.gl_viewport.scene_correction
        center = sc.transform_point([float(t[0]), float(t[1]), 0.0])
        center[2] = 0.0  # keep Z=0 for floor-level reference
        plane = PlaneItem(axis='xy', center=center.tolist(),
                          size=[5.0, 5.0], color=color, opacity=0.25)
        self.planes.append(plane)
        self.plane_panel.rebuild_list()
        idx = len(self.planes) - 1
        self.plane_panel.select_plane(idx)
        self.gl_viewport.update()
        self._planes_dock.raise_()
        self.status_label.setText(f"Created plane [{idx}] at target")

    def _delete_plane(self, idx):
        if idx < 0 or idx >= len(self.planes):
            return
        self.planes.pop(idx)
        self.plane_panel.rebuild_list()
        self.plane_panel.select_plane(-1)
        self.gl_viewport.update()
        self.status_label.setText(f"Deleted plane [{idx}]")

    def _on_plane_changed(self, idx):
        self.gl_viewport.update()

    # ------------------------------------------------------------------
    # Reference point & Coordinates
    # ------------------------------------------------------------------

    def _on_set_ref_point(self):
        self.gl_viewport._picking_ref_point = True
        self.status_label.setText("Click on point cloud to set reference origin...")

    def _on_clear_ref_point(self):
        self._ref_point = None
        self.gl_viewport.ref_point = None
        self.ref_panel.clear_ref_point()
        self.gl_viewport.update()
        self._refresh_bbox_panel_coords()
        self.status_label.setText("Reference point cleared")

    def _on_ref_point_picked(self, x, y, z):
        self._ref_point = np.array([x, y, z], dtype=np.float64)
        self.gl_viewport.ref_point = self._ref_point
        self.ref_panel.set_ref_point(x, y, z)
        self.gl_viewport.update()
        self._refresh_bbox_panel_coords()
        self.status_label.setText(f"Reference point set to ({x:.2f}, {y:.2f}, {z:.2f})")

    def _on_coord_mode_changed(self, index):
        self._coord_mode = "scene" if index == 0 else "relative"
        self._refresh_bbox_panel_coords()

    def _refresh_bbox_panel_coords(self):
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
        data = {
            "default_column_size": DEFAULT_SIZES["mts_column"].tolist(),
            "default_box_size": DEFAULT_SIZES["mts_box"].tolist(),
            "bboxes": [b.to_dict() for b in self.annotations],
        }
        if self.planes:
            data["planes"] = [p.to_dict() for p in self.planes]
        if self._ref_point is not None:
            data["reference_point"] = [round(float(v), 4) for v in self._ref_point]
        if HAS_YAML:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=None, sort_keys=False,
                          allow_unicode=True)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        self._yaml_path = path
        self.status_label.setText(
            f"Saved {len(self.annotations)} bboxes + {len(self.planes)} planes to {Path(path).name}")
        self.setWindowTitle(f"Locul3D Editor — {Path(path).name}")

    def _load_yaml(self, path: str):
        with open(path) as f:
            if path.endswith((".yaml", ".yml")) and HAS_YAML:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
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
        self._yaml_path = path
        self.bbox_panel.rebuild_list()
        self.plane_panel.rebuild_list()
        self.gl_viewport.selected_idx = -1
        self.gl_viewport.update()
        self.status_label.setText(
            f"Loaded {len(self.annotations)} bboxes + {len(self.planes)} planes from {Path(path).name}")
        self.setWindowTitle(f"Locul3D Editor — {Path(path).name}")

    def _on_save_yaml(self):
        if self._yaml_path:
            self._save_yaml(self._yaml_path)
        else:
            self._on_save_yaml_as()

    def _on_save_yaml_as(self):
        ext = "YAML (*.yaml *.yml)" if HAS_YAML else "JSON (*.json)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", f"{ext};;All Files (*)")
        if path:
            self._save_yaml(path)

    def _on_load_yaml(self):
        ext = "YAML/JSON (*.yaml *.yml *.json)" if HAS_YAML else "JSON (*.json)"
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", f"{ext};;All Files (*)")
        if path:
            self._load_yaml(path)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_point_picked(self, x, y, z):
        # Route to ref-point handler if in picking mode
        if getattr(self.gl_viewport, '_picking_ref_point', False):
            self.gl_viewport._picking_ref_point = False
            self._on_ref_point_picked(x, y, z)
            return
        self._create_bbox_at_position(x, y, z)

    def _on_bbox_selected(self, idx):
        self.bbox_panel.select_bbox(idx)
        self._bbox_dock.raise_()

    def _on_bbox_moved(self, idx):
        self.bbox_panel.update_values(idx)

    def _on_bbox_panel_changed(self, idx):
        self.gl_viewport.update()

    def _on_bbox_panel_selection(self, idx):
        self.gl_viewport.selected_idx = idx
        self.gl_viewport.update()

    # ------------------------------------------------------------------
    # Layer handlers
    # ------------------------------------------------------------------

    def _on_layer_changed(self):
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
            f"Scene clip: X=[{x0:.1f},{x1:.1f}] Y=[{y0:.1f},{y1:.1f}] Z=[{z0:.1f},{z1:.1f}]")

    def _on_scene_correction(self):
        """Open the Scene Correction dialog for live rotation/shift adjustment."""
        scene_dir = self.layer_manager.base_dir or ""
        dlg = CorrectionDialog(self.gl_viewport.scene_correction, scene_dir, self)
        dlg.correction_changed.connect(self._apply_correction)
        dlg.exec()

    def _apply_correction(self, c: SceneCorrection):
        """Apply correction values from dialog to viewport (live preview)."""
        self.gl_viewport.scene_correction = c
        self.gl_viewport.update()

    def _try_load_sidecar(self, scene_path: str):
        """Auto-detect and load a correction YAML sidecar next to a scene file."""
        p = Path(scene_path).resolve()  # resolve to absolute path
        print(f"Searching for correction sidecar in: {p.parent}/")
        sidecar = SceneCorrection.find_sidecar(str(p))
        if sidecar is None:
            print(f"  No correction sidecar found for: {p.name}")
            return
        try:
            c = SceneCorrection.load_yaml(sidecar)
            cli = self._cli_correction
            if cli.get('rotate_x', 0): c.rotate_x = cli['rotate_x']
            if cli.get('rotate_y', 0): c.rotate_y = cli['rotate_y']
            if cli.get('rotate_z', 0): c.rotate_z = cli['rotate_z']
            if cli.get('shift_x', 0): c.shift_x = cli['shift_x']
            if cli.get('shift_y', 0): c.shift_y = cli['shift_y']
            if cli.get('shift_z', 0): c.shift_z = cli['shift_z']
            self.gl_viewport.scene_correction = c
            self.gl_viewport.update()
            print(f"Scene correction loaded from: {sidecar}")
            print(f"  rot=({c.rotate_x}, {c.rotate_y}, {c.rotate_z})°  "
                  f"shift=({c.shift_x}, {c.shift_y}, {c.shift_z})")
            self.status_label.setText(f"Correction loaded: {Path(sidecar).name}")
        except Exception as e:
            print(f"Warning: failed to load correction sidecar: {e}")

    def _on_toggle_layer_colors(self, checked):
        self.gl_viewport.use_layer_colors = checked
        for l in self.layer_manager.layers:
            l.evict_byte_caches()
            self.gl_viewport.delete_vbos_for_layer(l.id)
        self.gl_viewport.update()

    def _toggle_view(self, attr, checked):
        setattr(self.gl_viewport, attr, checked)
        self.gl_viewport.update()

    def _on_point_size(self, val):
        self.gl_viewport.point_size = val
        self.gl_viewport.update()

    def _on_camera_preset(self, name):
        vp = self.gl_viewport
        presets = {"Top": (0, 89), "Front": (0, 0), "Right": (90, 0), "Isometric": (45, 30)}
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
        vp = self.gl_viewport
        self.cam_label.setText(
            f"Az:{vp.cam_azimuth%360:.0f} El:{vp.cam_elevation:.0f} D:{vp.cam_distance:.1f}")
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def _check_file_changes(self):
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
        key = event.key()
        mods = event.modifiers()

        # Scene correction keys: WASD+QE+arrows → route to viewport first
        _CORRECTION_KEYS = {
            Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
            Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
        }
        if key in _CORRECTION_KEYS:
            self.gl_viewport.keyPressEvent(event)
            return

        if key == Qt.Key.Key_Escape:
            # Exit panorama mode first if active
            if (hasattr(self.gl_viewport, '_panorama')
                    and self.gl_viewport._panorama
                    and self.gl_viewport._panorama.is_active):
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
