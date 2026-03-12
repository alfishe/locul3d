"""Info panel widget for displaying layer details and E57 metadata."""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ...core.constants import COLORS


class InfoPanel(QWidget):
    """Dockable widget showing metadata and processing stats in sections."""

    edit_wireframe_requested = Signal(object)  # emits LayerData

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(6)
        self._scroll.setWidget(self._scroll_content)
        self._layout.addWidget(self._scroll)

        # Placeholder text
        self._placeholder = QLabel("  No data loaded.\n  Import a file to see info.")
        self._placeholder.setStyleSheet(
            f"color: {COLORS.get('text_muted', '#888')}; font-size: 11px; padding: 12px;")
        self._scroll_layout.addWidget(self._placeholder)
        self._scroll_layout.addStretch()

    def clear(self):
        """Remove all content."""
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def populate(self, metadata: dict, stats: dict):
        """Build the info panel from E57 metadata and pipeline stats."""
        self.clear()

        # File Info
        self._add_section("File Info", [
            ("File", metadata.get("file_name", "N/A")),
            ("Format", metadata.get("format", "N/A")),
            ("Size", f"{metadata.get('file_size_mb', 0):.1f} MB"),
            ("Scans", str(metadata.get("scan_count", "N/A"))),
            ("Raw Points", f"{metadata.get('raw_point_count', 0):,}"),
            ("Fields", ", ".join(metadata.get("point_fields", []))),
        ])

        # Scan Pose
        pose = metadata.get("scan_pose", {})
        translation = pose.get("translation")
        rotation = pose.get("rotation_quaternion")
        pose_rows = []
        if translation:
            pose_rows.append(("Position", self._fmt_vec(translation, 2)))
        if rotation:
            pose_rows.append(("Quaternion", self._fmt_vec(rotation, 4)))
        scanner_pos = metadata.get("scanner_position_original")
        if scanner_pos:
            pose_rows.append(("Scanner (orig)", self._fmt_vec(scanner_pos, 2)))
        if pose_rows:
            self._add_section("Scan Pose", pose_rows)

        # Color Info
        color_rows = []
        if metadata.get("has_rgb"):
            color_rows.append(("Color Mode", "RGB (per-vertex)"))
        elif metadata.get("has_intensity"):
            color_rows.append(("Color Mode", "Intensity (mapped)"))
        else:
            color_rows.append(("Color Mode", "None"))
        if color_rows:
            self._add_section("Color", color_rows)

        # Ingestion Stats
        self._add_section("Ingestion", [
            ("Crop Radius", f"{metadata.get('crop_radius_m', 0)} m"),
            ("After Crop", f"{metadata.get('cropped_point_count', 0):,}"),
            ("Bounding Box", self._fmt_vec(metadata.get("bounding_box_m", []), 1) + " m"),
            ("BB Min", self._fmt_vec(metadata.get("bb_min", []), 2)),
            ("BB Max", self._fmt_vec(metadata.get("bb_max", []), 2)),
        ])

        # Processing Stats
        pipeline_rows = [
            ("Points After Filter", f"{stats.get('points_after_filter', 0):,}"),
            ("Points After Align", f"{stats.get('points_after_align', 0):,}"),
            ("Points Decimated", f"{stats.get('points_after_decimate', 0):,}"),
        ]
        self._add_section("Processing", pipeline_rows)

        # Timing
        timing_rows = []
        stage_names = [
            ("ingest_time", "Ingestion"),
            ("filter_time", "Filtering"),
            ("align_time", "Alignment"),
            ("decimate_time", "Decimation"),
            ("build_time", "Build Layers"),
        ]
        for key, label in stage_names:
            t = stats.get(key)
            if t is not None:
                timing_rows.append((label, f"{t:.2f}s"))
        total = stats.get("total_time")
        if total is not None:
            timing_rows.append(("Total", f"{total:.1f}s"))
        if timing_rows:
            self._add_section("Timing", timing_rows)

        self._scroll_layout.addStretch()

    def _show_pointcloud_info(self, layer):
        """Show detailed information for pointcloud layers."""
        geom_rows = [
            ("Points", f"{layer.point_count:,}"),
        ]
        has_vtx_rgb = layer.colors is not None and len(layer.colors) > 0
        geom_rows.append(("Vertex Colors", "Yes" if has_vtx_rgb else "No"))
        has_normals = layer.normals is not None and len(layer.normals) > 0
        geom_rows.append(("Normals", "Yes" if has_normals else "No"))

        if layer.points is not None and len(layer.points) > 0:
            bbox_min = layer.points.min(axis=0)
            bbox_max = layer.points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            bbox_center = (bbox_min + bbox_max) / 2
            geom_rows.append(("Bbox Size", f"({bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}) m"))
            geom_rows.append(("Bbox Center", f"({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f})"))
            geom_rows.append(("Bbox Min", f"({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f})"))
            geom_rows.append(("Bbox Max", f"({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})"))
            volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
            if volume > 0:
                density = layer.point_count / volume
                geom_rows.append(("Density", f"{density:.1f} pts/m³"))

        self._add_section("Point Cloud", geom_rows)

    def _show_mesh_info(self, layer):
        """Show detailed information for mesh layers."""
        geom_rows = [
            ("Vertices", f"{layer.point_count:,}"),
            ("Triangles", f"{layer.tri_count:,}"),
        ]
        has_vtx_rgb = layer.colors is not None and len(layer.colors) > 0
        geom_rows.append(("Vertex Colors", "Yes" if has_vtx_rgb else "No"))
        has_normals = layer.normals is not None and len(layer.normals) > 0
        geom_rows.append(("Normals", "Yes" if has_normals else "No"))

        if layer.points is not None and len(layer.points) > 0:
            bbox_min = layer.points.min(axis=0)
            bbox_max = layer.points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            bbox_center = (bbox_min + bbox_max) / 2
            geom_rows.append(("Bbox Size", f"({bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}) m"))
            geom_rows.append(("Bbox Center", f"({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f})"))

            if layer.triangles is not None and len(layer.triangles) > 0:
                total_area = 0.0
                for tri in layer.triangles:
                    v0, v1, v2 = layer.points[tri[0]], layer.points[tri[1]], layer.points[tri[2]]
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    cross = np.cross(edge1, edge2)
                    total_area += np.linalg.norm(cross) * 0.5
                geom_rows.append(("Surface Area", f"{total_area:.2f} m²"))

        self._add_section("Mesh", geom_rows)

    def _show_wireframe_info(self, layer):
        """Show detailed information for wireframe layers (OBB boxes)."""
        geom_rows = []
        if layer.line_points is not None:
            num_lines = len(layer.line_points) // 2
            geom_rows.append(("Lines", f"{num_lines}"))
            geom_rows.append(("Vertices", f"{len(layer.line_points)}"))

        meta = layer.meta
        if "center" in meta:
            center = meta["center"]
            geom_rows.append(("Center", f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"))
        if "extent" in meta:
            extent = meta["extent"]
            geom_rows.append(("Extent", f"({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}) m"))
            volume = extent[0] * extent[1] * extent[2]
            geom_rows.append(("Volume", f"{volume:.2f} m³"))
        if "rotation" in meta:
            geom_rows.append(("Rotation", "3×3 matrix"))

        if layer.line_points is not None and len(layer.line_points) > 0:
            bbox_min = layer.line_points.min(axis=0)
            bbox_max = layer.line_points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            geom_rows.append(("Bbox Size", f"({bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}) m"))

        self._add_section("Wireframe (OBB)", geom_rows)

    def _add_section(self, title: str, rows):
        """Add a titled section with key-value rows."""
        header = QLabel(f"  {title}")
        header.setStyleSheet(
            f"color: {COLORS.get('text_muted', '#888')}; font-size: 11px; font-weight: bold; "
            f"padding: 4px 0 2px 0; border-bottom: 1px solid {COLORS.get('border', '#444')};"
        )
        self._scroll_layout.addWidget(header)

        for key, value in rows:
            row = QHBoxLayout()
            row.setContentsMargins(8, 1, 4, 1)
            row.setSpacing(8)

            k_label = QLabel(key)
            k_label.setFixedWidth(110)
            k_label.setStyleSheet(
                f"color: {COLORS.get('text_secondary', '#aaa')}; font-size: 11px;")
            k_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            row.addWidget(k_label)

            v_label = QLabel(str(value))
            v_label.setStyleSheet(
                f"color: {COLORS.get('text', '#eee')}; font-size: 11px;")
            v_label.setWordWrap(True)
            row.addWidget(v_label, stretch=1)

            container = QWidget()
            container.setLayout(row)
            self._scroll_layout.addWidget(container)

    def _show_panorama_info(self, layer):
        """Show detailed information for panorama layers."""
        import math
        rows = []

        # Type
        ptype = getattr(layer, 'pano_type', None)
        if ptype:
            rows.append(("Panorama Type", ptype.capitalize()))

        # Station position
        pos = getattr(layer, 'pano_position', None)
        if pos is not None:
            rows.append(("Station X", f"{pos[0]:.3f} m"))
            rows.append(("Station Y", f"{pos[1]:.3f} m"))
            rows.append(("Station Z", f"{pos[2]:.3f} m"))

        # Station rotation
        rot = getattr(layer, 'pano_rotation', None)
        if rot is not None:
            w, x, y, z = rot
            rows.append(("Quaternion", f"({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})"))
            # Euler angles (ZYX convention)
            yaw = math.degrees(math.atan2(2 * (w * z + x * y),
                                          1 - 2 * (y * y + z * z)))
            pitch = math.degrees(math.asin(max(-1, min(1,
                                2 * (w * y - z * x)))))
            roll = math.degrees(math.atan2(2 * (w * x + y * z),
                                           1 - 2 * (x * x + y * y)))
            rows.append(("Yaw", f"{yaw:.1f}°"))
            rows.append(("Pitch", f"{pitch:.1f}°"))
            rows.append(("Roll", f"{roll:.1f}°"))

        # Image dimensions
        equirect = getattr(layer, 'pano_equirect', None)
        image_size = getattr(layer, 'pano_image_size', None)
        if equirect is not None:
            rows.append(("Image Size", f"{equirect.size[0]}×{equirect.size[1]}"))
            rows.append(("Image Mode", str(equirect.mode)))
        elif image_size is not None:
            rows.append(("Image Size", f"{image_size[0]}×{image_size[1]}"))

        # Cubemap faces
        faces = getattr(layer, 'pano_faces', None)
        face_bytes = getattr(layer, 'pano_face_bytes', None)
        if faces is not None:
            n_faces = sum(1 for f in faces if f is not None)
            rows.append(("Cubemap Faces", f"{n_faces}/6"))
            first = next((f for f in faces if f is not None), None)
            if first:
                rows.append(("Face Size", f"{first.size[0]}×{first.size[1]}"))
        elif face_bytes is not None:
            n_faces = sum(1 for f in face_bytes if f is not None)
            rows.append(("Cubemap Faces", f"{n_faces}/6"))

        if rows:
            self._add_section("Panorama", rows)

    def show_layer_info(self, layer):
        """Show detailed info for a single selected layer."""
        self.clear()

        rows = [
            ("Name", layer.name),
            ("ID", layer.id),
            ("Type", layer.layer_type.upper()),
        ]
        if layer.load_error:
            rows.append(("Status", f"ERROR: {layer.load_error}"))
        else:
            rows.append(("Status", "Loaded" if layer.loaded else "Not loaded"))
        self._add_section("Layer", rows)

        # Type-specific information
        if layer.layer_type == "pointcloud":
            self._show_pointcloud_info(layer)
        elif layer.layer_type == "mesh":
            self._show_mesh_info(layer)
        elif layer.layer_type == "panorama":
            self._show_panorama_info(layer)
        elif layer.layer_type == "wireframe":
            self._show_wireframe_info(layer)
            edit_btn = QPushButton("Edit Bounds...")
            edit_btn.setFixedHeight(24)
            edit_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS.get('input_bg', '#333')}; "
                f"color: {COLORS.get('text', '#eee')}; "
                f"border: 1px solid {COLORS.get('border', '#444')}; "
                f"border-radius: 4px; font-size: 11px; padding: 2px 12px; }}"
                f"QPushButton:hover {{ background: {COLORS.get('hover', '#555')}; }}"
            )
            edit_btn.clicked.connect(lambda _, l=layer: self.edit_wireframe_requested.emit(l))
            btn_container = QHBoxLayout()
            btn_container.setContentsMargins(8, 4, 8, 4)
            btn_container.addStretch()
            btn_container.addWidget(edit_btn)
            btn_container.addStretch()
            btn_w = QWidget()
            btn_w.setLayout(btn_container)
            self._scroll_layout.addWidget(btn_w)

        # Rendering settings
        render_rows = [
            ("Visible", "Yes" if layer.visible else "No"),
            ("Opacity", f"{layer.opacity * 100:.0f}%"),
        ]
        if layer.color:
            c = layer.color
            render_rows.append(("Layer Color", f"RGB({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"))
        self._add_section("Rendering", render_rows)

        # Bounding Box
        pts = layer.points
        if pts is not None and len(pts) > 0:
            bb_min = pts.min(axis=0)
            bb_max = pts.max(axis=0)
            bb_size = bb_max - bb_min
            center, radius = layer.get_bounds()
            self._add_section("Bounds", [
                ("Min", self._fmt_vec(bb_min, 2)),
                ("Max", self._fmt_vec(bb_max, 2)),
                ("Size", self._fmt_vec(bb_size, 2) + " m"),
                ("Center", self._fmt_vec(center, 2)),
                ("Radius", f"{radius:.2f} m"),
            ])
        elif layer.line_points is not None and len(layer.line_points) > 0:
            center, radius = layer.get_bounds()
            self._add_section("Bounds", [
                ("Center", self._fmt_vec(center, 2)),
                ("Radius", f"{radius:.2f} m"),
            ])

        # Surface metadata
        surface_meta = layer.meta.get("surface_meta")
        if surface_meta:
            sm_rows = [
                ("Orientation", surface_meta.get("orientation", "N/A")),
                ("Inliers", f"{surface_meta.get('num_inliers', 0):,}"),
                ("Area", f"{surface_meta.get('area_m2', 0):.2f} m2"),
                ("Normal", self._fmt_vec(surface_meta.get("normal", []), 4)),
                ("Centroid", self._fmt_vec(surface_meta.get("centroid", []), 3)),
            ]
            mesh_tris = surface_meta.get("mesh_tris")
            if mesh_tris:
                sm_rows.append(("Mesh Tris", f"{mesh_tris:,}"))
            self._add_section("Surface Properties", sm_rows)

        # OBB dimensions
        dims = layer.meta.get("dimensions_m")
        if dims:
            dims_str = " x ".join(f"{d:.2f}" for d in dims)
            self._add_section("Dimensions", [
                ("LxWxH", f"{dims_str} m"),
            ])

        self._scroll_layout.addStretch()

    @staticmethod
    def _fmt_vec(vals, decimals=2):
        """Format a list of numbers as a bracketed vector string."""
        if vals is None or (hasattr(vals, '__len__') and len(vals) == 0):
            return "N/A"
        parts = [f"{float(v):.{decimals}f}" for v in vals]
        return "[" + ", ".join(parts) + "]"
