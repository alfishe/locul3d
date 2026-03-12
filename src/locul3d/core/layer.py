"""Layer data model and management system."""

import os
from pathlib import Path
from typing import Optional

import numpy as np


class LayerData:
    """Holds geometry and metadata for a single layer."""

    def __init__(self, layer_def: dict, base_dir: str):
        self.id: str = layer_def["id"]
        self.name: str = layer_def["name"]
        self.layer_type: str = layer_def["type"]  # "pointcloud", "mesh", "wireframe", "panorama"
        self.visible: bool = layer_def.get("visible", True)
        self.opacity: float = layer_def.get("opacity", 1.0)
        self.color: Optional[list] = layer_def.get("color")  # [R,G,B] 0..1 or None
        self.meta: dict = layer_def

        # Panorama data (only for layer_type == "panorama")
        self.pano_position: Optional[np.ndarray] = None   # xyz of camera
        self.pano_rotation: Optional[tuple] = None         # (x,y,z,w) quaternion
        self.pano_faces: Optional[list] = None             # list of 6 PIL Images (cubemap)
        self.pano_type: Optional[str] = None               # "cubemap" or "spherical"
        self.pano_equirect: Optional[object] = None        # PIL equirect image
        self.pano_jpeg_bytes: Optional[bytes] = None       # compressed JPEG bytes
        self.pano_image_size: Optional[tuple] = None       # (w, h) pixel dimensions
        self.pano_face_bytes: Optional[list] = None        # per-face JPEG bytes (cubemap)

        # Geometry arrays
        self.points: Optional[np.ndarray] = None       # Nx3 float32
        self.colors: Optional[np.ndarray] = None       # Nx3 float32
        self.normals: Optional[np.ndarray] = None      # Nx3 float32
        self.triangles: Optional[np.ndarray] = None    # Mx3 uint32
        self.line_points: Optional[np.ndarray] = None  # Lx3 float32 (wireframe)

        # Stats
        self.point_count: int = 0
        self.tri_count: int = 0
        self.loaded: bool = False
        self.load_error: Optional[str] = None

        # Cached byte buffers for GL client-side arrays
        self._pts_bytes: Optional[bytes] = None
        self._normals_bytes: Optional[bytes] = None
        self._tris_bytes: Optional[bytes] = None
        self._lines_bytes: Optional[bytes] = None
        self._rgba_bytes: Optional[bytes] = None
        self._rgba_array: Optional[np.ndarray] = None
        self._rgba_opacity: Optional[float] = None

        self._base_dir = base_dir
        self._file_path = layer_def.get("file")
        self._file_mtime: float = 0.0  # mtime when last loaded

    # --- Loading ---

    def load(self):
        """Load geometry from disk."""
        if self.loaded:
            return
        try:
            if self.layer_type == "wireframe":
                self._load_wireframe()
            elif self._file_path:
                full_path = os.path.join(self._base_dir, self._file_path)
                if not os.path.exists(full_path):
                    self.load_error = f"File not found: {self._file_path}"
                    self.loaded = True
                    return
                from locul3d.utils.io import load_geometry
                load_geometry(full_path, self)
                self._file_mtime = os.path.getmtime(full_path)
            self.loaded = True
        except Exception as e:
            self.load_error = str(e)
            self.loaded = True

    def file_changed_on_disk(self) -> bool:
        """Check if the source file has been modified since last load."""
        if not self._file_path or not self._file_mtime:
            return False
        full_path = os.path.join(self._base_dir, self._file_path)
        try:
            return os.path.getmtime(full_path) > self._file_mtime
        except OSError:
            return False

    def reload(self):
        """Re-load geometry from disk (hot-reload on file change)."""
        if not self._file_path:
            return
        full_path = os.path.join(self._base_dir, self._file_path)
        if not os.path.exists(full_path):
            return
        # Clear cached GL buffers and geometry
        self.points = None
        self.colors = None
        self.normals = None
        self.triangles = None
        self.line_points = None
        self.point_count = 0
        self.tri_count = 0
        self._pts_bytes = None
        self._normals_bytes = None
        self._tris_bytes = None
        self._lines_bytes = None
        self._rgba_bytes = None
        self._rgba_array = None
        self._rgba_opacity = None
        self.load_error = None
        try:
            from locul3d.utils.io import load_geometry
            load_geometry(full_path, self)
            self._file_mtime = os.path.getmtime(full_path)
        except Exception as e:
            self.load_error = str(e)

    def _load_wireframe(self):
        """Build line segments for OBB wireframe from box_points."""
        from .constants import OBB_EDGES
        
        box_pts = self.meta.get("box_points")
        if not box_pts or len(box_pts) != 8:
            self.load_error = "Invalid box_points for wireframe"
            return
        corners = np.array(box_pts, dtype=np.float32)
        lines = []
        for i, j in OBB_EDGES:
            lines.append(corners[i])
            lines.append(corners[j])
        self.line_points = np.array(lines, dtype=np.float32)
        self.point_count = 8

    # --- Cached byte buffers ---

    def evict_byte_caches(self):
        self._pts_bytes = None
        self._normals_bytes = None
        self._tris_bytes = None
        self._lines_bytes = None
        self._rgba_bytes = None
        self._rgba_array = None
        self._rgba_opacity = None

    def release_source_data(self):
        """Release all numpy source data to free memory."""
        self.points = None
        self.colors = None
        self.normals = None
        self.triangles = None
        self.line_points = None
        self.evict_byte_caches()

    def get_pts_bytes(self) -> Optional[bytes]:
        if self._pts_bytes is None and self.points is not None:
            self._pts_bytes = self.points.tobytes()
        return self._pts_bytes

    def get_pts_array(self) -> Optional[np.ndarray]:
        """Return contiguous float32 points array for direct GL use."""
        if self.points is None:
            return None
        if self.points.dtype != np.float32 or not self.points.flags['C_CONTIGUOUS']:
            self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        return self.points

    def get_normals_bytes(self) -> Optional[bytes]:
        if self._normals_bytes is None and self.normals is not None:
            self._normals_bytes = self.normals.tobytes()
        return self._normals_bytes

    def get_normals_array(self) -> Optional[np.ndarray]:
        """Return contiguous float32 normals array for direct GL use."""
        if self.normals is None:
            return None
        if self.normals.dtype != np.float32 or not self.normals.flags['C_CONTIGUOUS']:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32)
        return self.normals

    def get_tris_bytes(self) -> Optional[bytes]:
        if self._tris_bytes is None and self.triangles is not None:
            self._tris_bytes = self.triangles.tobytes()
        return self._tris_bytes

    def get_tris_array(self) -> Optional[np.ndarray]:
        """Return contiguous uint32 triangles array for direct GL use."""
        if self.triangles is None:
            return None
        if self.triangles.dtype != np.uint32 or not self.triangles.flags['C_CONTIGUOUS']:
            self.triangles = np.ascontiguousarray(self.triangles, dtype=np.uint32)
        return self.triangles

    def get_lines_bytes(self) -> Optional[bytes]:
        if self._lines_bytes is None and self.line_points is not None:
            self._lines_bytes = self.line_points.tobytes()
        return self._lines_bytes

    def get_colors_array(self) -> Optional[np.ndarray]:
        """Return contiguous float32 colors array for direct GL use."""
        if self.colors is None:
            return None
        if self.colors.dtype != np.float32 or not self.colors.flags['C_CONTIGUOUS']:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        return self.colors

    def get_lines_array(self) -> Optional[np.ndarray]:
        """Return contiguous float32 line_points array for direct GL use."""
        if self.line_points is None:
            return None
        if self.line_points.dtype != np.float32 or not self.line_points.flags['C_CONTIGUOUS']:
            self.line_points = np.ascontiguousarray(self.line_points, dtype=np.float32)
        return self.line_points

    def get_rgba_bytes(self) -> Optional[bytes]:
        """Get RGBA color bytes with current opacity baked into alpha channel."""
        if self.colors is None:
            return None
        if self._rgba_bytes is None or self._rgba_opacity != self.opacity:
            rgba = np.empty((len(self.colors), 4), dtype=np.float32)
            rgba[:, :3] = self.colors
            rgba[:, 3] = self.opacity
            self._rgba_bytes = rgba.tobytes()
            self._rgba_opacity = self.opacity
        return self._rgba_bytes

    def get_rgba_array(self) -> Optional[np.ndarray]:
        """Get RGBA float32 array with opacity in alpha. No bytes copy."""
        if self.colors is None:
            return None
        if self._rgba_array is None or self._rgba_opacity != self.opacity:
            rgba = np.empty((len(self.colors), 4), dtype=np.float32)
            rgba[:, :3] = self.colors
            rgba[:, 3] = self.opacity
            self._rgba_array = rgba
            self._rgba_opacity = self.opacity
        return self._rgba_array

    # --- Bounds ---

    def get_bounds(self):
        """Return (center, radius) bounding sphere for this layer."""
        pts = self.points
        if pts is None or len(pts) == 0:
            pts = self.line_points
        if pts is None or len(pts) == 0:
            return np.zeros(3), 1.0
        center = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - center, axis=1).max())
        return center, max(radius, 0.1)


class LayerManager:
    """Manages loading and access to all layers."""

    def __init__(self):
        self.layers: list[LayerData] = []
        self.base_dir: str = ""
        self._scene_aabb = None   # cached (x0, x1, y0, y1, z0, z1) or None
        self._ceiling_z = None    # cached ceiling height or None
        self._ceiling_computed = False  # True once detection has run

    @property
    def scene_aabb(self):
        """Cached scene AABB across all geometry layers (excludes panoramas).

        Computed once on first access; call invalidate_scene_aabb() when
        layers are added or removed.
        """
        if self._scene_aabb is None:
            self._scene_aabb = self._compute_scene_aabb()
        return self._scene_aabb

    @property
    def ceiling_z(self):
        """Cached ceiling height, or None if not detected / not yet computed."""
        return self._ceiling_z

    def compute_ceiling_background(self):
        """Compute ceiling height silently. Call after all geometry is loaded.

        Result is cached in ceiling_z — not applied until user asks.
        """
        if self._ceiling_computed:
            return
        self._ceiling_computed = True
        from ..analysis.ceiling import CeilingDetector
        det = CeilingDetector()
        geom_layers = [l for l in self.layers if l.layer_type != "panorama"]
        self._ceiling_z = det.detect(geom_layers, max_samples=500_000)

    def invalidate_scene_aabb(self):
        """Force re-computation of scene AABB and ceiling on next access."""
        self._scene_aabb = None
        self._ceiling_z = None
        self._ceiling_computed = False

    def _compute_scene_aabb(self):
        """Compute union AABB from all geometry layers (panoramas excluded)."""
        global_min = None
        global_max = None
        for layer in self.layers:
            if layer.layer_type == "panorama":
                continue
            if layer.points is not None and len(layer.points) > 0:
                lmin = layer.points.min(axis=0)
                lmax = layer.points.max(axis=0)
                if global_min is None:
                    global_min = lmin.copy()
                    global_max = lmax.copy()
                else:
                    np.minimum(global_min, lmin, out=global_min)
                    np.maximum(global_max, lmax, out=global_max)
        if global_min is None:
            return None
        return (
            float(global_min[0]), float(global_max[0]),
            float(global_min[1]), float(global_max[1]),
            float(global_min[2]), float(global_max[2]),
        )

    def load_single_file(self, path: str):
        """Load a single PLY/OBJ file as one layer (appends to existing layers)."""
        from pathlib import Path
        from ..plugins.importers.loaders import load_geometry
        
        ext = Path(path).suffix.lower()
        layer_type = "mesh" if ext == ".obj" else "pointcloud"

        # For PLY, probe whether it contains triangles
        if ext == ".ply":
            try:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(path)
                if len(mesh.triangles) > 0:
                    layer_type = "mesh"
            except Exception:
                pass

        from .constants import AUTO_LAYER_COLORS
        color_idx = len(self.layers) % len(AUTO_LAYER_COLORS)
        auto_color = AUTO_LAYER_COLORS[color_idx] + [1.0]  # add alpha
        
        layer_def = {
            "id": f"file_{len(self.layers)}",
            "name": Path(path).name,
            "type": layer_type,
            "file": Path(path).name,
            "visible": True,
            "opacity": 1.0,
            "color": auto_color,
        }
        self.base_dir = str(Path(path).parent)
        layer = LayerData(layer_def, self.base_dir)
        load_geometry(path, layer)
        layer.loaded = True  # Mark as loaded so visible_layers() includes it
        # If file has per-vertex colors, drop the auto layer color so the
        # renderer falls through to vertex-color path.  Exception: wireframe
        # layers — extract the representative frame color from the per-line
        # colors so the swatch shows the actual wireframe color from the PLY.
        if layer.colors is not None and len(layer.colors) > 0:
            if layer.layer_type == "wireframe":
                # Use the median per-line color as the swatch color
                median_rgb = np.median(layer.colors[:, :3], axis=0)
                layer.color = median_rgb.tolist() + [1.0]
            else:
                layer.color = None
        self.layers.append(layer)
        self.invalidate_scene_aabb()

    def get_scene_bounds(self):
        """Compute union bounding sphere across all loaded layers."""
        global_min = None
        global_max = None
        for layer in self.layers:
            for pts in (layer.points, layer.line_points):
                if pts is not None and len(pts) > 0:
                    lmin = pts.min(axis=0)
                    lmax = pts.max(axis=0)
                    if global_min is None:
                        global_min = lmin
                        global_max = lmax
                    else:
                        np.minimum(global_min, lmin, out=global_min)
                        np.maximum(global_max, lmax, out=global_max)
        if global_min is None:
            return np.zeros(3), 10.0
        center = (global_min + global_max) / 2.0
        radius = float(np.linalg.norm(global_max - global_min) / 2.0)
        return center, max(radius, 1.0)

    def visible_layers(self) -> list[LayerData]:
        return [l for l in self.layers if l.visible and l.loaded and not l.load_error]

    def set_all_visible(self, visible: bool):
        for layer in self.layers:
            layer.visible = visible

    def solo_layer(self, layer_id: str):
        for layer in self.layers:
            layer.visible = (layer.id == layer_id)

    def total_stats(self) -> tuple[int, int, int]:
        """Return (total_layers, total_points, total_tris)."""
        pts = sum(l.point_count for l in self.layers)
        tris = sum(l.tri_count for l in self.layers)
        return len(self.layers), pts, tris
