"""PLY file importer plugin."""

from pathlib import Path
from typing import Optional

from ..base import ImporterPlugin
from ...core.layer import LayerData


class PLYImporter(ImporterPlugin):
    """PLY file importer using Open3D."""

    @property
    def name(self) -> str:
        return "PLY Importer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def file_extensions(self) -> list[str]:
        return ['.ply']

    def can_import(self, file_path: str) -> bool:
        """Check if file is a PLY file."""
        return Path(file_path).suffix.lower() == '.ply'

    def import_file(self, file_path: str) -> Optional[LayerData]:
        """Import PLY file and return LayerData object."""
        try:
            import open3d as o3d
        except ImportError:
            return None

        # Probe for mesh vs pointcloud
        mesh = o3d.io.read_triangle_mesh(file_path)
        is_mesh = len(mesh.triangles) > 0

        layer_def = {
            "id": f"ply_{Path(file_path).stem}",
            "name": Path(file_path).name,
            "type": "mesh" if is_mesh else "pointcloud",
            "visible": True,
            "opacity": 1.0,
            "color": None,  # Will use per-vertex colors if available
            "file": Path(file_path).name,
        }

        # Create layer with base_dir pointing to file's directory
        layer = LayerData(layer_def, str(Path(file_path).parent))
        layer.load()
        return layer
