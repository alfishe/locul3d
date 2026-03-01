"""OBJ file importer plugin."""

from pathlib import Path
from typing import Optional

from ..base import ImporterPlugin
from ...core.layer import LayerData


class OBJImporter(ImporterPlugin):
    """OBJ file importer using Open3D."""

    @property
    def name(self) -> str:
        return "OBJ Importer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def file_extensions(self) -> list[str]:
        return ['.obj']

    def can_import(self, file_path: str) -> bool:
        """Check if file is an OBJ file."""
        return Path(file_path).suffix.lower() == '.obj'

    def import_file(self, file_path: str) -> Optional[LayerData]:
        """Import OBJ file and return LayerData object."""
        try:
            import open3d as o3d
        except ImportError:
            return None

        # OBJs are typically meshes
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        layer_def = {
            "id": f"obj_{Path(file_path).stem}",
            "name": Path(file_path).name,
            "type": "mesh",
            "visible": True,
            "opacity": 1.0,
            "color": None,  # Will use per-vertex colors if available
            "file": Path(file_path).name,
        }

        # Create layer with base_dir pointing to file's directory
        layer = LayerData(layer_def, str(Path(file_path).parent))
        layer.load()
        return layer
