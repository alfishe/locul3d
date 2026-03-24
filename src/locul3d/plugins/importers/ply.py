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
        """Import PLY file and return LayerData object.

        Uses the fast binary parser for point clouds (30× faster than O3D).
        Falls back to Open3D for meshes, ASCII PLY, etc.
        """
        # Default to pointcloud — load_geometry will detect mesh/wireframe
        # and update layer_type.  This avoids the expensive O3D probe read
        # that used to double the load time.
        layer_def = {
            "id": f"ply_{Path(file_path).stem}",
            "name": Path(file_path).name,
            "type": "pointcloud",
            "visible": True,
            "opacity": 1.0,
            "color": None,
            "file": Path(file_path).name,
        }

        layer = LayerData(layer_def, str(Path(file_path).parent))
        layer.load()
        return layer
