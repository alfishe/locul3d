"""Scene management utilities."""

from .layer import LayerManager


def compute_scene_bounds(layer_manager: LayerManager):
    """Compute scene bounding sphere from all layers."""
    return layer_manager.get_scene_bounds()
