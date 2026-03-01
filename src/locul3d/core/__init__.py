"""Core data structures and utilities."""

from .constants import *
from .geometry import *
from .layer import *
from .scene import *

__all__ = [
    "BBoxItem",
    "PlaneItem",
    "LayerData",
    "LayerManager",
    "COLORS",
    "DARK_COLORS",
    "LIGHT_COLORS",
    "AABB_EDGES",
    "DEFAULT_SIZES",
]
