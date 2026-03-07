"""Geometry loaders — delegates to the canonical loader in utils/io.py.

This module exists for backward compatibility. All loading logic lives
in locul3d.utils.io.load_geometry.
"""

from ...utils.io import load_geometry  # noqa: F401 — re-export

__all__ = ["load_geometry"]
