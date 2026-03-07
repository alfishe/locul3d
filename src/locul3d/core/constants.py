"""Constants and configuration."""

import numpy as np
from PySide6.QtGui import QColor
from collections import OrderedDict

# ============================================================================
# Edge Connectivity
# ============================================================================

# OBB edge connectivity for Open3D's get_box_points() ordering
OBB_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 6), (1, 4),
    (2, 5), (2, 4),
    (3, 5), (3, 6),
    (7, 4), (7, 5), (7, 6),
]

# AABB edge connectivity: 8 corners ordered as
# 0=(xmin,ymin,zmin) 1=(xmax,ymin,zmin) 2=(xmax,ymax,zmin) 3=(xmin,ymax,zmin)
# 4=(xmin,ymin,zmax) 5=(xmax,ymin,zmax) 6=(xmax,ymax,zmax) 7=(xmin,ymax,zmax)
AABB_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

# Layer group definitions: (group_name, classifier_function)
LAYER_GROUPS = OrderedDict([
    ("Scan", lambda l: l.id in ("aligned", "unclassified", "decimated", "raw_scan")),
    ("Panoramas", lambda l: l.layer_type == "panorama"),
    ("Surfaces", lambda l: l.id.startswith("surface_") or l.layer_type == "mesh"),
    ("Overlays", lambda l: l.layer_type == "wireframe"),
])

# ============================================================================
# Color Palettes
# ============================================================================

DARK_COLORS = {
    'bg':              '#161821',
    'card':            '#1e212c',
    'text':            '#e0e0e8',
    'text_secondary':  '#b0b0c8',
    'text_muted':      '#8888a0',
    'accent':          '#4e8cff',
    'border':          '#2a2d3a',
    'hover':           '#3a4060',
    'selected':        '#2a2d3a',
    'input_bg':        '#1a1d27',
    'button_bg':       '#2a2d3a',
    'button_border':   '#3a4060',
    'checkbox_bg':     '#1a1d27',
    'checkbox_border': '#3a4060',
    'slider_groove':   '#2a2d3a',
    'swatch_border':   '#323250',
    'gl_bg':           (0.08, 0.08, 0.12, 1.0),
}

LIGHT_COLORS = {
    'bg':              '#f0f1f5',
    'card':            '#ffffff',
    'text':            '#1a1a2e',
    'text_secondary':  '#4b5563',
    'text_muted':      '#6b7280',
    'accent':          '#2563eb',
    'border':          '#d1d5db',
    'hover':           '#e5e7eb',
    'selected':        '#dbeafe',
    'input_bg':        '#f9fafb',
    'button_bg':       '#e5e7eb',
    'button_border':   '#d1d5db',
    'checkbox_bg':     '#ffffff',
    'checkbox_border': '#d1d5db',
    'slider_groove':   '#d1d5db',
    'swatch_border':   '#c0c0c0',
    'gl_bg':           (0.94, 0.94, 0.96, 1.0),
}

# Active color dict — mutated at runtime by ThemeManager
COLORS = DARK_COLORS.copy()

# ============================================================================
# BBox Defaults
# ============================================================================

DEFAULT_SIZES = {
    "mts_column": np.array([0.80, 0.60, 2.50]),
    "mts_box":    np.array([0.80, 0.60, 0.40]),
    "rack":       np.array([0.60, 1.20, 2.20]),
    "custom":     np.array([1.00, 1.00, 1.00]),
}

BBOX_COLORS = [
    [1.0, 0.5, 0.0],   # orange
    [0.2, 0.8, 0.2],   # green
    [0.3, 0.6, 1.0],   # blue
    [1.0, 0.3, 0.3],   # red
    [0.8, 0.2, 0.8],   # purple
    [1.0, 0.9, 0.2],   # yellow
    [0.0, 0.8, 0.8],   # cyan
    [1.0, 0.6, 0.8],   # pink
]

PLANE_COLORS = [
    [0.5, 0.5, 0.8],   # blue-gray
    [0.8, 0.5, 0.5],   # red-gray
    [0.5, 0.8, 0.5],   # green-gray
    [0.8, 0.8, 0.3],   # yellow-gray
    [0.7, 0.4, 0.8],   # purple
    [0.3, 0.8, 0.8],   # cyan
]

# ============================================================================
# Tool Modes
# ============================================================================

TOOL_SELECT = "select"
TOOL_MOVE   = "move"
TOOL_ROTATE = "rotate"
TOOL_SCALE  = "scale"

# ============================================================================
# Gizmo Constants
# ============================================================================

# Gizmo hit-test threshold in pixels
GIZMO_HIT_PX = 20

# Axis colors for gizmo
AXIS_COLORS = {
    0: (1.0, 0.2, 0.2),  # X = red
    1: (0.2, 1.0, 0.2),  # Y = green
    2: (0.3, 0.3, 1.0),  # Z = blue
}
AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}

# Auto-assigned distinct colors for layers loaded without a manifest
AUTO_LAYER_COLORS = [
    [0.20, 0.60, 1.00],  # blue
    [1.00, 0.40, 0.20],  # orange
    [0.30, 0.85, 0.40],  # green
    [0.90, 0.25, 0.55],  # pink
    [0.65, 0.45, 1.00],  # purple
    [0.00, 0.80, 0.75],  # teal
    [1.00, 0.75, 0.10],  # gold
    [0.55, 0.85, 0.20],  # lime
    [1.00, 0.50, 0.60],  # salmon
    [0.40, 0.50, 0.90],  # indigo
    [0.85, 0.55, 0.10],  # amber
    [0.10, 0.70, 0.55],  # jade
]


def _detect_system_scheme() -> str:
    """Return 'dark' or 'light' based on the OS color-scheme hint."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        app = QApplication.instance()
        style_hints = getattr(app, 'styleHints', None)
        if style_hints is not None:
            hints = style_hints()
            if hints and hints.colorScheme() == Qt.ColorScheme.Dark:
                return 'dark'
    except Exception:
        pass
    return 'light'


def _axis_qcolor(axis):
    """Return CSS color string for axis index."""
    r, g, b = AXIS_COLORS[axis]
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
