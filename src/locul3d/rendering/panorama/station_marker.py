"""Panorama station marker — 3D diamond gizmo at camera positions.

Default appearance is controlled by the constants below.  Override them
here or (in a future release) via the application / project config.
"""

import math

try:
    from OpenGL.GL import (
        glBegin, glEnd, glVertex3f, glColor4f,
        GL_TRIANGLES,
    )
    HAS_GL = True
except ImportError:
    HAS_GL = False

# =====================================================================
# Default station-marker appearance
# Change these to adjust how panorama stations look in the 3D scene.
# These will later be driven by app / project configuration.
# =====================================================================

# Marker colour (R, G, B) — values 0.0–1.0
MARKER_COLOR = (1.0, 0.7, 0.0)  # orange

# Marker opacity — 0.0 (invisible) to 1.0 (fully opaque)
MARKER_OPACITY = 0.6

# Size as a fraction of scene radius (0.008 = 0.8 % of scene extent)
MARKER_SIZE_FRACTION = 0.008

# Fallback absolute size when scene radius is unknown or zero
MARKER_SIZE_FALLBACK = 0.15

# Vertical stretch factor — how much taller than wide (>1 = taller)
MARKER_VERTICAL_STRETCH = 1.8

# Horizontal squeeze factor — how much thinner than size (<1 = thinner)
MARKER_HORIZONTAL_SQUEEZE = 0.5

# =====================================================================


def draw_station_marker(position, scene_radius: float = 0.0,
                        color=None, opacity: float = None,
                        size: float = None):
    """Draw a diamond-shaped 3D marker at the given position.

    The marker is a tall, thin double-cone (elongated octahedron)
    made of 8 triangles.  All appearance parameters fall back to
    the module-level defaults above when not explicitly supplied.

    Parameters
    ----------
    position : array-like, length 3
        XYZ world coordinates.
    scene_radius : float
        Current scene bounding radius, used to scale the marker.
    color : tuple of 3 floats, optional
        RGB colour override (0–1).  Falls back to ``MARKER_COLOR``.
    opacity : float, optional
        Alpha override (0–1).  Falls back to ``MARKER_OPACITY``.
    size : float, optional
        Absolute size override in world units.  When *None* the size
        is computed from *scene_radius* × ``MARKER_SIZE_FRACTION``.
    """
    if not HAS_GL:
        return

    # Resolve defaults
    color = color or MARKER_COLOR
    opacity = opacity if opacity is not None else MARKER_OPACITY
    if size is None:
        size = (scene_radius * MARKER_SIZE_FRACTION
                if scene_radius > 0 else MARKER_SIZE_FALLBACK)

    x, y, z = float(position[0]), float(position[1]), float(position[2])
    r, g, b = color[0], color[1], color[2]

    # Tall, thin diamond: vertical stretch, horizontal squeeze
    vz = size * MARKER_VERTICAL_STRETCH
    hr = size * MARKER_HORIZONTAL_SQUEEZE

    top = (x, y, z + vz)
    bot = (x, y, z - vz)
    ring = [
        (x + hr, y, z),
        (x, y + hr, z),
        (x - hr, y, z),
        (x, y - hr, z),
    ]

    glColor4f(r, g, b, opacity)
    glBegin(GL_TRIANGLES)
    for i in range(4):
        p1 = ring[i]
        p2 = ring[(i + 1) % 4]
        # Upper cone
        glVertex3f(*top)
        glVertex3f(*p1)
        glVertex3f(*p2)
        # Lower cone
        glVertex3f(*bot)
        glVertex3f(*p2)
        glVertex3f(*p1)
    glEnd()

