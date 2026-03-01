"""3D math utilities."""

import numpy as np


def project_point_to_camera_plane(screen_x, screen_y, center_point, 
                                   modelview, projection, viewport):
    """Project mouse to a camera-facing plane through center_point."""
    try:
        from OpenGL.GLU import gluUnProject
    except ImportError:
        return None
    
    origin, direction = ray_from_mouse(screen_x, screen_y, modelview, projection, viewport)
    
    # Camera look direction (from camera toward target)
    normal = direction
    
    denom = np.dot(direction, normal)
    if abs(denom) < 1e-12:
        return None
    t = np.dot(center_point - origin, normal) / denom
    if t < 0:
        return None
    return origin + direction * t


def project_point_to_plane(screen_x, screen_y, z_height, 
                          modelview, projection, viewport):
    """Project mouse to a plane at given Z height."""
    try:
        from OpenGL.GLU import gluUnProject
    except ImportError:
        return None
    
    origin, direction = ray_from_mouse(screen_x, screen_y, modelview, projection, viewport)
    
    if abs(direction[2]) < 1e-12:
        return None
    t = (z_height - origin[2]) / direction[2]
    if t < 0:
        return None
    return origin + direction * t


def ray_from_mouse(screen_x, screen_y, modelview, projection, viewport):
    """Generate ray from screen coordinates."""
    try:
        from OpenGL.GLU import gluUnProject
    except ImportError:
        return np.zeros(3), np.array([0, 0, 1])
    
    try:
        gl_y = float(viewport[3] - screen_y)
    except:
        gl_y = 0.0
    
    near = gluUnProject(float(screen_x), gl_y, 0.0, modelview, projection, viewport)
    far = gluUnProject(float(screen_x), gl_y, 1.0, modelview, projection, viewport)
    
    origin = np.array(near, dtype=np.float64)
    direction = np.array(far, dtype=np.float64) - origin
    length = np.linalg.norm(direction)
    if length > 0:
        direction /= length
    return origin, direction


def ray_aabb_intersect(origin, direction, bb_min, bb_max):
    """Ray-AABB intersection test. Returns t parameter or None."""
    t_min, t_max = -np.inf, np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if origin[i] < bb_min[i] or origin[i] > bb_max[i]:
                return None
        else:
            t1 = (bb_min[i] - origin[i]) / direction[i]
            t2 = (bb_max[i] - origin[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    if t_min > t_max or t_max < 0:
        return None
    return t_min if t_min >= 0 else t_max


def project_to_screen(world_point, modelview, projection, viewport):
    """Project 3D world point to 2D screen coordinates (Qt convention)."""
    try:
        from OpenGL.GLU import gluProject
    except ImportError:
        return 0, 0
    
    sx, sy, sz = gluProject(float(world_point[0]), float(world_point[1]),
                            float(world_point[2]), modelview, projection, viewport)
    return sx, viewport[3] - sy
