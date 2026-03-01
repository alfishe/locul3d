"""Camera controller for 3D viewport."""

import math
import numpy as np


class CameraController:
    """Manages camera state and transformations."""

    def __init__(self, scene_center=None, scene_radius=10.0):
        self.distance = 50.0
        self.azimuth = 45.0
        self.elevation = 30.0
        self.target = scene_center if scene_center is not None else np.array([0.0, 0.0, 0.0])
        self.fov = 45.0

    def get_eye_position(self):
        """Compute camera eye position from orbital parameters."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        x = self.target[0] + self.distance * math.cos(el) * math.cos(az)
        y = self.target[1] + self.distance * math.cos(el) * math.sin(az)
        z = self.target[2] + self.distance * math.sin(el)
        return np.array([x, y, z])

    def look_at(self, target, distance=None):
        """Set camera to look at target."""
        self.target = target if isinstance(target, np.ndarray) else np.array(target)
        if distance is not None:
            self.distance = distance

    def orbit(self, dx, dy):
        """Orbit camera around target."""
        self.azimuth -= dx * 0.3
        self.elevation = max(-89, min(89, self.elevation + dy * 0.3))

    def pan(self, dx, dy, scale=None):
        """Pan camera in screen space."""
        if scale is None:
            scale = self.distance * 0.002
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        right = np.array([-math.sin(az), math.cos(az), 0.0])
        up = np.array([-math.cos(az) * math.sin(el),
                       -math.sin(az) * math.sin(el),
                       math.cos(el)])
        self.target -= right * dx * scale
        self.target += up * dy * scale

    def dolly(self, delta, scene_radius=10.0):
        """Dolly camera toward/away from scene."""
        step = scene_radius * 0.03 * delta
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        forward = np.array([
            -math.cos(el) * math.cos(az),
            -math.cos(el) * math.sin(az),
            -math.sin(el),
        ])
        self.target += forward * step
        self.distance = min(self.distance, scene_radius * 0.3)
        self.distance = max(0.01, self.distance)

    def set_preset(self, preset_name):
        """Set camera to a preset view."""
        presets = {
            "Top": (0, 89),
            "Front": (0, 0),
            "Right": (90, 0),
            "Isometric": (45, 35.264),
        }
        if preset_name in presets:
            self.azimuth, self.elevation = presets[preset_name]
