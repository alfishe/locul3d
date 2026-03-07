"""Geometry data models for annotations."""

import math
import numpy as np
from typing import Optional


class BBoxItem:
    """One annotation bounding box with position, size, and Z-rotation."""

    def __init__(self, label="mts_column", center=None, size=None,
                 rotation_z=0.0, color=None, visible=True,
                 bb_min=None, bb_max=None, fill_opacity=0.0):
        self.label = label
        self.rotation_z = float(rotation_z)  # degrees around Z
        self.color = list(color) if color is not None else list([1.0, 0.5, 0.0])
        self.visible = visible
        self.fill_opacity = float(fill_opacity)  # 0=wireframe, >0=filled faces

        # Accept either center+size or min+max
        if bb_min is not None and bb_max is not None:
            mn = np.array(bb_min, dtype=np.float64)
            mx = np.array(bb_max, dtype=np.float64)
            self.center_pos = (mn + mx) / 2.0
            self.size = mx - mn
        else:
            self.center_pos = np.array(center if center is not None else [0, 0, 0],
                                       dtype=np.float64)
            self.size = np.array(size if size is not None else [1, 1, 1],
                                 dtype=np.float64)

    @property
    def bb_min(self):
        if self.rotation_z == 0.0:
            return self.center_pos - self.size / 2.0
        corners = self.corners()
        return corners.min(axis=0)

    @property
    def bb_max(self):
        if self.rotation_z == 0.0:
            return self.center_pos + self.size / 2.0
        corners = self.corners()
        return corners.max(axis=0)

    def corners(self):
        """Return 8 box corners, rotated around Z axis through center."""
        hs = self.size / 2.0
        # AABB corners relative to center
        local = np.array([
            [-hs[0], -hs[1], -hs[2]],
            [+hs[0], -hs[1], -hs[2]],
            [+hs[0], +hs[1], -hs[2]],
            [-hs[0], +hs[1], -hs[2]],
            [-hs[0], -hs[1], +hs[2]],
            [+hs[0], -hs[1], +hs[2]],
            [+hs[0], +hs[1], +hs[2]],
            [-hs[0], +hs[1], +hs[2]],
        ], dtype=np.float64)
        if self.rotation_z != 0.0:
            rad = math.radians(self.rotation_z)
            c, s = math.cos(rad), math.sin(rad)
            x, y = local[:, 0].copy(), local[:, 1].copy()
            local[:, 0] = c * x - s * y
            local[:, 1] = s * x + c * y
        return local + self.center_pos

    def to_dict(self):
        d = {
            "label": self.label,
            "center": [round(float(v), 4) for v in self.center_pos],
            "size": [round(float(v), 4) for v in self.size],
            "color": [round(float(v), 3) for v in self.color],
        }
        if self.rotation_z != 0.0:
            d["rotation_z"] = round(float(self.rotation_z), 2)
        if self.fill_opacity > 0.0:
            d["fill_opacity"] = round(float(self.fill_opacity), 2)
        return d

    @classmethod
    def from_dict(cls, d):
        # Support both center+size and min+max formats
        if "center" in d:
            item = cls(
                label=d.get("label", "custom"),
                center=d["center"],
                size=d.get("size", [1, 1, 1]),
                rotation_z=d.get("rotation_z", 0.0),
                color=d.get("color"),
            )
            item.fill_opacity = d.get("fill_opacity", 0.0)
            return item
        else:
            item = cls(
                label=d.get("label", "custom"),
                bb_min=d["min"],
                bb_max=d["max"],
                rotation_z=d.get("rotation_z", 0.0),
                color=d.get("color"),
            )
            item.fill_opacity = d.get("fill_opacity", 0.0)
            return item

    def __repr__(self):
        return f"BBoxItem(label={self.label!r}, center={self.center_pos.tolist()}, size={self.size.tolist()})"


class PlaneItem:
    """A semi-transparent reference plane aligned to an axis pair."""

    AXES = ('xy', 'xz', 'yz')

    def __init__(self, axis='xy', center=None, size=None,
                 color=None, opacity=0.3, visible=True, global_coords=False):
        self.axis = axis  # 'xy', 'xz', 'yz'
        self.center = np.array(center if center is not None else [0, 0, 0],
                               dtype=np.float64)
        self.size = np.array(size if size is not None else [10.0, 10.0],
                             dtype=np.float64)  # width, height in-plane
        self.color = list(color) if color is not None else [0.5, 0.5, 0.8]
        self.opacity = float(opacity)
        self.visible = visible
        self.global_coords = global_coords  # True = unaffected by scene correction

    def corners(self):
        """Return 4 corners of the plane quad in world space.

        Center is the origin corner; W extends along first axis,
        H extends along second axis.
        """
        w, h = self.size[0], self.size[1]
        c = self.center
        if self.axis == 'xy':
            return np.array([
                [c[0],     c[1],     c[2]],
                [c[0] + w, c[1],     c[2]],
                [c[0] + w, c[1] + h, c[2]],
                [c[0],     c[1] + h, c[2]],
            ])
        elif self.axis == 'xz':
            return np.array([
                [c[0],     c[1], c[2]],
                [c[0] + w, c[1], c[2]],
                [c[0] + w, c[1], c[2] + h],
                [c[0],     c[1], c[2] + h],
            ])
        else:  # yz
            return np.array([
                [c[0], c[1],     c[2]],
                [c[0], c[1] + w, c[2]],
                [c[0], c[1] + w, c[2] + h],
                [c[0], c[1] - w, c[2] + h],
            ])

    def to_dict(self):
        d = {
            'axis': self.axis,
            'center': [round(float(v), 4) for v in self.center],
            'size': [round(float(v), 4) for v in self.size],
            'color': [round(float(v), 3) for v in self.color],
            'opacity': round(self.opacity, 2),
        }
        if self.global_coords:
            d['global_coords'] = True
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            axis=d.get('axis', 'xy'),
            center=d.get('center', [0, 0, 0]),
            size=d.get('size', [10.0, 10.0]),
            color=d.get('color'),
            opacity=d.get('opacity', 0.3),
            global_coords=d.get('global_coords', False),
        )

    def __repr__(self):
        return f"PlaneItem(axis={self.axis!r}, center={self.center.tolist()})"
