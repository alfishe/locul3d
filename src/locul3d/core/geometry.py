"""Geometry data models for annotations."""

import enum
import math
import numpy as np
from typing import Optional


class AnnotationCategory(enum.Enum):
    """Pipeline annotation category for grouping bboxes and gaps."""
    RACK = "rack"
    EMPTY_SPACE = "empty_space"


class GapItem:
    """A measurement annotation between two points with bracket + ticks.

    Supports two orientations:
      - "top":   bracket above racks, ticks extend down to rack tops (rack gaps)
      - "front": bracket in front of rack row, ticks extend back to face (empty spaces)
    """

    def __init__(self, edge_a=None, edge_b=None, gap_mm=0.0, axis=0, visible=True,
                 anchor_a=None, anchor_b=None, tick_dir=None, color=None,
                 category: Optional[AnnotationCategory] = None,
                 label_t: float = 0.5):
        self.edge_a = np.array(edge_a if edge_a is not None else [0, 0, 0],
                               dtype=np.float64)
        self.edge_b = np.array(edge_b if edge_b is not None else [0, 0, 0],
                               dtype=np.float64)
        self.gap_mm = float(gap_mm)
        self.axis = int(axis)  # 0=X, 1=Y corridor axis
        self.visible = visible
        # Where ticks connect to the bbox face
        self.anchor_a = np.array(anchor_a if anchor_a is not None else self.edge_a,
                                 dtype=np.float64)
        self.anchor_b = np.array(anchor_b if anchor_b is not None else self.edge_b,
                                 dtype=np.float64)
        # Direction to extend ticks past the bracket (away from bbox)
        self.tick_dir = np.array(tick_dir if tick_dir is not None else [0, 0, 0.03],
                                 dtype=np.float64)
        # Annotation color (RGB 0-1); None = use default
        self.color = tuple(color) if color is not None else None
        self.category = category
        self.label_t = label_t  # 0.0=edge_a, 0.5=midpoint, 1.0=edge_b


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
        self.save_format = "center"  # "center" or "corners" — per-item

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

    def to_dict(self, format=None):
        """Serialise bbox to dict.

        Args:
            format: 'center' or 'corners'. Defaults to this item's save_format.
        """
        fmt = format or self.save_format
        d = {"label": self.label}
        if fmt == "corners":
            d["min"] = [round(float(v), 4) for v in self.bb_min]
            d["max"] = [round(float(v), 4) for v in self.bb_max]
        else:
            d["center"] = [round(float(v), 4) for v in self.center_pos]
            d["size"] = [round(float(v), 4) for v in self.size]
        d["color"] = [round(float(v), 3) for v in self.color]
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
            item.save_format = "center"
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
            item.save_format = "corners"
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
