"""Scene correction: rotation and shift adjustments for axis alignment.

Stores Euler-angle rotations (degrees) and linear shifts (scene units)
for each coordinate axis.  Applied as GL transforms in the rendering
pipeline so the underlying point data is never modified.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


SIDECAR_NAMES = [
    "{stem}.correction.yaml",
    "{stem}.correction.yml",
    "correction.yaml",
    "correction.yml",
]


@dataclass
class SceneCorrection:
    """Rotation + shift corrections for scene axis alignment."""

    rotate_x: float = 0.0   # degrees
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    shift_x: float = 0.0    # scene units (metres)
    shift_y: float = 0.0
    shift_z: float = 0.0

    # ---- persistence ----

    @property
    def is_identity(self) -> bool:
        return (self.rotate_x == 0 and self.rotate_y == 0 and self.rotate_z == 0
                and self.shift_x == 0 and self.shift_y == 0 and self.shift_z == 0)

    def rotation_matrix(self) -> "np.ndarray":
        """Build the combined rotation matrix (Rx * Ry * Rz)."""
        import numpy as np
        R = np.eye(3)
        if self.rotate_x != 0:
            rad = np.radians(self.rotate_x)
            c, s = np.cos(rad), np.sin(rad)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            R = Rx @ R
        if self.rotate_y != 0:
            rad = np.radians(self.rotate_y)
            c, s = np.cos(rad), np.sin(rad)
            Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            R = Ry @ R
        if self.rotate_z != 0:
            rad = np.radians(self.rotate_z)
            c, s = np.cos(rad), np.sin(rad)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R = Rz @ R
        return R

    def bake_points(self, pts: "np.ndarray") -> "np.ndarray":
        """Apply correction to an Nx3 point array (in-place friendly).

        Order matches GL pipeline: rotate first, then shift.
        Returns the transformed array (float64).
        """
        import numpy as np
        out = np.asarray(pts, dtype=np.float64)
        R = self.rotation_matrix()
        if not np.allclose(R, np.eye(3)):
            out = (R @ out.T).T
        out[:, 0] += self.shift_x
        out[:, 1] += self.shift_y
        out[:, 2] += self.shift_z
        return out

    def transform_point(self, point) -> "np.ndarray":
        """Apply correction transforms to a 3D point (same order as GL pipeline).

        Order: rotate X, Y, Z first, then shift in world space.
        """
        import numpy as np
        p = np.array(point, dtype=np.float64).copy()
        # Rotate X
        if self.rotate_x != 0:
            rad = np.radians(self.rotate_x)
            c, s = np.cos(rad), np.sin(rad)
            y, z = p[1], p[2]
            p[1] = c * y - s * z
            p[2] = s * y + c * z
        # Rotate Y
        if self.rotate_y != 0:
            rad = np.radians(self.rotate_y)
            c, s = np.cos(rad), np.sin(rad)
            x, z = p[0], p[2]
            p[0] = c * x + s * z
            p[2] = -s * x + c * z
        # Rotate Z
        if self.rotate_z != 0:
            rad = np.radians(self.rotate_z)
            c, s = np.cos(rad), np.sin(rad)
            x, y = p[0], p[1]
            p[0] = c * x - s * y
            p[1] = s * x + c * y
        # Shift (in world space, after rotation)
        p[0] += self.shift_x
        p[1] += self.shift_y
        p[2] += self.shift_z
        return p

    def as_dict(self) -> dict:
        return {
            "correction": {
                "rotate_x": self.rotate_x,
                "rotate_y": self.rotate_y,
                "rotate_z": self.rotate_z,
                "shift_x": self.shift_x,
                "shift_y": self.shift_y,
                "shift_z": self.shift_z,
            }
        }

    def save_yaml(self, path: str) -> None:
        """Write correction values to a YAML file."""
        if not HAS_YAML:
            # Fallback: write manually (simple enough structure)
            with open(path, "w") as f:
                f.write("# Scene correction (degrees / scene units)\n")
                f.write("correction:\n")
                f.write(f"  rotate_x: {self.rotate_x}\n")
                f.write(f"  rotate_y: {self.rotate_y}\n")
                f.write(f"  rotate_z: {self.rotate_z}\n")
                f.write(f"  shift_x: {self.shift_x}\n")
                f.write(f"  shift_y: {self.shift_y}\n")
                f.write(f"  shift_z: {self.shift_z}\n")
            return
        with open(path, "w") as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str) -> "SceneCorrection":
        """Load correction values from a YAML file."""
        if not HAS_YAML:
            # Minimal fallback parser for our simple format
            return cls._parse_simple(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        c = data.get("correction", {})
        return cls(
            rotate_x=float(c.get("rotate_x", 0)),
            rotate_y=float(c.get("rotate_y", 0)),
            rotate_z=float(c.get("rotate_z", 0)),
            shift_x=float(c.get("shift_x", 0)),
            shift_y=float(c.get("shift_y", 0)),
            shift_z=float(c.get("shift_z", 0)),
        )

    @classmethod
    def _parse_simple(cls, path: str) -> "SceneCorrection":
        """Fallback YAML parser when PyYAML is not installed."""
        vals = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if key in ("rotate_x", "rotate_y", "rotate_z",
                               "shift_x", "shift_y", "shift_z"):
                        try:
                            vals[key] = float(value)
                        except ValueError:
                            pass
        return cls(**vals)

    # ---- sidecar auto-detection ----

    @classmethod
    def find_sidecar(cls, scene_path: str) -> Optional[str]:
        """Look for a correction YAML sidecar next to *scene_path*.

        Searches for (in order):
          <stem>.correction.yaml
          <stem>.correction.yml
          correction.yaml
          correction.yml
        Returns the first match, or None.
        """
        p = Path(scene_path)
        parent = p.parent
        stem = p.stem

        for pattern in SIDECAR_NAMES:
            candidate = parent / pattern.format(stem=stem)
            if candidate.exists():
                return str(candidate)
        return None
