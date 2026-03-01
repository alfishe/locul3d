"""
Camera Model — Intrinsics, Extrinsics, Projection
====================================================
Pinhole camera with distortion, COLMAP format parser,
3D→2D projection and visibility testing.

Ported from pitch POC (poc/photo-projection/camera.py).
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Radial distortion coefficients (k1, k2)
    k1: float = 0.0
    k2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0,      self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1.0],
        ], dtype=np.float64)

    @property
    def fov_x_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def fov_y_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_deg: float) -> "CameraIntrinsics":
        """Create intrinsics from field of view."""
        fx = width / (2 * math.tan(math.radians(fov_x_deg / 2)))
        fy = fx  # square pixels
        return cls(width=width, height=height, fx=fx, fy=fy,
                   cx=width / 2, cy=height / 2)


@dataclass
class CameraPose:
    """Camera extrinsics — position and orientation in world space."""
    name: str
    R: np.ndarray  # 3×3 rotation matrix (world-to-camera)
    t: np.ndarray  # 3×1 translation (world-to-camera)
    intrinsics: CameraIntrinsics
    image_path: Optional[str] = None

    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction in world space (viewing direction)."""
        return self.R.T @ np.array([0, 0, 1], dtype=np.float64)

    @property
    def view_matrix_4x4(self) -> np.ndarray:
        """4×4 view matrix (world-to-camera)."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.R
        m[:3, 3] = self.t.flatten()
        return m

    @property
    def projection_matrix_4x4(self) -> np.ndarray:
        """4×4 OpenGL projection matrix from intrinsics."""
        w, h = self.intrinsics.width, self.intrinsics.height
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        near, far = 0.1, 200.0
        m = np.zeros((4, 4), dtype=np.float64)
        m[0, 0] = 2 * fx / w
        m[1, 1] = 2 * fy / h
        m[0, 2] = 1 - 2 * cx / w
        m[1, 2] = 2 * cy / h - 1
        m[2, 2] = -(far + near) / (far - near)
        m[2, 3] = -2 * far * near / (far - near)
        m[3, 2] = -1
        return m


def project_points(points_3d: np.ndarray, camera: CameraPose
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) world coordinates
        camera: CameraPose with intrinsics

    Returns:
        pixels: (N, 2) pixel coordinates
        depths: (N,) distances along camera Z axis
        valid: (N,) boolean mask — True if point is in front and inside image
    """
    N = len(points_3d)
    # Transform to camera space
    pts_cam = (camera.R @ points_3d.T).T + camera.t.flatten()  # (N, 3)
    depths = pts_cam[:, 2]  # Z in camera coords

    # Project to normalized image coordinates
    z_safe = np.where(depths > 0.01, depths, 0.01)
    x_norm = pts_cam[:, 0] / z_safe
    y_norm = pts_cam[:, 1] / z_safe

    # Apply radial distortion
    k1, k2 = camera.intrinsics.k1, camera.intrinsics.k2
    if k1 != 0 or k2 != 0:
        r2 = x_norm**2 + y_norm**2
        distort = 1 + k1 * r2 + k2 * r2**2
        x_norm *= distort
        y_norm *= distort

    # Apply intrinsics
    K = camera.intrinsics.K
    px = K[0, 0] * x_norm + K[0, 2]
    py = K[1, 1] * y_norm + K[1, 2]

    pixels = np.column_stack([px, py])

    # Validity mask
    w, h = camera.intrinsics.width, camera.intrinsics.height
    valid = (
        (depths > 0.01) &
        (px >= 0) & (px < w) &
        (py >= 0) & (py < h)
    )

    return pixels, depths, valid


def sample_image_at_pixels(image: np.ndarray, pixels: np.ndarray,
                           valid: np.ndarray) -> np.ndarray:
    """
    Sample RGB colors from image at projected pixel coordinates.

    Args:
        image: (H, W, 3) uint8 or float32
        pixels: (N, 2) float pixel coords
        valid: (N,) boolean mask

    Returns:
        colors: (N, 3) float32 in [0, 1]
    """
    N = len(pixels)
    colors = np.zeros((N, 3), dtype=np.float32)

    if not np.any(valid):
        return colors

    h, w = image.shape[:2]
    px = np.clip(pixels[valid, 0].astype(int), 0, w - 1)
    py = np.clip(pixels[valid, 1].astype(int), 0, h - 1)

    sampled = image[py, px]
    if sampled.dtype == np.uint8:
        sampled = sampled.astype(np.float32) / 255.0
    colors[valid] = sampled[:, :3]

    return colors


def compute_visibility_scores(points_3d: np.ndarray, camera: CameraPose,
                              depths: np.ndarray, valid: np.ndarray
                              ) -> np.ndarray:
    """
    Score each point for how well this camera can see it.
    Higher = better view (closer, more perpendicular).

    Returns:
        scores: (N,) float, 0 for invalid points
    """
    N = len(points_3d)
    scores = np.zeros(N, dtype=np.float32)

    if not np.any(valid):
        return scores

    cam_pos = camera.position
    view_dirs = points_3d[valid] - cam_pos
    dists = np.linalg.norm(view_dirs, axis=1).clip(0.01)
    view_dirs /= dists[:, None]

    # Score: inverse distance, penalize grazing angles (dot with -forward)
    forward = camera.forward
    cos_angle = np.sum(view_dirs * forward, axis=1).clip(0, 1)

    # Combined score: close + facing camera = good
    scores[valid] = cos_angle / (1.0 + dists * 0.1)

    return scores


def frustum_corners(camera: CameraPose, near=0.5, far=20.0) -> np.ndarray:
    """
    Return 8 corners of the camera frustum in world coordinates.
    Order: [near_tl, near_tr, near_br, near_bl, far_tl, far_tr, far_br, far_bl]
    """
    w, h = camera.intrinsics.width, camera.intrinsics.height
    fx, fy = camera.intrinsics.fx, camera.intrinsics.fy
    cx, cy = camera.intrinsics.cx, camera.intrinsics.cy

    corners_2d = np.array([
        [0, 0], [w, 0], [w, h], [0, h]  # tl, tr, br, bl
    ], dtype=np.float64)

    # Unproject to camera space at z=1
    dirs = np.zeros((4, 3), dtype=np.float64)
    dirs[:, 0] = (corners_2d[:, 0] - cx) / fx
    dirs[:, 1] = (corners_2d[:, 1] - cy) / fy
    dirs[:, 2] = 1.0

    corners = np.zeros((8, 3), dtype=np.float64)
    corners[:4] = dirs * near
    corners[4:] = dirs * far

    # Transform to world space
    R_inv = camera.R.T
    t_inv = camera.position
    for i in range(8):
        corners[i] = R_inv @ corners[i] + t_inv

    return corners.astype(np.float32)


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[0, 1] = 2*(qx*qy - qz*qw)
    R[0, 2] = 2*(qx*qz + qy*qw)
    R[1, 0] = 2*(qx*qy + qz*qw)
    R[1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[1, 2] = 2*(qy*qz - qx*qw)
    R[2, 0] = 2*(qx*qz - qy*qw)
    R[2, 1] = 2*(qy*qz + qx*qw)
    R[2, 2] = 1 - 2*(qx*qx + qy*qy)
    return R
