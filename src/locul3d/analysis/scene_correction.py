"""Scene correction auto-detection with diagnostics.

Analyzes a raw point cloud to compute rotation and shift corrections that
align the scene to world axes: floor at Z=0, walls parallel to X and Y.

Algorithm Overview
──────────────────
The correction has two independent components:

**Floor alignment** (rotate_x, rotate_y, shift_z):
    SVD plane fit on the lowest-Z percentile of points. The plane normal
    determines the tilt correction; the plane offset becomes shift_z.

**Wall alignment** (rotate_z) — multi-step surface detection:

    Step 1 — DETECT large vertical surfaces (≥ min_surface_area m²):
        1a. Extract wall-band points (Z ∈ [band_min, band_max] above floor)
        1b. Divide XY into an adaptive grid of cells (0.3–2.0 m, adjusts
            to point density so each cell has ~40+ points)
        1c. For each cell with ≥ 20 points, estimate surface normal via SVD.
            Filter: planar (σ₂/σ₁ < 0.3) AND vertical (|nz| < 0.25)
        1d. Merge adjacent vertical cells via BFS flood-fill with
            8-connectivity. Merge condition: normal angle within ±15° of
            the component's seed angle (mod 90°). This prevents angle
            drift along long walls.
        1e. Compute surface area from bounding box (width × height).
            Keep surfaces ≥ min_surface_area.
        1f. REFINE each large surface at full resolution: collect ALL
            wall-band points inside the surface's bounding box (+ margin),
            filter to inliers within 10 cm of the initial plane, refit
            the normal via SVD on the full-resolution inlier set.

    Step 2 — CLASSIFY surfaces (parallel/perpendicular filter):
        2a. Build an area-weighted histogram of surface normal angles
            (mod 90°, which collapses parallel AND perpendicular walls
            to the same angle).
        2b. Find the peak → dominant wall direction.
        2c. Mark surfaces within ±angle_tolerance of the dominant
            direction as "qualifying". Non-qualifying surfaces (columns,
            equipment, angled objects) are excluded.

    Step 3 — OPTIMIZE rotation angle:
        3a. Sweep candidate angles ±10° around the dominant direction
            at 0.01° resolution.
        3b. For each candidate θ: compute weighted mean angular error
            of all qualifying surfaces vs θ.
        3c. Best θ (minimum error) → snap to nearest axis (0° or 90°)
            → rotate_z correction.

Diagnostics
───────────
All intermediate data (cells, surfaces, qualifying flags, histogram) is
returned in `CorrectionDiagnostics` for visualization in the viewport.
The debug overlay renders qualifying surfaces as green quads, non-qualifying
as orange, with normal arrows and correction direction indicators.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.correction import SceneCorrection


@dataclass
class DetectedSurface:
    """A merged wall surface from connected cells."""
    normal: np.ndarray          # average 3D unit normal
    centroid: np.ndarray        # 3D centroid (point-weighted)
    point_count: int            # total points across all cells
    cell_count: int             # number of grid cells
    area: float                 # estimated area (cell_count * cell_size²)
    angle_deg: float            # XY normal angle (mod 90°)
    bbox_min: np.ndarray        # axis-aligned bounding box min
    bbox_max: np.ndarray        # axis-aligned bounding box max
    qualifying: bool = False    # passed the parallel/perpendicular filter


@dataclass
class CorrectionDiagnostics:
    """Diagnostic data from auto-detection for visualization."""
    # Floor
    floor_points: Optional[np.ndarray] = None
    floor_normal: Optional[np.ndarray] = None
    floor_centroid: Optional[np.ndarray] = None

    # Wall surfaces (after floor correction)
    wall_band_points: Optional[np.ndarray] = None
    surfaces: list[DetectedSurface] = field(default_factory=list)
    # Keep old name for viewport compatibility
    @property
    def wall_planes(self):
        return self.surfaces

    # Angle result
    wall_correction_deg: float = 0.0
    peak_angle_deg: float = 0.0

    # Stats
    total_points: int = 0
    floor_point_count: int = 0
    wall_band_point_count: int = 0
    wall_cells_total: int = 0
    wall_cells_vertical: int = 0
    surfaces_total: int = 0
    surfaces_large: int = 0
    surfaces_qualifying: int = 0


def auto_detect_correction(
    points: np.ndarray,
    floor_percentile: float = 5.0,
    wall_band_min: float = 0.5,
    wall_band_max: float = 2.0,
    min_surface_area: float = 5.0,
    angle_tolerance: float = 5.0,
    center: bool = False,
) -> tuple[SceneCorrection, CorrectionDiagnostics]:
    """Compute scene correction from a raw point cloud.

    Performs floor detection (tilt + vertical shift) followed by
    multi-step wall surface detection (Z rotation).

    Args:
        points:           (N, 3) float64 array of XYZ coordinates.
        floor_percentile: Percentile of Z values for floor candidate
                          selection (default 5 = bottom 5%).
        wall_band_min:    Min height above detected floor for wall
                          sampling band (meters, default 0.5).
        wall_band_max:    Max height above detected floor for wall
                          sampling band (meters, default 2.0).
        min_surface_area: Minimum surface area in m² to qualify as a
                          wall surface (default 5.0).
        angle_tolerance:  Max angular deviation from the dominant wall
                          direction to qualify a surface (degrees,
                          default 5.0).
        center:           If True, compute shift_x/shift_y to center
                          the scene at the origin.

    Returns:
        Tuple of (SceneCorrection, CorrectionDiagnostics).
        The correction contains rotate_x/y/z and shift_x/y/z.
        The diagnostics contain all intermediate data for visualization.
    """
    diag = CorrectionDiagnostics(total_points=len(points))
    result = SceneCorrection()

    # Deterministic subsample: use a coordinate-based spatial hash so
    # the *same* subset is selected regardless of input array ordering
    # (Open3D voxel_down_sample can reorder points across process runs).
    # This is O(N) — far cheaper than lexsort on millions of points.
    _MAX_DETECT = 2_000_000
    if len(points) > _MAX_DETECT:
        # Hash each point by its quantised coordinates (1 mm grid)
        q = (points * 1000.0).astype(np.int64)
        h = q[:, 0] * np.int64(1000003) + q[:, 1] * np.int64(1000033) + q[:, 2]
        del q
        stride = max(len(points) // _MAX_DETECT, 2)
        mask = (h % stride) == 0
        del h
        points = points[mask]
        del mask

    print(f"  ── Auto-detect ({len(points):,} points) ──")

    # ── Step 1: Floor detection ──────────────────────────────
    floor_normal, floor_d, floor_pts = _detect_floor_plane(points, floor_percentile)
    result.shift_z = floor_d
    result.rotate_x, result.rotate_y = _floor_rotation_angles(floor_normal)

    diag.floor_points = floor_pts
    diag.floor_normal = floor_normal
    diag.floor_centroid = floor_pts.mean(axis=0) if len(floor_pts) > 0 else np.zeros(3)
    diag.floor_point_count = len(floor_pts)

    print(f"  Step 1 — Floor: {len(floor_pts):,} pts (bottom {floor_percentile}%)")
    print(f"    normal=[{floor_normal[0]:.4f}, {floor_normal[1]:.4f}, {floor_normal[2]:.4f}]")
    print(f"    → rx={result.rotate_x:.4f}°, ry={result.rotate_y:.4f}°, sz={floor_d:.4f}")

    # ── Step 2: Apply floor correction, then detect walls ────
    corrected = _apply_rotation(points, result.rotate_x, result.rotate_y)
    corrected[:, 2] += result.shift_z

    print(f"  Step 2 — Walls: Z=[{wall_band_min:.1f}, {wall_band_max:.1f}]m, "
          f"min area={min_surface_area}m², tolerance=±{angle_tolerance}°")

    result.rotate_z, wall_diag = _detect_wall_angle_surfaces(
        corrected, wall_band_min, wall_band_max,
        min_surface_area, angle_tolerance)
    del corrected  # free large array

    diag.wall_band_points = wall_diag.get('band_points')
    diag.surfaces = wall_diag.get('surfaces', [])
    diag.wall_band_point_count = wall_diag.get('band_count', 0)
    diag.wall_cells_total = wall_diag.get('cells_total', 0)
    diag.wall_cells_vertical = wall_diag.get('cells_vertical', 0)
    diag.surfaces_total = wall_diag.get('surfaces_total', 0)
    diag.surfaces_large = wall_diag.get('surfaces_large', 0)
    diag.surfaces_qualifying = wall_diag.get('surfaces_qualifying', 0)
    diag.wall_correction_deg = result.rotate_z
    diag.peak_angle_deg = wall_diag.get('peak_angle', 0.0)

    # Log surfaces
    large = [s for s in diag.surfaces if s.area >= min_surface_area]
    qualifying = [s for s in diag.surfaces if s.qualifying]
    print(f"    Cells: {diag.wall_cells_vertical} vertical / "
          f"{diag.wall_cells_total} total")
    print(f"    Surfaces: {diag.surfaces_total} merged → "
          f"{diag.surfaces_large} large (≥{min_surface_area}m²) → "
          f"{diag.surfaces_qualifying} qualifying (±{angle_tolerance}°)")
    for j, s in enumerate(sorted(large, key=lambda s: s.area, reverse=True)):
        q = "✓" if s.qualifying else "✗"
        print(f"    [{q}] {s.area:.1f}m² ({s.cell_count} cells, "
              f"{s.point_count:,} pts), angle={s.angle_deg:.2f}° (mod 90°)")
    if diag.peak_angle_deg > 0.001 or diag.peak_angle_deg < -0.001:
        print(f"    Peak: {diag.peak_angle_deg:.2f}° (mod 90°)")
    print(f"    → rz={result.rotate_z:.4f}°")

    # ── Step 3: Optional centering ───────────────────────────
    if center:
        fully_rotated = _apply_z_rotation(corrected, result.rotate_z)
        centroid = fully_rotated.mean(axis=0)
        result.shift_x = -round(float(centroid[0]), 4)
        result.shift_y = -round(float(centroid[1]), 4)

    return result, diag


# ── Floor detection ──────────────────────────────────────────


def _detect_floor_plane(
    points: np.ndarray, percentile: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane to the lowest-Z percentile of points."""
    z_vals = points[:, 2]
    threshold = np.percentile(z_vals, percentile)
    floor_mask = z_vals <= threshold
    floor_pts = points[floor_mask]

    if len(floor_pts) < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, floor_pts

    centroid = floor_pts.mean(axis=0)
    centered = floor_pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]
    if normal[2] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    return normal, float(d), floor_pts


def _floor_rotation_angles(normal: np.ndarray) -> tuple[float, float]:
    """Compute rotate_x and rotate_y to align floor normal to +Z."""
    if abs(normal[2] - 1.0) < 1e-8:
        return 0.0, 0.0
    rotate_x = -math.degrees(math.atan2(normal[1], normal[2]))
    rotate_y = math.degrees(math.atan2(normal[0], normal[2]))
    return round(rotate_x, 4), round(rotate_y, 4)


# ── Wall detection: multi-step ───────────────────────────────


def _detect_wall_angle_surfaces(
    points: np.ndarray, band_min: float, band_max: float,
    min_surface_area: float, angle_tolerance: float,
    cell_size: float = 0.5, min_pts_per_cell: int = 20,
    verticality_thresh: float = 0.25,
) -> tuple[float, dict]:
    """Multi-step wall detection.

    Step 1: Voxelize → local normals → merge into surfaces → filter by area.
    Step 2: Classify parallel/perpendicular surfaces.
    Step 3: Optimize rotation angle via least mean error.
    """
    diag: dict = {'surfaces': [], 'band_points': None, 'band_count': 0,
                  'cells_total': 0, 'cells_vertical': 0,
                  'surfaces_total': 0, 'surfaces_large': 0,
                  'surfaces_qualifying': 0, 'peak_angle': 0.0}

    z = points[:, 2]
    wall_mask = (z >= band_min) & (z <= band_max)
    wall_pts = points[wall_mask]
    diag['band_count'] = len(wall_pts)

    if len(wall_pts) < 100:
        return 0.0, diag

    # Subsample for visualization
    max_vis = 50_000
    if len(wall_pts) > max_vis:
        rng = np.random.default_rng(seed=99)
        vis_idx = rng.choice(len(wall_pts), size=max_vis, replace=False)
        diag['band_points'] = wall_pts[vis_idx].copy()
    else:
        diag['band_points'] = wall_pts.copy()

    # ────────────────────────────────────────────────────────
    # STEP 1: Detect large vertical surfaces
    # ────────────────────────────────────────────────────────

    # 1a. Adaptive cell size: ensure enough points per cell
    #     Target ~50+ pts/cell.  cell_area = n_pts / (xy_area / cell²)
    x_min, y_min = wall_pts[:, 0].min(), wall_pts[:, 1].min()
    x_max, y_max = wall_pts[:, 0].max(), wall_pts[:, 1].max()
    xy_area = max((x_max - x_min) * (y_max - y_min), 1.0)
    density = len(wall_pts) / xy_area  # pts per m²
    # cell_size² * density ≈ target pts per cell
    adaptive_cell = math.sqrt(max(min_pts_per_cell * 2 / max(density, 1), 0.09))
    cell_size = max(0.3, min(adaptive_cell, 2.0))

    cx_idx = ((wall_pts[:, 0] - x_min) / cell_size).astype(np.int32)
    cy_idx = ((wall_pts[:, 1] - y_min) / cell_size).astype(np.int32)
    nx_cells = cx_idx.max() + 2

    cell_keys = cx_idx + cy_idx * nx_cells
    sort_idx = np.argsort(cell_keys)
    sorted_keys = cell_keys[sort_idx]
    sorted_pts = wall_pts[sort_idx]

    changes = np.where(np.diff(sorted_keys) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(sorted_keys)]])

    # Build cell data: key -> (normal, n_points, centroid, grid_x, grid_y)
    cells = {}
    for s, e in zip(starts, ends):
        n_pts = e - s
        if n_pts < min_pts_per_cell:
            continue
        diag['cells_total'] += 1

        cell_pts = sorted_pts[s:e]
        centroid = cell_pts.mean(axis=0)
        centered = cell_pts - centroid
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        normal = Vt[2]
        # Planarity check
        if S[1] > 1e-10 and S[2] / S[1] > 0.3:
            continue
        # Verticality check
        if abs(normal[2]) > verticality_thresh:
            continue
        if normal[0] < 0:
            normal = -normal

        diag['cells_vertical'] += 1
        key = int(sorted_keys[s])
        gx = key % nx_cells
        gy = key // nx_cells
        angle = math.degrees(math.atan2(normal[1], normal[0])) % 90.0
        raw_angle = math.degrees(math.atan2(normal[1], normal[0]))  # 0..180
        cells[key] = {
            'normal': normal, 'n_pts': n_pts, 'centroid': centroid,
            'gx': gx, 'gy': gy, 'angle': angle, 'raw_angle': raw_angle,
            'pts_min': cell_pts.min(axis=0), 'pts_max': cell_pts.max(axis=0),
        }

    # 1b. Merge adjacent cells with similar normals into surfaces
    #     Compare each neighbor against the SEED cell angle (not the
    #     immediate neighbor) to prevent angle drift along long walls.
    #     Use 8-connectivity (including diagonals) for better merging
    #     of cells along angled walls.
    visited = set()
    surfaces = []
    _neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for key, cell in cells.items():
        if key in visited:
            continue

        # BFS flood-fill — compare all to seed raw angle
        # Use RAW angles (not mod-90) so perpendicular walls
        # don't merge into one surface.
        seed_raw = cell['raw_angle']
        queue = deque([key])
        visited.add(key)
        component = [key]

        while queue:
            cur = queue.popleft()
            cur_cell = cells[cur]
            gx, gy = cur_cell['gx'], cur_cell['gy']

            for dx, dy in _neighbors:
                nkey = (gx + dx) + (gy + dy) * nx_cells
                if nkey in visited or nkey not in cells:
                    continue
                diff = abs(cells[nkey]['raw_angle'] - seed_raw)
                if diff > 15:  # merge tolerance vs seed
                    continue
                visited.add(nkey)
                queue.append(nkey)
                component.append(nkey)

        # Compute surface properties
        total_pts = sum(cells[k]['n_pts'] for k in component)
        n_cells = len(component)

        # Weighted average normal and centroid
        w_normal = np.zeros(3)
        w_centroid = np.zeros(3)
        bbox_min = np.full(3, np.inf)
        bbox_max = np.full(3, -np.inf)
        for k in component:
            c = cells[k]
            w = c['n_pts']
            w_normal += c['normal'] * w
            w_centroid += c['centroid'] * w
            bbox_min = np.minimum(bbox_min, c['pts_min'])
            bbox_max = np.maximum(bbox_max, c['pts_max'])

        w_normal /= np.linalg.norm(w_normal) + 1e-10
        w_centroid /= total_pts
        if w_normal[0] < 0:
            w_normal = -w_normal
        angle_deg = math.degrees(math.atan2(w_normal[1], w_normal[0])) % 90.0

        # Area from bounding box: width (XY extent along tangent) × height (Z extent)
        bbox_span = bbox_max - bbox_min
        width_xy = math.sqrt(bbox_span[0]**2 + bbox_span[1]**2)
        height_z = bbox_span[2]
        area = width_xy * height_z

        surf = DetectedSurface(
            normal=w_normal, centroid=w_centroid,
            point_count=total_pts, cell_count=n_cells,
            area=area, angle_deg=angle_deg,
            bbox_min=bbox_min, bbox_max=bbox_max,
        )
        surfaces.append(surf)

    diag['surfaces_total'] = len(surfaces)

    # 1c. Filter by area
    large_surfaces = [s for s in surfaces if s.area >= min_surface_area]
    diag['surfaces_large'] = len(large_surfaces)

    if not large_surfaces:
        diag['surfaces'] = surfaces
        return 0.0, diag

    # 1d. Refine normals using full-resolution point cloud
    #     For each large surface, collect ALL wall-band points inside
    #     the surface bounding box, filter to plane inliers, refit via SVD.
    margin = cell_size  # expand bbox by one cell on each side
    plane_thresh = 0.10  # 10cm inlier distance

    for surf in large_surfaces:
        bmin = surf.bbox_min - margin
        bmax = surf.bbox_max + margin

        # Select points inside expanded bounding box
        in_box = (
            (wall_pts[:, 0] >= bmin[0]) & (wall_pts[:, 0] <= bmax[0]) &
            (wall_pts[:, 1] >= bmin[1]) & (wall_pts[:, 1] <= bmax[1]) &
            (wall_pts[:, 2] >= bmin[2]) & (wall_pts[:, 2] <= bmax[2])
        )
        box_pts = wall_pts[in_box]

        if len(box_pts) < 50:
            continue

        # Filter to points near the initial plane
        d = -np.dot(surf.normal, surf.centroid)
        dists = np.abs(box_pts @ surf.normal + d)
        inlier_mask = dists < plane_thresh
        inlier_pts = box_pts[inlier_mask]

        if len(inlier_pts) < 50:
            continue

        # Refit normal from full-resolution inliers
        centroid = inlier_pts.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        refined = Vt[2]
        # Verticality sanity check
        if abs(refined[2]) > verticality_thresh:
            continue
        if refined[0] < 0:
            refined = -refined

        # Update surface with refined values
        surf.normal = refined
        surf.centroid = centroid
        surf.point_count = len(inlier_pts)
        surf.angle_deg = math.degrees(math.atan2(refined[1], refined[0])) % 90.0

    # ────────────────────────────────────────────────────────
    # STEP 2: Classify — find dominant direction, keep ±5°
    # ────────────────────────────────────────────────────────

    # Find dominant angle via area-weighted histogram (mod 90°)
    angles = np.array([s.angle_deg for s in large_surfaces])
    areas = np.array([s.area for s in large_surfaces])

    n_bins = 360
    hist, edges = np.histogram(angles, bins=n_bins, range=(0, 90), weights=areas)
    # Circular smoothing
    kernel = np.ones(5)
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smoothed = np.convolve(padded, kernel, mode='same')[2:-2]

    peak_bin = int(np.argmax(smoothed))
    # Sub-bin precision
    s0 = float(smoothed[(peak_bin - 1) % n_bins])
    s1 = float(smoothed[peak_bin])
    s2 = float(smoothed[(peak_bin + 1) % n_bins])
    denom = s0 - 2 * s1 + s2
    delta = 0.5 * (s0 - s2) / denom if abs(denom) > 1e-10 else 0.0
    dominant_angle = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0 + delta * (90.0 / n_bins)

    # Mark qualifying surfaces (within ±tolerance of dominant angle mod 90°)
    for s in surfaces:
        if s.area < min_surface_area:
            continue
        diff = abs(s.angle_deg - dominant_angle)
        if diff > 45:
            diff = 90 - diff
        if diff <= angle_tolerance:
            s.qualifying = True

    qualifying = [s for s in surfaces if s.qualifying]
    diag['surfaces_qualifying'] = len(qualifying)

    if not qualifying:
        diag['surfaces'] = surfaces
        return 0.0, diag

    # ────────────────────────────────────────────────────────
    # STEP 3: Optimize — find rotation minimizing mean error
    # ────────────────────────────────────────────────────────

    # For each candidate angle θ, compute weighted mean angular error:
    #   error(θ) = Σ area_i * min(|angle_i - θ| mod 90°, 90 - |angle_i - θ| mod 90°)
    # Search near the dominant angle for the exact optimum.

    q_angles = np.array([s.angle_deg for s in qualifying])
    q_weights = np.array([s.area for s in qualifying])

    # Coarse sweep: ±10° around dominant angle
    best_theta = dominant_angle
    best_error = np.inf
    for theta_offset in np.linspace(-10, 10, 2001):
        theta = dominant_angle + theta_offset
        diffs = np.abs(q_angles - theta)
        # Circular distance mod 90°
        diffs = np.minimum(diffs % 90, 90 - diffs % 90)
        error = np.sum(q_weights * diffs)
        if error < best_error:
            best_error = error
            best_theta = theta

    peak_angle = best_theta % 90.0
    diag['peak_angle'] = peak_angle

    # Snap to nearest axis
    if peak_angle > 45.0:
        correction = -(peak_angle - 90.0)
    else:
        correction = -peak_angle

    diag['surfaces'] = surfaces
    return float(round(correction, 4)), diag


# ── Rotation helpers ─────────────────────────────────────────


def _apply_rotation(points: np.ndarray, rx_deg: float, ry_deg: float) -> np.ndarray:
    rx, ry = math.radians(rx_deg), math.radians(ry_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    return ((Ry @ Rx) @ points.T).T.copy()


def _apply_z_rotation(points: np.ndarray, rz_deg: float) -> np.ndarray:
    rz = math.radians(rz_deg)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return (Rz @ points.T).T.copy()
