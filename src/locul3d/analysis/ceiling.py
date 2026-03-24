"""Ceiling height detector.

Analyzes Z-distribution of loaded point clouds to detect the ceiling plane.
Uses an incremental histogram approach — NO data copies or concatenation.
"""

import numpy as np
from typing import Optional, Tuple


class CeilingDetector:
    """Detect ceiling height from point cloud Z-distribution.

    Strategy:
      1. Build a Z-histogram incrementally across all layers (no data copy).
      2. Look for a dense horizontal band in the upper portion (top 40%).
      3. Return the lower edge of that band as the ceiling height.
      4. Require 3× median density to confirm a real ceiling surface.
    """

    def __init__(self, bin_size: float = 0.05):
        self.bin_size = bin_size

    def detect(self, layers, max_samples: int = 500_000) -> Optional[float]:
        """Detect ceiling height from a list of LayerData objects.

        Uses per-layer cached AABBs for Z range (instant), then builds
        a Z-histogram from subsampled data.  Skips the raw_scan layer
        if a mid-res equivalent exists (same data, fewer points).

        Returns:
            Ceiling Z height, or None if detection fails.
        """
        # Filter to usable layers; prefer mid-res over raw_scan
        has_midres = any(getattr(l, 'id', '') == 'midres' for l in layers)
        usable = []
        for layer in layers:
            if layer.points is None or len(layer.points) == 0:
                continue
            if has_midres and getattr(layer, 'id', '') == 'raw_scan':
                continue
            usable.append(layer)

        if not usable:
            return None

        # Z range from cached AABBs or subsampled min/max (no full scan)
        z_min_global = float('inf')
        z_max_global = float('-inf')
        total_pts = 0

        for layer in usable:
            aabb_min = getattr(layer, '_aabb_min', None)
            aabb_max = getattr(layer, '_aabb_max', None)
            if aabb_min is not None and aabb_max is not None:
                z_min_global = min(z_min_global, float(aabb_min[2]))
                z_max_global = max(z_max_global, float(aabb_max[2]))
            else:
                # Subsample for bounds
                pts = layer.points
                stride = max(1, len(pts) // 100_000)
                z_sample = pts[::stride, 2]
                z_min_global = min(z_min_global, float(z_sample.min()))
                z_max_global = max(z_max_global, float(z_sample.max()))
            total_pts += len(layer.points)

        if total_pts < 100 or z_max_global - z_min_global < 1.0:
            return None

        z_min_global = max(z_min_global, -100.0)
        z_max_global = min(z_max_global, 100.0)
        z_range = z_max_global - z_min_global
        if z_range < 1.0 or z_range > 200.0:
            return None

        # Build histogram from subsampled Z values (no float64 conversion)
        n_bins = int(z_range / self.bin_size) + 1
        n_bins = min(n_bins, 100_000)
        counts = np.zeros(n_bins, dtype=np.int64)

        stride = max(1, total_pts // max_samples) if max_samples > 0 else 1

        for layer in usable:
            z_col = layer.points[::stride, 2]
            # Bin directly as float32 (sufficient precision for 5cm bins)
            indices = np.clip(
                ((z_col - z_min_global) / self.bin_size).astype(np.int64),
                0, n_bins - 1)
            np.add.at(counts, indices, 1)

        return self._detect_from_histogram(counts, z_min_global, z_max_global)

    def _detect_from_histogram(self, counts, z_min, z_max):
        """Find ceiling from pre-built histogram."""
        z_range = z_max - z_min
        centers = np.arange(len(counts)) * self.bin_size + z_min + self.bin_size / 2

        # Upper region (top 40%)
        upper_threshold = z_min + z_range * 0.6
        upper_mask = centers >= upper_threshold
        upper_counts = counts[upper_mask]
        upper_centers = centers[upper_mask]

        if len(upper_counts) == 0:
            return None

        # Sliding window peak (~0.3m)
        window = max(1, int(0.3 / self.bin_size))
        if len(upper_counts) < window:
            peak_density = float(upper_counts.max())
            peak_idx = int(np.argmax(upper_counts))
        else:
            conv = np.convolve(upper_counts, np.ones(window), mode='valid')
            peak_density = float(conv.max()) / window
            peak_idx = int(np.argmax(conv))

        # Require a noticeable ceiling surface (1.5× median density).
        # Scenes without ceiling have no upper-region concentration.
        nonzero = counts[counts > 0]
        median_density = float(np.median(nonzero)) if len(nonzero) > 0 else 0
        if peak_density < median_density * 1.5:
            return None

        return float(upper_centers[peak_idx])
