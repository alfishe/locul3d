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

    def detect(self, layers, max_samples: int = 0) -> Optional[float]:
        """Detect ceiling height from a list of LayerData objects.

        Reads Z values directly from layer point arrays — no copies.
        If max_samples > 0, subsamples each layer proportionally.

        Returns:
            Ceiling Z height, or None if detection fails.
        """
        # First pass: find Z range across all geometry layers
        z_min_global = float('inf')
        z_max_global = float('-inf')
        total_pts = 0

        for layer in layers:
            if layer.points is None or len(layer.points) == 0:
                continue
            z_col = layer.points[:, 2]
            z_min_global = min(z_min_global, float(z_col.min()))
            z_max_global = max(z_max_global, float(z_col.max()))
            total_pts += len(z_col)

        if total_pts < 100 or z_max_global - z_min_global < 1.0:
            return None

        # Build histogram incrementally (no concatenation)
        n_bins = int((z_max_global - z_min_global) / self.bin_size) + 1
        counts = np.zeros(n_bins, dtype=np.int64)

        # Subsampling stride
        stride = max(1, total_pts // max_samples) if max_samples > 0 else 1

        for layer in layers:
            if layer.points is None or len(layer.points) == 0:
                continue
            z_col = layer.points[::stride, 2]
            # Bin indices for this layer's Z values
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

        # Require clear ceiling surface (3× median density)
        nonzero = counts[counts > 0]
        median_density = float(np.median(nonzero)) if len(nonzero) > 0 else 0
        if peak_density < median_density * 3:
            return None

        return float(upper_centers[peak_idx])
