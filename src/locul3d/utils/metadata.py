"""Metadata handlers for pipeline object-cluster directories."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from ..core.geometry import AnnotationCategory, BBoxItem, GapItem

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class MetadataHandler(ABC):
    """Base class for auto-detecting and parsing *_metadata.yaml files."""

    file_pattern: str  # glob pattern, e.g. "rack_*_metadata.yaml"
    category: AnnotationCategory
    display_name: str  # shown in Annotations panel, e.g. "Racks"
    bbox_color: tuple  # RGB 0-1
    gap_color: tuple  # RGB 0-1
    neighbor_index_key: str  # key in neighbor dict, e.g. "rack_index"

    def detect(self, directory: Path) -> bool:
        """Return True if this handler's metadata files exist in directory."""
        return any(directory.glob(self.file_pattern))

    # Cross-axis offset from rack face for width annotations (meters)
    _WIDTH_OFFSET = 0.10

    def parse(
        self, directory: Path
    ) -> Tuple[List[BBoxItem], List[GapItem]]:
        """Parse metadata files → (bboxes, gaps)."""
        if not _HAS_YAML:
            return [], []

        # Load all metadata files
        items = {}
        for path in sorted(directory.glob(self.file_pattern)):
            with open(path) as f:
                data = yaml.safe_load(f)
            if data is None:
                continue
            items[data["index"]] = data

        # Create BBoxItems
        bboxes = []
        for idx in sorted(items):
            r = items[idx]
            bboxes.append(BBoxItem(
                label=self.category.value,
                center=r["center"],
                size=r["size"],
                color=list(self.bbox_color),
            ))

        # Deduplicate neighbor pairs
        pairs = {}  # (min_idx, max_idx) → gap_mm
        for idx, r in items.items():
            for side in ("neighbor_left", "neighbor_right"):
                nb = r.get(side)
                if nb is None:
                    continue
                other = nb[self.neighbor_index_key]
                if other not in items:
                    continue
                key = (min(idx, other), max(idx, other))
                if key not in pairs:
                    pairs[key] = nb["gap_mm"]

        # Detect corridor axis
        if pairs:
            axis = _detect_corridor_axis(items, pairs)
        else:
            axis = _detect_corridor_axis_from_spread(items)
        cross_axis = 1 - axis

        # Build neighbor gap annotations (above rack tops)
        gaps = []
        for (a_idx, b_idx), gap_mm in pairs.items():
            ra, rb = items[a_idx], items[b_idx]
            ac, asz = ra["center"], ra["size"]
            bc, bsz = rb["center"], rb["size"]

            if ac[axis] > bc[axis]:
                ac, asz, bc, bsz = bc, bsz, ac, asz

            a_right = ac[axis] + asz[axis] / 2
            b_left = bc[axis] - bsz[axis] / 2
            cross = (ac[cross_axis] + bc[cross_axis]) / 2
            a_top_z = ac[2] + asz[2] / 2
            b_top_z = bc[2] + bsz[2] / 2
            arrow_z = max(a_top_z, b_top_z) + 0.05

            if axis == 1:
                edge_a = [cross, a_right, arrow_z]
                edge_b = [cross, b_left, arrow_z]
                anchor_a = [cross, a_right, a_top_z]
                anchor_b = [cross, b_left, b_top_z]
            else:
                edge_a = [a_right, cross, arrow_z]
                edge_b = [b_left, cross, arrow_z]
                anchor_a = [a_right, cross, a_top_z]
                anchor_b = [b_left, cross, b_top_z]

            gaps.append(GapItem(
                edge_a, edge_b, gap_mm, axis, True,
                anchor_a=anchor_a, anchor_b=anchor_b,
                tick_dir=[0, 0, 0.03],
                color=self.gap_color,
                category=self.category,
            ))

        # Build width annotations at Z=0, offset into corridor
        # Cluster racks by cross-axis position to detect row sides
        cross_positions = sorted(
            [(r["center"][cross_axis], idx) for idx, r in items.items()])
        rows = _cluster_cross_rows(cross_positions)

        # For each row, determine inward direction (toward nearest other row)
        row_means = [sum(x for x, _ in row) / len(row) for row in rows]
        row_directions = {}  # idx → cross_sign
        for ri, row in enumerate(rows):
            mean = row_means[ri]
            others = [m for i, m in enumerate(row_means) if i != ri]
            if others:
                nearest = min(others, key=lambda m: abs(m - mean))
                sign = 1 if mean < nearest else -1
            else:
                sign = 1  # single row, default right
            for _, idx in row:
                row_directions[idx] = sign

        for idx in sorted(items):
            r = items[idx]
            length_mm = r.get("length_mm")
            if length_mm is None:
                continue

            c = r["center"]
            sz = r["size"]
            rack_left = c[axis] - sz[axis] / 2
            rack_right = c[axis] + sz[axis] / 2
            rack_cross = c[cross_axis]

            cross_sign = row_directions.get(idx, 1)
            bracket_cross = rack_cross + cross_sign * (
                sz[cross_axis] / 2 + self._WIDTH_OFFSET)

            # Anchor at the rack's inner face
            rack_inner = rack_cross + cross_sign * sz[cross_axis] / 2

            if axis == 1:
                edge_a = [bracket_cross, rack_left, 0.0]
                edge_b = [bracket_cross, rack_right, 0.0]
                anchor_a = [rack_inner, rack_left, 0.0]
                anchor_b = [rack_inner, rack_right, 0.0]
                tick_dir = [cross_sign * 0.03, 0, 0]
            else:
                edge_a = [rack_left, bracket_cross, 0.0]
                edge_b = [rack_right, bracket_cross, 0.0]
                anchor_a = [rack_left, rack_inner, 0.0]
                anchor_b = [rack_right, rack_inner, 0.0]
                tick_dir = [0, cross_sign * 0.03, 0]

            gaps.append(GapItem(
                edge_a, edge_b, length_mm, axis, True,
                anchor_a=anchor_a, anchor_b=anchor_b,
                tick_dir=tick_dir,
                color=self.gap_color,
                category=self.category,
            ))

        return bboxes, gaps


def _detect_corridor_axis(items, pairs):
    """Determine corridor axis (0=X, 1=Y) from neighbor pair offsets."""
    x_total = 0.0
    y_total = 0.0
    for a_idx, b_idx in pairs:
        ac = items[a_idx]["center"]
        bc = items[b_idx]["center"]
        x_total += abs(ac[0] - bc[0])
        y_total += abs(ac[1] - bc[1])
    return 1 if y_total >= x_total else 0


def _cluster_cross_rows(sorted_positions, gap_threshold=0.8):
    """Group sorted (cross_val, idx) pairs into rows by cross-axis gaps."""
    if not sorted_positions:
        return []
    rows = [[sorted_positions[0]]]
    for i in range(1, len(sorted_positions)):
        if sorted_positions[i][0] - sorted_positions[i - 1][0] > gap_threshold:
            rows.append([])
        rows[-1].append(sorted_positions[i])
    return rows


def _detect_corridor_axis_from_spread(items):
    """Fallback: determine corridor axis from coordinate spread."""
    xs = [r["center"][0] for r in items.values()]
    ys = [r["center"][1] for r in items.values()]
    dx = max(xs) - min(xs) if xs else 0
    dy = max(ys) - min(ys) if ys else 0
    return 1 if dy >= dx else 0


class RackMetadataHandler(MetadataHandler):
    """Parse rack_N_metadata.yaml files into bbox + gap annotations."""

    file_pattern = "rack_*_metadata.yaml"
    category = AnnotationCategory.RACK
    display_name = "Racks"
    bbox_color = (1.0, 0.5, 0.0)       # orange
    gap_color = (0.0, 0.85, 0.85)      # cyan
    neighbor_index_key = "rack_index"


class EmptySpaceMetadataHandler(MetadataHandler):
    """Parse empty_N_metadata.yaml files into bbox + gap annotations."""

    file_pattern = "empty_*_metadata.yaml"
    category = AnnotationCategory.EMPTY_SPACE
    display_name = "Empty Spaces"
    bbox_color = (1.0, 0.2, 0.2)       # red
    gap_color = (0.2, 0.9, 0.2)        # green
    neighbor_index_key = "empty_index"
