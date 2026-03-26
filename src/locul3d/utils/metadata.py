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

        if not pairs:
            return bboxes, []

        # Detect corridor axis from neighbor pairs
        axis = _detect_corridor_axis(items, pairs)
        cross_axis = 1 - axis

        # Build GapItems
        gaps = []
        for (a_idx, b_idx), gap_mm in pairs.items():
            ra, rb = items[a_idx], items[b_idx]
            ac, asz = ra["center"], ra["size"]
            bc, bsz = rb["center"], rb["size"]

            # Sort so a is the one with smaller corridor-axis position
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
