"""Metadata handlers for pipeline object-cluster directories."""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..core.geometry import AnnotationCategory, BBoxItem, GapItem, MeasurementType, WallDistStyle

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_all_metadata(directory: Path) -> Dict[str, dict]:
    """Read all *_metadata.yaml files, group by 'kind' field.

    Returns {kind_str: {sequential_idx: data_dict, ...}, ...}.
    Files without a 'kind' field are grouped under 'unknown'.
    """
    if not _HAS_YAML:
        return {}
    groups = defaultdict(dict)
    for path in sorted(directory.glob("*_metadata.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        kind = data.get("kind", "unknown")
        idx = len(groups[kind])
        groups[kind][idx] = data
    return dict(groups)


class MetadataHandler(ABC):
    """Base class for parsing metadata items into bbox + gap annotations."""

    kind: str  # YAML 'kind' field value, e.g. "rack", "mts", "region_subzone"
    category: AnnotationCategory
    display_name: str  # shown in Annotations panel, e.g. "Racks"
    bbox_color: tuple  # RGB 0-1
    gap_color: tuple  # RGB 0-1
    neighbor_index_key: str  # key in neighbor dict, e.g. "rack_index"
    use_item_z: bool = False  # True = annotations at item Z; False = at Z=0
    wall_dist_style: WallDistStyle = WallDistStyle.STAGGERED
    # Which measurement types to generate (empty set = bboxes only)
    enabled_measurements: frozenset = frozenset({
        MeasurementType.DIMENSION,
        MeasurementType.NEIGHBOR,
        MeasurementType.WALL_DISTANCE,
    })

    # Cross-axis offset from rack inner face (meters)
    _WIDTH_OFFSET = 0.10      # tier 1: width brackets
    _WALL_DIST_OFFSET = 0.22  # tier 2: wall distance brackets
    wall_dist_color: tuple = (1.0, 0.2, 0.2)  # red

    def parse(
        self, items: dict
    ) -> Tuple[List[BBoxItem], List[GapItem]]:
        """Parse pre-loaded metadata items → (bboxes, gaps).

        Args:
            items: {idx: data_dict, ...} — already grouped by kind.
        """
        if not items:
            return [], []

        # Detect format: new format has "width_mm", legacy has "length_mm"
        sample = next(iter(items.values()))
        new_format = "width_mm" in sample and "index" not in sample

        # Create BBoxItems (one per metadata file, sorted by index)
        bboxes = []
        sorted_indices = sorted(items)
        idx_to_bbox = {}  # item index → BBoxItem for gap linkage
        for idx in sorted_indices:
            r = items[idx]
            bbox = BBoxItem(
                label=r.get("id", self.category.value),
                center=r["center"],
                size=r["size"],
                color=list(self.bbox_color),
            )
            bboxes.append(bbox)
            idx_to_bbox[idx] = bbox

        if not self.enabled_measurements:
            return bboxes, []

        # Detect corridor axis
        axis = _detect_corridor_axis_from_spread(items)
        cross_axis = 1 - axis

        # Build neighbor gap pairs
        pairs = {}  # (min_idx, max_idx) → gap_mm
        if MeasurementType.NEIGHBOR not in self.enabled_measurements:
            pass
        elif new_format:
            pairs = _compute_neighbor_pairs_from_scalars(items, axis)
        else:
            # Legacy format: neighbor dicts with index and gap
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
            # Refine axis from explicit pairs if available
            if pairs:
                axis = _detect_corridor_axis(items, pairs)
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

            neighbor_gap = GapItem(
                edge_a, edge_b, gap_mm, axis, True,
                anchor_a=anchor_a, anchor_b=anchor_b,
                tick_dir=[0, 0, 0.03],
                color=self.gap_color,
                category=self.category,
            )
            neighbor_gap.measurement_type = MeasurementType.NEIGHBOR
            neighbor_gap.parent_bboxes = [
                b for b in [idx_to_bbox.get(a_idx), idx_to_bbox.get(b_idx)] if b]
            gaps.append(neighbor_gap)

        # Build dimension annotations at Z=0, offset into corridor
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
            if MeasurementType.DIMENSION not in self.enabled_measurements:
                break
            r = items[idx]
            # Support both "length_mm" (legacy) and "width_mm" (current)
            length_mm = r.get("length_mm") or r.get("width_mm")
            if length_mm is None:
                continue

            c = r["center"]
            sz = r["size"]
            ann_z = c[2] if self.use_item_z else 0.0
            rack_left = c[axis] - sz[axis] / 2
            rack_right = c[axis] + sz[axis] / 2
            rack_cross = c[cross_axis]

            cross_sign = row_directions.get(idx, 1)
            bracket_cross = rack_cross + cross_sign * (
                sz[cross_axis] / 2 + self._WIDTH_OFFSET)

            # Anchor at the rack's inner face
            rack_inner = rack_cross + cross_sign * sz[cross_axis] / 2

            if axis == 1:
                edge_a = [bracket_cross, rack_left, ann_z]
                edge_b = [bracket_cross, rack_right, ann_z]
                anchor_a = [rack_inner, rack_left, ann_z]
                anchor_b = [rack_inner, rack_right, ann_z]
                tick_dir = [cross_sign * 0.03, 0, 0]
            else:
                edge_a = [rack_left, bracket_cross, ann_z]
                edge_b = [rack_right, bracket_cross, ann_z]
                anchor_a = [rack_left, rack_inner, ann_z]
                anchor_b = [rack_right, rack_inner, ann_z]
                tick_dir = [0, cross_sign * 0.03, 0]

            width_gap = GapItem(
                edge_a, edge_b, length_mm, axis, True,
                anchor_a=anchor_a, anchor_b=anchor_b,
                tick_dir=tick_dir,
                color=self.gap_color,
                category=self.category,
            )
            width_gap.measurement_type = MeasurementType.DIMENSION
            owner = idx_to_bbox.get(idx)
            if owner:
                width_gap.parent_bboxes = [owner]
            gaps.append(width_gap)

        if MeasurementType.WALL_DISTANCE not in self.enabled_measurements:
            return bboxes, gaps

        # Build wall distance annotations at Z=0, staggered per row
        # Group racks by row for shared spine
        row_items = {}  # row_index → [(idx, data)]
        for ri, row in enumerate(rows):
            for _, idx in row:
                row_items.setdefault(ri, []).append((idx, items[idx]))

        for ri, rack_list in row_items.items():
            # Normalize wall distance data for both formats
            with_wall = []
            for idx, r in rack_list:
                wall_dist, wall_coord = _get_wall_distance(r, axis)
                if wall_dist is not None and wall_coord is not None:
                    with_wall.append((idx, r, wall_dist, wall_coord))
            if not with_wall:
                continue

            # Sort by distance (closest to wall → smallest offset)
            with_wall.sort(key=lambda x: x[2])

            wall_high = with_wall[0][3]
            cross_sign = row_directions.get(with_wall[0][0], 1)
            row_cross = sum(r["center"][cross_axis] for _, r, _, _ in with_wall) / len(with_wall)
            mean_half = sum(r["size"][cross_axis] / 2 for _, r, _, _ in with_wall) / len(with_wall)
            row_inner = row_cross + cross_sign * mean_half

            if self.wall_dist_style == WallDistStyle.COMB:
                gaps.extend(self._build_wall_dist_comb(
                    with_wall, idx_to_bbox, axis, cross_axis,
                    cross_sign, row_inner, wall_high))
            else:
                gaps.extend(self._build_wall_dist_staggered(
                    with_wall, idx_to_bbox, axis, cross_axis,
                    cross_sign, row_inner, wall_high))

        return bboxes, gaps

    def _build_wall_dist_staggered(self, with_wall, idx_to_bbox, axis,
                                    cross_axis, cross_sign, row_inner, wall_high):
        """Staggered style: one parallel line per item, offset from each other."""
        gaps = []
        for step_i, (idx, r, wall_dist, _wc) in enumerate(with_wall):
            c = r["center"]
            sz = r["size"]
            ann_z = c[2] if self.use_item_z else 0.0
            rack_far = c[axis] + sz[axis] / 2
            rack_inner = c[cross_axis] + cross_sign * sz[cross_axis] / 2
            bracket_cross = row_inner + cross_sign * (
                self._WALL_DIST_OFFSET + step_i * 0.08)

            if axis == 1:
                edge_a = [bracket_cross, rack_far, ann_z]
                edge_b = [bracket_cross, wall_high, ann_z]
                anchor_a = [rack_inner, rack_far, ann_z]
                anchor_b = [bracket_cross, wall_high, ann_z]
                tick_dir = [cross_sign * 0.03, 0, 0]
            else:
                edge_a = [rack_far, bracket_cross, ann_z]
                edge_b = [wall_high, bracket_cross, ann_z]
                anchor_a = [rack_far, rack_inner, ann_z]
                anchor_b = [wall_high, bracket_cross, ann_z]
                tick_dir = [0, cross_sign * 0.03, 0]

            wall_gap = GapItem(
                edge_a, edge_b, wall_dist, axis, True,
                anchor_a=anchor_a, anchor_b=anchor_b,
                tick_dir=tick_dir,
                color=self.wall_dist_color,
                category=self.category,
                label_t=0.05,
            )
            wall_gap.measurement_type = MeasurementType.WALL_DISTANCE
            owner = idx_to_bbox.get(idx)
            if owner:
                wall_gap.parent_bboxes = [owner]
            gaps.append(wall_gap)

        # Spine at wall_high connecting all bracket tips
        n = len(with_wall)
        spine_z = (sum(r["center"][2] for _, r, _, _ in with_wall)
                   / len(with_wall)) if self.use_item_z else 0.0
        inner_cross = row_inner + cross_sign * self._WALL_DIST_OFFSET
        outer_cross = row_inner + cross_sign * (
            self._WALL_DIST_OFFSET + (n - 1) * 0.08)
        if axis == 1:
            spine_a = [inner_cross, wall_high, spine_z]
            spine_b = [outer_cross, wall_high, spine_z]
        else:
            spine_a = [wall_high, inner_cross, spine_z]
            spine_b = [wall_high, outer_cross, spine_z]
        spine = GapItem(
            spine_a, spine_b, None, cross_axis, True,
            anchor_a=spine_a, anchor_b=spine_b,
            tick_dir=[0, 0, 0],
            color=self.wall_dist_color,
            category=self.category,
        )
        spine.measurement_type = MeasurementType.WALL_DISTANCE
        spine.parent_bboxes = [
            b for b in [idx_to_bbox.get(idx) for idx, _, _, _ in with_wall] if b]
        gaps.append(spine)
        return gaps

    def _build_wall_dist_comb(self, with_wall, idx_to_bbox, axis,
                               cross_axis, cross_sign, row_inner, wall_high):
        """Comb style: central spine along corridor, cross-axis ticks to item projections.

        Layout (for axis=1, Y=corridor, X=cross):
          - Spine: vertical line along corridor axis (Y) at a central X,
            from first to last item Y position
          - Ticks: horizontal lines from spine to each item's X position,
            at that item's Y — labeled with wall distance
          - Projections: short corridor-axis bars at each item's floor
            footprint showing the projected item outline
        """
        gaps = []
        all_parents = [b for b in [idx_to_bbox.get(idx) for idx, _, _, _ in with_wall] if b]

        # Spine X position: offset from row inner face into the corridor
        spine_cross = row_inner + cross_sign * self._WALL_DIST_OFFSET

        # Spine extent: from wall to farthest item from wall
        item_positions = []
        for _, r, _, _ in with_wall:
            c = r["center"]
            sz = r["size"]
            item_positions.append(c[axis] - sz[axis] / 2)
            item_positions.append(c[axis] + sz[axis] / 2)
        # Find the item edge farthest from wall_high
        farthest = max(item_positions, key=lambda p: abs(p - wall_high))
        spine_start = wall_high
        spine_end = farthest
        spine_z = (sum(r["center"][2] for _, r, _, _ in with_wall)
                   / len(with_wall)) if self.use_item_z else 0.0

        # Spine: runs along corridor axis at spine_cross
        if axis == 1:
            spine_a = [spine_cross, spine_start, spine_z]
            spine_b = [spine_cross, spine_end, spine_z]
        else:
            spine_a = [spine_start, spine_cross, spine_z]
            spine_b = [spine_end, spine_cross, spine_z]
        spine = GapItem(
            spine_a, spine_b, None, axis, True,
            anchor_a=spine_a, anchor_b=spine_b,
            tick_dir=[0, 0, 0],
            color=self.wall_dist_color,
            category=self.category,
        )
        spine.measurement_type = MeasurementType.WALL_DISTANCE
        spine.parent_bboxes = list(all_parents)
        gaps.append(spine)

        for idx, r, wall_dist, _wc in with_wall:
            c = r["center"]
            sz = r["size"]
            item_cross = c[cross_axis]
            item_corridor = c[axis]
            ann_z = c[2] if self.use_item_z else 0.0

            # Tick: from spine to near edge of item projection bar
            half_cross = sz[cross_axis] / 2
            item_near_edge = item_cross + cross_sign * half_cross
            if axis == 1:
                edge_a = [spine_cross, item_corridor, ann_z]
                edge_b = [item_near_edge, item_corridor, ann_z]
                tick_dir = [0, 0, 0.03]
            else:
                edge_a = [item_corridor, spine_cross, ann_z]
                edge_b = [item_corridor, item_near_edge, ann_z]
                tick_dir = [0, 0, 0.03]

            tick = GapItem(
                edge_a, edge_b, wall_dist, cross_axis, True,
                anchor_a=edge_a, anchor_b=edge_b,
                tick_dir=tick_dir,
                color=self.wall_dist_color,
                category=self.category,
                label_t=0.5,
            )
            tick.measurement_type = MeasurementType.WALL_DISTANCE
            owner = idx_to_bbox.get(idx)
            if owner:
                tick.parent_bboxes = [owner]
            gaps.append(tick)

            # Projection: item wall face projected on floor (along cross-axis)
            if axis == 1:
                proj_a = [item_cross - half_cross, item_corridor, ann_z]
                proj_b = [item_cross + half_cross, item_corridor, ann_z]
            else:
                proj_a = [item_corridor, item_cross - half_cross, ann_z]
                proj_b = [item_corridor, item_cross + half_cross, ann_z]
            proj = GapItem(
                proj_a, proj_b, None, cross_axis, True,
                anchor_a=proj_a, anchor_b=proj_b,
                tick_dir=[0, 0, 0],
                color=self.bbox_color,
                category=self.category,
            )
            proj.measurement_type = MeasurementType.WALL_DISTANCE
            if owner:
                proj.parent_bboxes = [owner]
            gaps.append(proj)

        return gaps


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


def _compute_neighbor_pairs_from_scalars(items, axis):
    """Build neighbor pairs from new-format neighbor_left_mm/neighbor_right_mm.

    Sorts racks by corridor-axis position within each cross-axis row.
    For consecutive racks, uses neighbor_right_mm of the lower-position
    rack (or neighbor_left_mm of the higher-position rack) as the gap.
    """
    cross_axis = 1 - axis
    cross_positions = sorted(
        [(r["center"][cross_axis], idx) for idx, r in items.items()])
    rows = _cluster_cross_rows(cross_positions)

    pairs = {}
    for row in rows:
        # Sort row items by corridor-axis position
        row_sorted = sorted(row, key=lambda x: items[x[1]]["center"][axis])
        for i in range(len(row_sorted) - 1):
            _, idx_a = row_sorted[i]
            _, idx_b = row_sorted[i + 1]
            ra, rb = items[idx_a], items[idx_b]
            # Prefer neighbor_right_mm of left rack, fall back to neighbor_left_mm of right rack
            gap_mm = ra.get("neighbor_right_mm")
            if gap_mm is None:
                gap_mm = rb.get("neighbor_left_mm")
            if gap_mm is not None:
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                pairs[key] = float(gap_mm)
    return pairs


def _get_wall_distance(rack_data, axis):
    """Extract wall distance (mm) and wall coordinate for a rack.

    Returns (distance_mm, wall_coordinate) or (None, None).

    Supports:
      - Legacy: "distance_to_high_wall_mm" + "wall_high"
      - Current: "distance_wall_a_mm" + "distance_wall_b_mm"
        Always uses wall B (high end of corridor axis).
    """
    # Legacy format
    if "distance_to_high_wall_mm" in rack_data and "wall_high" in rack_data:
        return rack_data["distance_to_high_wall_mm"], rack_data["wall_high"]

    # Current format — always use wall B
    db = rack_data.get("distance_wall_b_mm")
    if db is None:
        return None, None

    c = rack_data["center"]
    sz = rack_data["size"]
    wall_coord = c[axis] + sz[axis] / 2 + db / 1000.0
    return db, wall_coord


class RackMetadataHandler(MetadataHandler):
    kind = "rack"
    category = AnnotationCategory.RACK
    display_name = "Racks"
    bbox_color = (1.0, 0.5, 0.0)       # orange
    gap_color = (0.0, 0.85, 0.85)      # cyan
    neighbor_index_key = "rack_index"


class EmptySpaceMetadataHandler(MetadataHandler):
    kind = "empty_space"
    category = AnnotationCategory.EMPTY_SPACE
    display_name = "Empty Spaces"
    bbox_color = (0.2, 0.4, 1.0)       # blue
    gap_color = (0.2, 0.9, 0.2)        # green
    neighbor_index_key = "empty_index"


class RackRegionMetadataHandler(MetadataHandler):
    kind = "region_subzone"
    category = AnnotationCategory.REGION_SUBZONE
    display_name = "Rack Regions"
    bbox_color = (0.5, 0.5, 0.5)       # grey
    gap_color = (0.4, 0.4, 0.4)
    neighbor_index_key = "region_index"
    enabled_measurements = frozenset()  # bboxes only


class MtsMetadataHandler(MetadataHandler):
    kind = "mts_stack"
    category = AnnotationCategory.MTS
    display_name = "MTS Stacks"
    bbox_color = (1.0, 0.8, 0.2)       # yellow
    gap_color = (0.9, 0.6, 0.1)        # darker gold
    neighbor_index_key = "mts_index"
    wall_dist_style = WallDistStyle.COMB
    enabled_measurements = frozenset({
        MeasurementType.NEIGHBOR,
        MeasurementType.WALL_DISTANCE,
    })


class MtsBoxMetadataHandler(MetadataHandler):
    kind = "mts_box"
    category = AnnotationCategory.MTS_BOX
    display_name = "MTS Boxes"
    bbox_color = (0.2, 0.8, 1.0)       # cyan
    gap_color = (0.1, 0.6, 0.9)        # darker cyan
    neighbor_index_key = "mts_box_index"
    use_item_z = True
    enabled_measurements = frozenset({MeasurementType.DIMENSION})


# Registry: kind → handler instance
METADATA_HANDLERS = {h.kind: h for h in [
    RackMetadataHandler(),
    EmptySpaceMetadataHandler(),
    RackRegionMetadataHandler(),
    MtsMetadataHandler(),
    MtsBoxMetadataHandler(),
]}
