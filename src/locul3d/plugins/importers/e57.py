"""E57 file importer — full processing pipeline (first-class citizen).

This module provides:
  - E57ImportResult: container for pipeline output
  - E57ImportWorker: QThread background worker with staged pipeline
  - E57ProgressDialog: modal dialog with per-stage progress, log, timer
  - E57Importer: plugin interface wrapper

Requires:
  - pye57 (E57 file reading)
  - open3d (point cloud processing)
  - libe57 + Pillow (optional, for panorama extraction)
"""

import os
import sys
import time
import copy
from pathlib import Path
from typing import Optional, List

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QTextEdit,
)
from PySide6.QtGui import QTextCursor

from ...core.layer import LayerData
from ...core.constants import COLORS

# Optional heavy dependencies -------------------------------------------------

try:
    import pye57
    HAS_PYE57 = True
    _PYE57_ERR = ""
except ImportError as _e:
    HAS_PYE57 = False
    _PYE57_ERR = str(_e)
except Exception as _e:
    HAS_PYE57 = False
    _PYE57_ERR = f"unexpected: {_e}"

try:
    import open3d as o3d
    HAS_O3D = True
    _O3D_ERR = ""
except ImportError as _e:
    HAS_O3D = False
    _O3D_ERR = str(_e)
except Exception as _e:
    HAS_O3D = False
    _O3D_ERR = f"unexpected: {_e}"

try:
    from scipy.spatial import ConvexHull, cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import libe57
    HAS_LIBE57 = True
except ImportError:
    try:
        from pye57 import libe57
        HAS_LIBE57 = True
    except ImportError:
        HAS_LIBE57 = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Processing parameters -------------------------------------------------------

E57_SOR_K_NEIGHBORS = 20
E57_SOR_STD_RATIO = 2.0
E57_VOXEL_SIZE_FILTER = 0.05       # 50mm pre-decimation
E57_VOXEL_SIZE_MESH = 0.15         # 150mm final decimation
E57_RANSAC_DIST_THRESHOLD = 0.05   # 50mm plane fitting tolerance
E57_RANSAC_MIN_POINTS = 2000
E57_MAX_PLANES = 15
E57_CROP_RADIUS = None             # None = load all points

# Scenes with more points than this skip the raw_scan layer to save
# RAM.  348M pts × 24 bytes ≈ 7.8 GB — holding that in memory alongside
# the aligned + decimated copies is wasteful for interactive use.
E57_RAW_LAYER_POINT_THRESHOLD = 50_000_000  # 50M — skip raw layer above this

E57_SURFACE_COLORS = [
    [0.90, 0.10, 0.10], [0.10, 0.80, 0.10], [0.10, 0.10, 0.90],
    [0.90, 0.90, 0.10], [0.90, 0.10, 0.90], [0.10, 0.90, 0.90],
    [1.00, 0.50, 0.00], [0.50, 0.00, 1.00], [0.00, 0.50, 0.00],
    [0.80, 0.40, 0.40], [0.40, 0.80, 0.40], [0.40, 0.40, 0.80],
    [0.80, 0.60, 0.20], [0.60, 0.20, 0.80], [0.20, 0.80, 0.60],
]

# Stage definitions for progress dialog
_E57_STAGES = [
    ("Ingestion", "Reading E57 point cloud data"),
    ("Filtering", "Voxel downsampling + noise removal"),
    ("Alignment", "Ground plane detection + Z-up rotation"),
    ("Decimation", "Final voxel downsampling"),
    ("Building Layers", "Creating viewer layers from data"),
]


# =============================================================================
# E57ImportResult
# =============================================================================

class E57ImportResult:
    """Container for pipeline results passed back to the main thread."""

    def __init__(self):
        self.layers: List[LayerData] = []
        self.base_dir: str = ""
        self.error: Optional[str] = None
        self.metadata: dict = {}
        self.stats: dict = {}


# =============================================================================
# E57ImportWorker  (QThread)
# =============================================================================

class E57ImportWorker(QThread):
    """Background worker: ingests an E57 file, runs processing pipeline,
    emits progress signals for each step."""

    stage_started = Signal(str, str)
    stage_progress = Signal(str, int)
    log_message = Signal(str)
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(self, e57_path: str, parent=None):
        super().__init__(parent)
        self._path = e57_path
        self._cancelled = False
        # Alignment transform (set by _stage_align)
        self._align_R = np.eye(3)        # rotation matrix
        self._align_center = np.zeros(3) # rotation center
        self._align_z_shift = 0.0        # vertical translation

    def cancel(self):
        self._cancelled = True

    def _log(self, msg: str):
        self.log_message.emit(msg)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        try:
            result = self._run_pipeline()
            if self._cancelled:
                self.finished_err.emit("Import cancelled")
            else:
                self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")

    def _run_pipeline(self) -> E57ImportResult:
        import gc

        result = E57ImportResult()
        result.base_dir = str(Path(self._path).parent)
        stats = {}
        t_total = time.time()

        # Stage 1: Ingest — returns raw numpy arrays (float32 + uint8)
        self.stage_started.emit("Ingestion", "Reading E57 file...")
        t0 = time.time()
        xyz, colors_u8, meta = self._stage_ingest()
        stats["ingest_time"] = time.time() - t0
        stats["points_after_crop"] = len(xyz)
        if self._cancelled:
            return result

        # Keep raw arrays for the raw scan layer
        raw_arrays = {"points": xyz, "colors_u8": colors_u8}

        # Stride-subsample: 348M → ~5M pts in O(1).
        # We only need this reduced set for the O3D processing pipeline
        # (SOR, RANSAC align, decimate).  The full raw data stays in
        # raw_arrays for the raw_scan layer.
        target_pts = 5_000_000
        stride = max(1, len(xyz) // target_pts)
        ds_xyz = xyz[::stride]
        ds_colors = colors_u8[::stride] if colors_u8 is not None else None
        n_ds = len(ds_xyz)
        self._log(f"Stride subsample: {len(xyz):,} → {n_ds:,} pts "
                  f"(stride={stride})")

        # Build O3D cloud from the small subset only
        t0 = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ds_xyz.astype(np.float64))
        if ds_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                ds_colors.astype(np.float64) / 255.0)
        self._log(f"O3D cloud from {n_ds:,} pts: {time.time() - t0:.1f}s")
        del ds_xyz, ds_colors
        gc.collect()

        if self._cancelled:
            return result

        # Stage 2: Filter (voxel + SOR on the ~5M stride subset)
        self.stage_started.emit("Filtering", "Voxel downsampling + noise removal...")
        t0 = time.time()
        pcd_filtered = self._stage_filter(pcd)
        stats["filter_time"] = time.time() - t0
        stats["points_after_filter"] = len(pcd_filtered.points)
        del pcd
        gc.collect()
        if self._cancelled:
            return result

        # Stage 3: Align
        self.stage_started.emit("Alignment", "Detecting ground plane...")
        t0 = time.time()
        pcd_aligned = self._stage_align(pcd_filtered)
        stats["align_time"] = time.time() - t0
        stats["points_after_align"] = len(pcd_aligned.points)
        if pcd_filtered is not pcd_aligned:
            del pcd_filtered
            gc.collect()
        if self._cancelled:
            return result

        surfaces = []
        surface_clouds = []
        measurements = {}

        # Stage 4: Decimate
        self.stage_started.emit("Decimation", "Final voxel downsampling...")
        t0 = time.time()
        pcd_decimated = self._stage_decimate(pcd_aligned)
        stats["decimate_time"] = time.time() - t0
        stats["points_after_decimate"] = len(pcd_decimated.points)
        if self._cancelled:
            return result

        # Stage 5: Build layers
        self.stage_started.emit("Building Layers", "Creating viewer layers...")
        t0 = time.time()
        layers = self._build_layers(
            raw_arrays, pcd_aligned, pcd_decimated,
            surfaces, surface_clouds, measurements,
        )
        stats["build_time"] = time.time() - t0

        del pcd_aligned, pcd_decimated
        gc.collect()

        # Panorama extraction
        pano_layers = self._extract_panoramas(self._path)
        self._align_panorama_layers(pano_layers)
        layers.extend(pano_layers)

        stats["total_time"] = time.time() - t_total
        stats["surfaces"] = []

        result.layers = layers
        result.metadata = meta
        result.stats = stats
        return result

    # ------------------------------------------------------------------
    # Stage 1: Ingestion — fast binary E57 reader + libe57 fallback
    # ------------------------------------------------------------------

    def _stage_ingest(self):
        """Read E57 into float32 XYZ + uint8 RGB arrays.

        Attempts a fast binary read first (bypasses libE57Format's per-
        packet codec, ~6s for 348M pts).  Falls back to libe57 if the
        binary structure is non-standard (~27s).
        """
        import gc, struct

        path = self._path
        file_size_mb = os.path.getsize(path) / 1024 / 1024
        self._log(f"File: {Path(path).name} ({file_size_mb:.1f} MB)")

        # Read header & metadata via pye57 (fast — just XML)
        e57 = pye57.E57(path)
        header = e57.get_header(0)
        n = header.point_count
        self._log(f"Scans: {e57.scan_count} | Points: {n:,}")
        self._log(f"Fields: {header.point_fields}")

        meta = {"images": []}
        pose_meta = {}
        if hasattr(header, "rotation") and header.rotation is not None:
            pose_meta["rotation_quaternion"] = list(header.rotation)
        if hasattr(header, "translation") and header.translation is not None:
            pose_meta["translation"] = list(header.translation)
        meta["scan_pose"] = pose_meta
        meta["file_name"] = Path(path).name
        meta["file_size_mb"] = round(file_size_mb, 1)
        meta["format"] = "E57"
        meta["scan_count"] = e57.scan_count
        meta["raw_point_count"] = n
        meta["point_fields"] = list(header.point_fields)
        available = set(header.point_fields)
        has_rgb = all(f in available for f in
                      ('colorRed', 'colorGreen', 'colorBlue'))
        e57.close()

        self.stage_progress.emit("Reading E57 data...", 10)

        # --- Try fast binary path ---
        xyz, colors_u8 = None, None
        try:
            xyz, colors_u8 = self._fast_binary_read(path, n, has_rgb)
        except Exception as exc:
            self._log(f"Fast binary read failed ({exc}), "
                      "falling back to libe57")

        # --- Fallback: libe57 SourceDestBuffer ---
        if xyz is None:
            xyz, colors_u8 = self._libe57_read(path, n, has_rgb, available)

        self.stage_progress.emit("Processing...", 60)

        scanner_pos = np.array(
            pose_meta.get("translation", xyz.mean(axis=0).tolist()),
            dtype=np.float32,
        )
        self._log(f"Scanner position: [{scanner_pos[0]:.2f}, "
                  f"{scanner_pos[1]:.2f}, {scanner_pos[2]:.2f}]")
        meta["scanner_position_original"] = scanner_pos.tolist()

        # Optional crop
        if E57_CROP_RADIUS is not None:
            dists = np.linalg.norm(xyz - scanner_pos, axis=1)
            crop_mask = dists < E57_CROP_RADIUS
            n_cropped = int(crop_mask.sum())
            self._log(f"Cropping to {E57_CROP_RADIUS}m: "
                      f"{n_cropped:,} pts ({100 * n_cropped / len(xyz):.1f}%)")
            xyz = xyz[crop_mask] - scanner_pos
            if colors_u8 is not None:
                colors_u8 = colors_u8[crop_mask]
        else:
            n_cropped = len(xyz)

        bb_min = xyz.min(axis=0)
        bb_max = xyz.max(axis=0)
        bb_size = bb_max - bb_min
        self._log(f"Bounding box: [{bb_size[0]:.1f} x "
                  f"{bb_size[1]:.1f} x {bb_size[2]:.1f}] m")

        meta["crop_radius_m"] = (E57_CROP_RADIUS
                                 if E57_CROP_RADIUS is not None
                                 else "disabled")
        meta["cropped_point_count"] = n_cropped
        meta["has_rgb"] = has_rgb
        meta["has_intensity"] = "intensity" in available
        meta["bounding_box_m"] = [round(float(v), 2)
                                  for v in bb_size.tolist()]
        meta["bb_min"] = [round(float(v), 2) for v in bb_min.tolist()]
        meta["bb_max"] = [round(float(v), 2) for v in bb_max.tolist()]

        mem_mb = xyz.nbytes / 1024**2
        if colors_u8 is not None:
            mem_mb += colors_u8.nbytes / 1024**2
        self._log(f"Result: {len(xyz):,} pts, {mem_mb:.0f} MB (f32+u8)")

        self.stage_progress.emit("Done", 100)
        return xyz, colors_u8, meta

    # ------------------------------------------------------------------
    # Fast binary E57 reader — bypasses libE57Format codec
    # ------------------------------------------------------------------

    def _fast_binary_read(self, path, n, has_rgb):
        """Read E57 data packets directly via numpy.

        E57 files store data in CRC-protected 1024-byte pages.  Each
        page has 1020 data bytes + 4-byte CRC.  Data is organized as
        CompressedVector packets with per-field bytestreams.

        For uncompressed E57 files (float32 XYZ + uint8 RGB, ~1:1
        compression ratio), we can read 4× faster than libE57Format
        by stripping CRC pages and extracting packets vectorized.

        Raises ValueError if the packet structure is non-standard.
        """
        import struct
        import gc

        PAGE_SIZE = 1024
        DATA_PER_PAGE = 1020

        # 1. Read entire file (SSD-speed, ~0.8s for 5GB)
        t0 = time.time()
        with open(path, 'rb') as f:
            raw_arr = np.fromfile(f, dtype=np.uint8)
        self._log(f"File read: {time.time() - t0:.2f}s "
                  f"({len(raw_arr) / (time.time() - t0) / 1024**3:.1f} GB/s)")

        # 2. Strip CRC pages (numpy reshape + slice, ~0.7s)
        t0 = time.time()
        n_pages = len(raw_arr) // PAGE_SIZE
        pages = raw_arr[:n_pages * PAGE_SIZE].reshape(n_pages, PAGE_SIZE)
        logical = np.ascontiguousarray(pages[:, :DATA_PER_PAGE]).ravel()
        self._log(f"CRC strip: {time.time() - t0:.2f}s")
        del raw_arr, pages
        gc.collect()

        # 3. Probe first data packet to detect structure
        #    The CompressedVector data section starts at physical
        #    offset 48 (per XML fileOffset).  The first real data
        #    packet (type=1, bcount≥6) usually starts at logical 80
        #    after a small section-index header.

        # Search for first data packet (type=1, bcount≥6, consistent structure)
        probe_pos = 48
        pkt0_pos = None
        while probe_pos < min(len(logical), 8192):
            if logical[probe_pos] == 1:  # type = data
                # Read enough for header + bytestream count
                hdr_raw = logical[probe_pos:probe_pos + 6].tobytes()
                if len(hdr_raw) < 6:
                    break
                _, _, plen_m1, bcount = struct.unpack('<BBHH', hdr_raw)
                plen = plen_m1 + 1

                # Structural verification: plen must be header + sum(bslens)
                if 6 < bcount < 64 and plen > (6 + bcount * 2):
                    sl_raw = logical[probe_pos + 6:probe_pos + 6 + bcount * 2].tobytes()
                    if len(sl_raw) == bcount * 2:
                        bslens = struct.unpack(f'<{bcount}H', sl_raw)
                        header_size = 6 + bcount * 2
                        if sum(bslens) + header_size == plen and bslens[0] > 0:
                            pkt0_pos = probe_pos
                            break
            probe_pos += 1

        if pkt0_pos is None:
            raise ValueError("Could not find first data packet")

        # Parse first packet header using small tobytes() on header only
        hdr_bytes = logical[pkt0_pos:pkt0_pos + 6].tobytes()
        _, _, plen0_m1, bcount0 = struct.unpack('<BBHH', hdr_bytes)
        pkt_size = plen0_m1 + 1
        pkt_header = 6 + bcount0 * 2
        sl_bytes = logical[pkt0_pos + 6:pkt0_pos + pkt_header].tobytes()
        bslens0 = struct.unpack(f'<{bcount0}H', sl_bytes)
        pts_per_pkt = bslens0[0] // 4  # float32 XYZ

        self._log(f"Packet: {pkt_size} bytes, {bcount0} streams, "
                  f"{pts_per_pkt} pts/pkt")

        # Validate: uniform float32 XYZ + uint8 RGB structure
        if bcount0 < 6:
            raise ValueError(f"Expected ≥6 bytestreams, got {bcount0}")
        if pts_per_pkt == 0:
            raise ValueError("Zero points in first packet")
        for ax in range(3):
            if bslens0[ax] != pts_per_pkt * 4:
                raise ValueError(
                    f"Stream {ax} size {bslens0[ax]} != "
                    f"expected {pts_per_pkt * 4} (non-float32 XYZ?)")
        if has_rgb:
            for ch in range(3):
                if bslens0[3 + ch] != pts_per_pkt:
                    raise ValueError(
                        f"Stream {3+ch} size {bslens0[3+ch]} != "
                        f"expected {pts_per_pkt} (non-uint8 RGB?)")

        # 4. Vectorized extraction using ravel() — no intermediate
        #    tobytes() copy, no np.ascontiguousarray() overhead.
        #    ravel() on a strided view does the contiguous copy
        #    internally in C with optimal cache access patterns.
        n_full_pkts = n // pts_per_pkt
        n_remainder = n % pts_per_pkt
        n_full = n_full_pkts * pts_per_pkt

        t0 = time.time()
        # Reshape logical numpy array into packet rows
        pkt_view = logical[pkt0_pos:pkt0_pos + n_full_pkts * pkt_size]
        pkt_view = pkt_view.reshape(n_full_pkts, pkt_size)

        # XYZ: slice columns from packet view, ravel, reinterpret as f32
        xyz = np.empty((n, 3), dtype=np.float32)
        for ax in range(3):
            off = pkt_header + sum(bslens0[:ax])
            sz = bslens0[ax]
            col = pkt_view[:, off:off + sz].ravel()
            xyz[:n_full, ax] = col.view(np.float32)

        # RGB: same approach, uint8 columns
        colors_u8 = None
        if has_rgb:
            colors_u8 = np.empty((n, 3), dtype=np.uint8)
            for ch in range(3):
                off = pkt_header + sum(bslens0[:3 + ch])
                sz = bslens0[3 + ch]
                colors_u8[:n_full, ch] = pkt_view[:, off:off + sz].ravel()
        del pkt_view

        # Handle last (shorter) packet
        if n_remainder > 0:
            last_pos = pkt0_pos + n_full_pkts * pkt_size
            last_sl = struct.unpack_from(
                f'<{bcount0}H',
                logical[last_pos + 6:last_pos + pkt_header].tobytes())
            dp = last_pos + pkt_header
            for ax in range(3):
                sl = last_sl[ax]
                npts = sl // 4
                tmp = logical[dp:dp + sl].copy()
                xyz[n_full:n_full + npts, ax] = tmp.view(np.float32)
                dp += sl
            if has_rgb:
                for ch in range(3):
                    sl = last_sl[3 + ch]
                    colors_u8[n_full:n_full + sl, ch] = logical[dp:dp + sl]
                    dp += sl

        t_parse = time.time() - t0
        self._log(f"Binary parse: {t_parse:.1f}s "
                  f"({n / t_parse / 1e6:.0f}M pts/s)")
        gc.collect()
        return xyz, colors_u8

    # ------------------------------------------------------------------
    # Fallback: libe57 SourceDestBuffer reader
    # ------------------------------------------------------------------

    def _libe57_read(self, path, n, has_rgb, available):
        """Read via libe57 C++ bindings (slower but more robust)."""
        import gc
        from pye57 import libe57 as _libe57

        e57 = pye57.E57(path)
        header = e57.get_header(0)
        buffers = _libe57.VectorSourceDestBuffer()
        arrays = {}

        for field in ('cartesianX', 'cartesianY', 'cartesianZ'):
            arr = np.empty(n, dtype=np.float32)
            buf = _libe57.SourceDestBuffer(
                e57.image_file, field, arr, n, True, True)
            arrays[field] = arr
            buffers.append(buf)

        if has_rgb:
            for field in ('colorRed', 'colorGreen', 'colorBlue'):
                arr = np.empty(n, dtype=np.uint8)
                buf = _libe57.SourceDestBuffer(
                    e57.image_file, field, arr, n, True, True)
                arrays[field] = arr
                buffers.append(buf)

        t0 = time.time()
        header.points.reader(buffers).read()
        self._log(f"libe57 read: {time.time() - t0:.1f}s")
        e57.close()

        xyz = np.column_stack([
            arrays['cartesianX'],
            arrays['cartesianY'],
            arrays['cartesianZ'],
        ])
        colors_u8 = None
        if has_rgb:
            colors_u8 = np.column_stack([
                arrays['colorRed'],
                arrays['colorGreen'],
                arrays['colorBlue'],
            ])
        gc.collect()
        return xyz, colors_u8

    # ------------------------------------------------------------------
    # Stage 2: Filter
    # ------------------------------------------------------------------

    def _stage_filter(self, pcd):
        n_orig = len(pcd.points)
        self._log(f"Input points: {n_orig:,}")

        self.stage_progress.emit("Voxel downsampling...", 20)
        pcd = pcd.voxel_down_sample(voxel_size=E57_VOXEL_SIZE_FILTER)
        n_down = len(pcd.points)
        self._log(f"After {E57_VOXEL_SIZE_FILTER*1000:.0f}mm voxel: "
                  f"{n_down:,} ({100 * n_down / n_orig:.1f}%)")

        self.stage_progress.emit("Statistical outlier removal...", 50)
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=E57_SOR_K_NEIGHBORS, std_ratio=E57_SOR_STD_RATIO)
        n_clean = len(pcd_clean.points)
        self._log(f"After SOR: {n_clean:,} (removed {n_down - n_clean:,})")
        self.stage_progress.emit("Done", 100)
        return pcd_clean

    def _stage_filter_sor_only(self, pcd):
        """Statistical outlier removal only (voxel done in numpy)."""
        n_orig = len(pcd.points)
        self._log(f"SOR input: {n_orig:,} pts (pre-downsampled)")
        self.stage_progress.emit("Statistical outlier removal...", 50)
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=E57_SOR_K_NEIGHBORS, std_ratio=E57_SOR_STD_RATIO)
        n_clean = len(pcd_clean.points)
        self._log(f"After SOR: {n_clean:,} (removed {n_orig - n_clean:,})")
        self.stage_progress.emit("Done", 100)
        return pcd_clean

    # ------------------------------------------------------------------
    # Stage 3: Align
    # ------------------------------------------------------------------

    def _stage_align(self, pcd):
        pts = np.asarray(pcd.points)
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        z_range = z_max - z_min
        self._log(f"Input: {len(pts):,} points")
        self._log(f"Z range: [{z_min:.2f}, {z_max:.2f}] (span: {z_range:.2f}m)")

        z_cutoff = z_min + 0.20 * z_range
        floor_mask = pts[:, 2] < z_cutoff
        floor_indices = np.where(floor_mask)[0]
        self._log(f"Floor search: Z < {z_cutoff:.2f} ({len(floor_indices):,} points)")

        if len(floor_indices) < 500:
            self._log("WARNING: Too few floor points, skipping alignment")
            self.stage_progress.emit("Skipped", 100)
            return pcd

        self.stage_progress.emit("RANSAC plane fitting...", 30)
        floor_cloud = pcd.select_by_index(floor_indices.tolist())
        plane_model, inliers = floor_cloud.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=2000)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal

        self._log(f"Ground normal: [{normal[0]:.4f}, {normal[1]:.4f}, "
                  f"{normal[2]:.4f}]  ({len(inliers):,} inliers)")

        self.stage_progress.emit("Rotating...", 60)
        target = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c_val = np.dot(normal, target)

        if s < 1e-6:
            self._log("Ground already aligned")
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c_val))
            angle_deg = np.degrees(np.arcsin(min(s, 1.0)))
            self._log(f"Rotation: {angle_deg:.2f} degrees")

        # Store rotation center BEFORE applying rotation
        rot_center = np.array(pcd.get_center())
        pcd.rotate(R, center=rot_center)

        self.stage_progress.emit("Shifting to ground...", 80)
        pts_aligned = np.asarray(pcd.points)
        z_floor = np.percentile(pts_aligned[:, 2], 5)
        pcd.translate([0, 0, -z_floor])

        # Store alignment transform for child elements (panoramas, etc.)
        self._align_R = R
        self._align_center = rot_center
        self._align_z_shift = z_floor

        # RMSE check
        pts_final = np.asarray(pcd.points)
        near_floor = pts_final[np.abs(pts_final[:, 2]) < 0.15]
        if len(near_floor) > 100:
            floor_rmse = np.sqrt(np.mean(near_floor[:, 2] ** 2))
            self._log(f"Floor RMSE: {floor_rmse * 1000:.1f} mm")
            if floor_rmse > 0.015:
                self._log("WARNING: RMSE > 15mm - possible drift")

        self._log(f"Output: {len(pcd.points):,} points")
        self.stage_progress.emit("Done", 100)
        return pcd

    # ------------------------------------------------------------------
    # Stage 4: Decimate
    # ------------------------------------------------------------------

    def _stage_decimate(self, pcd):
        n_before = len(pcd.points)
        self._log(f"Input: {n_before:,} points")

        self.stage_progress.emit("Voxel downsampling...", 50)
        pcd_dec = pcd.voxel_down_sample(voxel_size=E57_VOXEL_SIZE_MESH)
        n_after = len(pcd_dec.points)
        self._log(f"Output: {n_after:,} points "
                  f"({100 * (1 - n_after / n_before):.1f}% reduction)")
        self.stage_progress.emit("Done", 100)
        return pcd_dec

    # ------------------------------------------------------------------
    # Build layers
    # ------------------------------------------------------------------

    def _build_layers(self, raw_arrays, pcd_aligned, pcd_decimated,
                      surfaces, surface_clouds, measurements):
        import gc

        layers = []

        # Raw scan layer — always present for resolution switching
        self.stage_progress.emit("Raw scan...", 5)
        # Apply alignment transform to raw points so all layers share
        # the same coordinate system (rotation + Z-shift from _stage_align).
        raw_pts = raw_arrays["points"].astype(np.float64)
        raw_pts = (self._align_R @ (raw_pts - self._align_center).T).T + self._align_center
        raw_pts[:, 2] -= self._align_z_shift
        raw_arrays["points"] = raw_pts.astype(np.float32)
        del raw_pts
        layers.append(self._make_raw_layer(raw_arrays))
        del raw_arrays
        gc.collect()

        self.stage_progress.emit("Mid-res cloud...", 10)
        layers.append(self._make_pcd_layer(
            pcd_aligned, "midres", "Mid-Resolution Point Cloud",
            visible=True, opacity=0.6))

        # Per-surface clouds
        for i, surf in enumerate(surfaces):
            pct = 10 + int(60 * i / max(len(surfaces), 1))
            self.stage_progress.emit(f"Surface {i}...", pct)
            layers.append(self._make_pcd_layer(
                surface_clouds[i], f"surface_{i}",
                f"Surface {i}: {surf['orientation']}",
                visible=False, color=surf["color"], opacity=1.0))

        # Unclassified remainder
        remainder = surface_clouds[-1] if len(surface_clouds) > len(surfaces) else None
        if remainder and len(remainder.points) > 0:
            self.stage_progress.emit("Unclassified...", 75)
            layers.append(self._make_pcd_layer(
                remainder, "unclassified", "Unclassified Points",
                visible=False, color=[0.4, 0.4, 0.4], opacity=0.3))

        # Decimated cloud
        self.stage_progress.emit("Decimated cloud...", 80)
        layers.append(self._make_pcd_layer(
            pcd_decimated, "decimated", "Decimated Cloud",
            visible=False, opacity=0.8))

        self.stage_progress.emit("Done", 100)
        self._log(f"Created {len(layers)} layers")
        return layers

    def _make_raw_layer(self, raw_arrays):
        layer_def = {
            "id": "raw_scan", "name": "Raw E57 Scan",
            "type": "pointcloud", "visible": False,
            "color": None, "opacity": 0.5,
        }
        layer = LayerData(layer_def, "")
        layer.points = raw_arrays["points"]
        # Use compact uint8 colors if available (saves ~3× memory)
        if "colors_u8" in raw_arrays:
            layer.colors_u8 = raw_arrays["colors_u8"]
            layer.colors = None  # will be generated on demand
        elif "colors" in raw_arrays:
            layer.colors = raw_arrays["colors"]
        layer.point_count = len(layer.points)
        layer.loaded = True
        mem_mb = layer.points.nbytes / 1024**2
        if hasattr(layer, 'colors_u8') and layer.colors_u8 is not None:
            mem_mb += layer.colors_u8.nbytes / 1024**2
        elif layer.colors is not None:
            mem_mb += layer.colors.nbytes / 1024**2
        self._log(f"Layer 'Raw E57 Scan': {layer.point_count:,} points "
                  f"({mem_mb:.0f} MB)")
        return layer

    def _make_pcd_layer(self, pcd, layer_id, name, visible=True,
                        color=None, opacity=1.0):
        layer_def = {
            "id": layer_id, "name": name, "type": "pointcloud",
            "visible": visible, "color": color, "opacity": opacity,
        }
        layer = LayerData(layer_def, "")
        layer.points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            # Store as compact uint8 to save 3× memory
            colors_f64 = np.asarray(pcd.colors)  # O3D float64
            layer.colors_u8 = (colors_f64 * 255).clip(0, 255).astype(np.uint8)
            layer.colors = None  # generated on demand from colors_u8
            self._log(f"Layer '{name}': {len(layer.points):,} points "
                      f"(colors: uint8 compact)")
        else:
            self._log(f"Layer '{name}': {len(layer.points):,} points (no colors)")
        if pcd.has_normals():
            layer.normals = np.asarray(pcd.normals, dtype=np.float32)
        layer.point_count = len(layer.points)
        layer.loaded = True
        return layer

    # ------------------------------------------------------------------
    # Panorama extraction (optional: libe57 + Pillow)
    # ------------------------------------------------------------------

    def _extract_panoramas(self, path: str) -> list:
        """Extract panorama images using the modular panorama subpackage.

        Delegates to rendering.panorama.extractor — see that module for
        the full extraction logic and configurable constants.
        """
        try:
            from locul3d.rendering.panorama import PanoramaManager
        except ImportError:
            self._log("Skipping panoramas (panorama module not available)")
            return []

        stations = PanoramaManager.extract(path, log_fn=self._log)
        if not stations:
            return []

        # Convert station dicts → LayerData objects
        layers = []
        for station in stations:
            layer_def = {
                "id": station["id"],
                "name": station["name"],
                "type": "panorama",
                "visible": True,
                "color": list(station["color"]),
                "opacity": station["opacity"],
            }
            layer = LayerData(layer_def, "")
            layer.pano_position = station["position"]
            layer.pano_rotation = station.get("rotation")
            layer.pano_type = station["type"]
            layer.pano_equirect = station.get("equirect")
            layer.pano_faces = station.get("faces")
            layer.pano_jpeg_bytes = station.get("jpeg_bytes")
            layer.pano_image_size = station.get("image_size")
            layer.pano_face_bytes = station.get("face_bytes")
            layer.point_count = 1  # marker
            layer.loaded = True
            layers.append(layer)

        return layers

    def _align_panorama_layers(self, layers):
        """Apply alignment transform (R, center, z_shift) to panorama layers.

        This ensures panorama positions and quaternions match the
        aligned/decimated point cloud coordinate frame.
        """
        R = self._align_R
        center = self._align_center
        z_shift = self._align_z_shift

        # Check if alignment is identity (no rotation needed)
        if np.allclose(R, np.eye(3)) and abs(z_shift) < 1e-6:
            return

        # Convert rotation matrix to quaternion for quaternion composition
        R_quat = self._rotation_matrix_to_quat(R)

        count = 0
        for layer in layers:
            if layer.layer_type != "panorama":
                continue

            # Transform position: R @ (pos - center) + center - [0,0,z_shift]
            if layer.pano_position is not None:
                pos = np.array(layer.pano_position, dtype=np.float64)
                pos_rotated = R @ (pos - center) + center
                pos_rotated[2] -= z_shift
                layer.pano_position = pos_rotated.tolist()

            # Transform quaternion: R_quat * Q_original
            if layer.pano_rotation is not None:
                qw, qx, qy, qz = layer.pano_rotation
                # Multiply R_quat * original_quat
                rw, rx, ry, rz = R_quat
                nw = rw*qw - rx*qx - ry*qy - rz*qz
                nx = rw*qx + rx*qw + ry*qz - rz*qy
                ny = rw*qy - rx*qz + ry*qw + rz*qx
                nz = rw*qz + rx*qy - ry*qx + rz*qw
                layer.pano_rotation = (nw, nx, ny, nz)

            count += 1

        if count > 0:
            self._log(f"Applied alignment transform to {count} panorama(s)")

    @staticmethod
    def _rotation_matrix_to_quat(R):
        """Convert a 3×3 rotation matrix to quaternion (w, x, y, z)."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return (w, x, y, z)


# =============================================================================
# E57ProgressDialog
# =============================================================================

class E57ProgressDialog(QDialog):
    """Modal dialog with stage-by-stage progress for E57 import."""

    def __init__(self, e57_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import E57")
        self.setMinimumSize(560, 480)
        self.setModal(True)
        self._result: Optional[E57ImportResult] = None
        self._worker: Optional[E57ImportWorker] = None
        self._start_time = 0.0
        self._setup_ui(e57_path)

    def _setup_ui(self, e57_path: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel(f"Importing: {Path(e57_path).name}")
        header.setStyleSheet(
            f"font-size: 14px; font-weight: bold; "
            f"color: {COLORS.get('text', '#eee')};")
        layout.addWidget(header)

        # Stage indicators
        self._stage_widgets = {}
        stages_frame = QFrame()
        stages_frame.setStyleSheet(
            f"QFrame {{ background: {COLORS.get('input_bg', '#222')}; "
            f"border-radius: 6px; padding: 8px; }}")
        stages_layout = QVBoxLayout(stages_frame)
        stages_layout.setSpacing(6)

        for stage_name, stage_desc in _E57_STAGES:
            row = QHBoxLayout()
            row.setSpacing(8)

            icon_label = QLabel("  ")
            icon_label.setFixedWidth(20)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet(
                f"color: {COLORS.get('hover', '#555')}; font-size: 12px;")
            row.addWidget(icon_label)

            name_label = QLabel(stage_name)
            name_label.setFixedWidth(120)
            name_label.setStyleSheet(
                f"color: {COLORS.get('text_muted', '#888')}; font-size: 12px;")
            row.addWidget(name_label)

            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFixedHeight(14)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {COLORS.get('hover', '#555')}; "
                f"border-radius: 3px; }}")
            row.addWidget(pbar, stretch=1)

            stages_layout.addLayout(row)
            self._stage_widgets[stage_name] = (icon_label, name_label, pbar)

        layout.addWidget(stages_frame)

        self._time_label = QLabel("Elapsed: 0.0s")
        self._time_label.setStyleSheet(
            f"color: {COLORS.get('text_muted', '#888')}; font-size: 11px;")
        layout.addWidget(self._time_label)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(180)
        self._log_view.setStyleSheet(
            f"QTextEdit {{ background: {COLORS.get('bg', '#111')}; "
            f"color: {COLORS.get('text_secondary', '#aaa')}; "
            f"font-family: monospace; font-size: 11px; "
            f"border: 1px solid {COLORS.get('border', '#444')}; "
            f"border-radius: 4px; padding: 4px; }}")
        layout.addWidget(self._log_view)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(100)

        self._current_stage: Optional[str] = None
        self._completed_stages: set = set()

    def start(self, worker: E57ImportWorker):
        self._worker = worker
        self._start_time = time.time()

        worker.stage_started.connect(self._on_stage_started)
        worker.stage_progress.connect(self._on_stage_progress)
        worker.log_message.connect(self._on_log)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.finished_err.connect(self._on_finished_err)

        self._on_stage_started("Ingestion",
                               "Opening E57 file (this may take a moment for large files)...")
        QTimer.singleShot(50, worker.start)

    def get_result(self) -> Optional[E57ImportResult]:
        return self._result

    def _on_stage_started(self, stage_name: str, detail: str):
        if self._current_stage and self._current_stage in self._stage_widgets:
            self._mark_stage_done(self._current_stage)

        self._current_stage = stage_name
        if stage_name in self._stage_widgets:
            icon, name, pbar = self._stage_widgets[stage_name]
            icon.setText(">>")
            icon.setStyleSheet(
                f"color: {COLORS.get('accent', '#36f')}; "
                f"font-size: 11px; font-weight: bold;")
            name.setStyleSheet(
                f"color: {COLORS.get('text', '#eee')}; "
                f"font-size: 12px; font-weight: bold;")
            pbar.setValue(0)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {COLORS.get('accent', '#36f')}; "
                f"border-radius: 3px; }}")

        self._on_log(f"--- {stage_name}: {detail}")

    def _on_stage_progress(self, detail: str, pct: int):
        if self._current_stage and self._current_stage in self._stage_widgets:
            _, _, pbar = self._stage_widgets[self._current_stage]
            pbar.setValue(pct)

    def _mark_stage_done(self, stage_name: str):
        if stage_name in self._stage_widgets:
            icon, name, pbar = self._stage_widgets[stage_name]
            icon.setText("OK")
            icon.setStyleSheet("color: #40c040; font-size: 11px; font-weight: bold;")
            name.setStyleSheet("color: #80c080; font-size: 12px;")
            pbar.setValue(100)
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: #40c040; border-radius: 3px; }}")
            self._completed_stages.add(stage_name)

    def _on_log(self, msg: str):
        self._log_view.append(msg)
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_view.setTextCursor(cursor)

    def _on_finished_ok(self, result):
        if self._current_stage:
            self._mark_stage_done(self._current_stage)
        elapsed = time.time() - self._start_time
        self._time_label.setText(f"Completed in {elapsed:.1f}s")
        self._on_log(f"\nImport complete ({elapsed:.1f}s)")
        self._result = result
        self._timer.stop()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)
        QTimer.singleShot(600, self.accept)

    def _on_finished_err(self, error: str):
        elapsed = time.time() - self._start_time
        self._time_label.setText(f"Failed after {elapsed:.1f}s")
        if self._current_stage and self._current_stage in self._stage_widgets:
            icon, name, pbar = self._stage_widgets[self._current_stage]
            icon.setText("!!")
            icon.setStyleSheet("color: #ff4040; font-size: 11px; font-weight: bold;")
            name.setStyleSheet("color: #ff6060; font-size: 12px;")
            pbar.setStyleSheet(
                f"QProgressBar {{ background: {COLORS.get('border', '#444')}; "
                f"border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: #ff4040; border-radius: 3px; }}")
        self._on_log(f"\nERROR: {error}")
        self._timer.stop()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._on_log("Cancelling...")
            self._worker.cancel()
            self._worker.wait(3000)
        self.reject()

    def _update_elapsed(self):
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            self._time_label.setText(f"Elapsed: {elapsed:.1f}s")


# =============================================================================
# Plugin interface (simple wrapper)
# =============================================================================

class E57Importer:
    """E57 importer for the plugin system — wraps E57ImportWorker."""

    @property
    def name(self) -> str:
        return "E57 Importer"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def file_extensions(self) -> list:
        return ['.e57']

    @staticmethod
    def is_available() -> bool:
        return HAS_PYE57 and HAS_O3D

    @staticmethod
    def missing_deps_message() -> str:
        """Return a human-readable message listing missing dependencies."""
        missing = []
        if not HAS_PYE57:
            missing.append(f"pye57 ({_PYE57_ERR})" if _PYE57_ERR else "pye57")
        if not HAS_O3D:
            missing.append(f"open3d ({_O3D_ERR})" if _O3D_ERR else "open3d")
        if missing:
            return "E57 import requires: " + ", ".join(missing)
        return ""

    def can_import(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == '.e57'

    def create_worker(self, file_path: str, parent=None) -> Optional[E57ImportWorker]:
        if not self.is_available():
            return None
        return E57ImportWorker(file_path, parent)

    def create_dialog(self, file_path: str, parent=None) -> Optional[E57ProgressDialog]:
        if not self.is_available():
            return None
        return E57ProgressDialog(file_path, parent)
