"""E57 file importer -- full processing pipeline (first-class citizen).

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
# RAM.  348M pts × 24 bytes ≈ 7.8 GB -- holding that in memory alongside
# the aligned + decimated copies is wasteful for interactive use.
E57_RAW_LAYER_POINT_THRESHOLD = 50_000_000  # 50M -- skip raw layer above this

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
        # E57 pose from file (set by _stage_ingest)
        self._e57_pose_rotation = None   # quaternion [w, x, y, z]
        self._e57_pose_translation = None  # [x, y, z]
        self._e57_pose_R = np.eye(3)     # rotation matrix from quaternion
        self._e57_pose_t = np.zeros(3)   # translation vector

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
                # print(f"[Worker] Pipeline finished @ {time.time():.3f}. Emitting signal...")
                self.finished_ok.emit(result)
                # print(f"[Worker] Signal emitted @ {time.time():.3f}")
        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")

    def _run_pipeline(self) -> E57ImportResult:
        import gc

        result = E57ImportResult()
        result.base_dir = str(Path(self._path).parent)
        stats = {}
        t_total = time.time()

        # Stage 1: Ingest -- returns raw numpy arrays (float32 + uint8)
        self.stage_started.emit("Ingestion", "Reading E57 file...")
        t0 = time.time()
        xyz, colors_u8, meta = self._stage_ingest()
        stats["ingest_time"] = time.time() - t0
        stats["points_after_crop"] = len(xyz)
        if self._cancelled:
            return result

        # Keep raw arrays for the raw scan layer
        raw_arrays = {"points": xyz, "colors_u8": colors_u8}

        # Stride-subsample for the O3D processing pipeline (RANSAC align,
        # voxel decimate).  2M points is plenty for plane fitting; using
        # fewer points makes voxel + alignment much faster.
        target_pts = 2_000_000
        stride = max(1, len(xyz) // target_pts)
        # np.ascontiguousarray converts from F-order column-major to
        # C-order row-major — critical for O3D which needs row-contiguous
        # float64 input (F-order .astype(float64) is 44× slower).
        ds_xyz = np.ascontiguousarray(xyz[::stride])
        ds_colors = (np.ascontiguousarray(colors_u8[::stride])
                     if colors_u8 is not None else None)
        n_ds = len(ds_xyz)
        self._log(f"Stride subsample: {len(xyz):,} → {n_ds:,} pts "
                  f"(stride={stride})")

        # Build O3D cloud from the small C-contiguous subset
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

        # Stage 2: Filter (voxel only — SOR is skipped because it costs
        # ~7s for negligible benefit on stride-subsampled data; RANSAC
        # alignment is already robust to outliers)
        self.stage_started.emit("Filtering", "Voxel downsampling...")
        t0 = time.time()
        pcd_filtered = pcd.voxel_down_sample(
            voxel_size=E57_VOXEL_SIZE_FILTER)
        n_filt = len(pcd_filtered.points)
        self._log(f"Voxel filter {E57_VOXEL_SIZE_FILTER*1000:.0f}mm: "
                  f"{len(pcd.points):,} → {n_filt:,}")
        stats["filter_time"] = time.time() - t0
        stats["points_after_filter"] = n_filt
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
        t_pano = time.time()
        self.stage_started.emit("Panoramas", "Extracting images...")
        pano_layers = self._extract_panoramas(self._path)
        self._align_panorama_layers(pano_layers)
        layers.extend(pano_layers)
        self._log(f"Panorama extraction: {time.time() - t_pano:.2f}s")

        stats["total_time"] = time.time() - t_total
        stats["surfaces"] = []

        # Return freed heap pages to the OS.  Without this, peak RSS
        # stays ~2× higher than live data because glibc/macOS hold
        # freed mmap'd regions in the process address space.
        gc.collect()
        try:
            import ctypes
            if sys.platform == 'darwin':
                ctypes.CDLL('libSystem.B.dylib').malloc_zone_pressure_relief(
                    None, 0)
            elif sys.platform.startswith('linux'):
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            # Windows: HeapCompact not needed — CRT returns large blocks
        except (OSError, AttributeError):
            pass

        result.layers = layers
        result.metadata = meta
        result.stats = stats
        return result

    # ------------------------------------------------------------------
    # Stage 1: Ingestion -- fast binary E57 reader + libe57 fallback
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

        # Read header & metadata via pye57 (fast -- just XML)
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

        # Store pose for later application
        self._e57_pose_rotation = pose_meta.get("rotation_quaternion")
        self._e57_pose_translation = pose_meta.get("translation")
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
            xyz, colors_u8 = self._fast_binary_read(path, n, has_rgb, header.point_fields)
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
            # Optimized: Squared-norm check avoids 350M square roots (slow)
            r2 = E57_CROP_RADIUS ** 2
            dx = xyz[:, 0] - scanner_pos[0]
            dy = xyz[:, 1] - scanner_pos[1]
            dz = xyz[:, 2] - scanner_pos[2]
            crop_mask = (dx**2 + dy**2 + dz**2) < r2
            n_cropped = int(crop_mask.sum())
            self._log(f"Cropping to {E57_CROP_RADIUS}m: "
                      f"{n_cropped:,} pts ({100 * n_cropped / len(xyz):.1f}%)")
            xyz = xyz[crop_mask] - scanner_pos
            if colors_u8 is not None:
                colors_u8 = colors_u8[crop_mask]
        else:
            n_cropped = len(xyz)

        # Fast subsampled AABB for metadata (100k points is plenty for BB/stats)
        bb_sample = xyz[::max(1, len(xyz) // 100_000)]
        bb_min = bb_sample.min(axis=0)
        bb_max = bb_sample.max(axis=0)
        bb_size = bb_max - bb_min
        self._log(f"Bounding box: [{bb_size[0]:.1f} x "
                  f"{bb_size[1]:.1f} x {bb_size[2]:.1f}] m")
        
        # Cache for direct layer hand-off later
        self._ingest_aabb_min = bb_min
        self._ingest_aabb_max = bb_max

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

    @staticmethod
    def _parse_prototype(xml_bytes):
        """Parse the E57 XML prototype of the first scan's points.

        Returns dict in prototype (= bytestream) order:
            field_name → {'type': 'float'|'scaledInteger'|'integer',
                          'bits': int,      # bitPackCodec width per record
                          'minimum': int, 'maximum': int,
                          'scale': float, 'offset': float}
        """
        import re
        xml_str = xml_bytes.decode('utf-8', errors='replace')

        fields = {}
        # Use the prototype of data3D[0]/points -- some files have other
        # CompressedVectors (e.g. pointGroupingSchemes) earlier in the XML
        points = re.search(r'<points\b[^>]*type="CompressedVector"', xml_str)
        proto_match = re.compile(r'<prototype[^>]*>(.*?)</prototype>',
                                 re.DOTALL).search(
            xml_str, points.start() if points else 0)
        if not proto_match:
            return fields

        proto = proto_match.group(1)
        # Each field is a tag: <fieldName attr1="val" .../> (or <...></...>),
        # extension fields are namespaced (e.g. <nor:normalX .../>):
        #   <cartesianX type="ScaledInteger" minimum="..." maximum="..." scale="..." offset="..."/>
        #   <cartesianX type="Float" precision="single" .../>
        #   <colorRed type="Integer" minimum="0" maximum="255"/>
        for m in re.finditer(r'<([\w:.-]+)\s+([^>]*?)/?>', proto):
            name = m.group(1)
            attrs = m.group(2)

            def _attr(key, default=None):
                am = re.search(rf'\b{key}="([^"]*)"', attrs)
                return am.group(1) if am else default

            ftype = (_attr('type') or '').lower()
            if ftype in ('scaledinteger', 'integer'):
                mn = int(float(_attr('minimum', '0')))
                mx = int(float(_attr('maximum', '0')))
                fields[name] = {
                    'type': 'scaledInteger' if ftype == 'scaledinteger' else 'integer',
                    'bits': (mx - mn).bit_length(),
                    'minimum': mn, 'maximum': mx,
                    'scale': float(_attr('scale', '1')),
                    'offset': float(_attr('offset', '0')),
                }
            elif ftype == 'float':
                prec = (_attr('precision') or 'double').lower()
                fields[name] = {'type': 'float',
                                'bits': 32 if prec == 'single' else 64,
                                'minimum': 0, 'maximum': 0,
                                'scale': 1.0, 'offset': 0.0}
        return fields

    @staticmethod
    def _unpack_bits(buf, count, bits):
        """Decode `count` unsigned values of `bits` width from a bitPackCodec
        byte stream (little-endian, LSB-first, records may span bytes)."""
        if bits == 0:
            return np.zeros(count, dtype=np.uint64)
        if bits in (8, 16, 32, 64):
            return buf[:count * bits // 8].view(f'<u{bits // 8}')
        if bits % 8 == 0:
            # Whole bytes (24/40/48/56-bit): combine byte columns
            nb = bits // 8
            cols = buf[:count * nb].reshape(count, nb)
            out = cols[:, 0].astype(np.uint64)
            for j in range(1, nb):
                out |= cols[:, j].astype(np.uint64) << np.uint64(8 * j)
            return out
        if bits > 56:
            raise ValueError(f"Unsupported bitPack width: {bits}")

        out = np.empty(count, dtype=np.uint64)
        mask = np.uint64((1 << bits) - 1)
        span = (bits + 7 + 7) // 8          # bytes touched by one record
        padded = np.concatenate([buf, np.zeros(8, dtype=np.uint8)])
        BLOCK = 1 << 22                     # bound temporary memory
        for s in range(0, count, BLOCK):
            e = min(s + BLOCK, count)
            bitpos = np.arange(s, e, dtype=np.uint64) * np.uint64(bits)
            byte = (bitpos >> np.uint64(3)).astype(np.int64)
            acc = padded[byte].astype(np.uint64)
            for j in range(1, span):
                acc |= padded[byte + j].astype(np.uint64) << np.uint64(8 * j)
            out[s:e] = (acc >> (bitpos & np.uint64(7))) & mask
        return out

    def _fast_binary_read(self, path, n, has_rgb, field_names):
        """Vectorized E57 binary reader -- bypasses libE57Format codec.

        Adapts to the actual E57 data structure by parsing the XML
        prototype to discover field encodings (bitPackCodec):
          - Float (single/double)
          - ScaledInteger: raw unsigned offset from minimum,
            value = (raw + minimum) * scale + offset
          - Integer: value = raw + minimum
        Any bit width is supported; 8/16/32/64-bit fields are copied
        directly, others are bit-unpacked.

        E57 page layout: every 1024 physical bytes, the last 4 are a CRC32C
        checksum.  The first 1020 bytes carry payload ("logical" bytes).
        Data packets are laid out contiguously in the logical stream.

        Packets are walked by their own headers, so packets of different
        sizes (e.g. incrementally written files) and a short last packet
        are handled.  Runs of identically laid out packets (the common case)
        are extracted with one strided 2D copy per field instead of a
        per-packet loop.
        """
        import gc, struct, re

        PAGE_SIZE = 1024
        DATA_PER_PAGE = 1020

        def _phys_to_log(p):
            return (p // PAGE_SIZE) * DATA_PER_PAGE + (p % PAGE_SIZE)

        def _log_to_phys(l):
            return (l // DATA_PER_PAGE) * PAGE_SIZE + (l % DATA_PER_PAGE)

        def _strip_crc_region(raw, phys_start, n_log_bytes):
            """CRC-strip a small region (headers / XML) into logical bytes."""
            sp = phys_start // PAGE_SIZE
            ep = _log_to_phys(_phys_to_log(phys_start) + n_log_bytes) // PAGE_SIZE + 1
            np_ = ep - sp
            r = raw[sp * PAGE_SIZE : ep * PAGE_SIZE].reshape(np_, PAGE_SIZE)
            lg = np.ascontiguousarray(r[:, :DATA_PER_PAGE]).ravel()
            lo = _phys_to_log(phys_start) - sp * DATA_PER_PAGE
            return lg[lo : lo + n_log_bytes]

        # -- 1. Read entire file (single sequential I/O) ------------------
        t0 = time.time()
        raw_arr = np.fromfile(path, dtype=np.uint8)
        t_read = time.time() - t0
        file_mb = len(raw_arr) / 1024**2
        self._log(f"File read: {t_read:.2f}s ({file_mb / max(t_read, 1e-6):.0f} MB/s)")
        n_pages = len(raw_arr) // PAGE_SIZE

        # -- 2. Parse E57 header → XML → CV offset + field types ----------
        raw_hdr = bytes(raw_arr[:48])
        xml_phys_off = struct.unpack_from('<Q', raw_hdr, 24)[0]
        xml_phys_len = struct.unpack_from('<Q', raw_hdr, 32)[0]

        xml_bytes = _strip_crc_region(raw_arr, xml_phys_off,
                                      xml_phys_len).tobytes()
        m = re.search(rb'<points\b[^>]*\bfileOffset="(\d+)"', xml_bytes)
        cv_phys_off = int(m.group(1)) if m else 48

        # Parse field encodings from prototype
        proto = self._parse_prototype(xml_bytes)
        stream_names = list(proto)          # bytestream order
        xyz_names = ('cartesianX', 'cartesianY', 'cartesianZ')
        rgb_names = ('colorRed', 'colorGreen', 'colorBlue') if has_rgb else ()
        if any(f not in proto for f in xyz_names + rgb_names):
            raise ValueError("Required fields not found in prototype")
        self._log("Fields: " + ", ".join(
            f"{f}={proto[f]['type']}/{proto[f]['bits']}b"
            for f in xyz_names + rgb_names))

        # Section header: id, reserved, sectionLength, dataPhysicalOffset
        cv_log = _phys_to_log(cv_phys_off)
        sec = _strip_crc_region(raw_arr, cv_phys_off, 32).tobytes()
        section_len, data_phys_off = struct.unpack_from('<QQ', sec, 8)
        section_end_log = cv_log + section_len
        first_pkt_log = _phys_to_log(data_phys_off)
        if not (cv_log + 32 <= first_pkt_log < section_end_log):
            first_pkt_log = cv_log + 32
        self._log(f"CV phys={cv_phys_off}, first data pkt at "
                  f"log={first_pkt_log}, section end log={section_end_log}")

        # -- 3. Destination byte buffers per field ------------------------
        # Output arrays (always float32 / uint8, F-order for column access).
        # Byte-aligned fields matching the output type are written straight
        # into the output columns; others go to a byte buffer and are decoded.
        xyz = np.empty((n, 3), dtype=np.float32, order='F')
        colors_u8 = (np.empty((n, 3), dtype=np.uint8, order='F')
                     if has_rgb else None)

        targets = []   # [stream index, dest uint8 buffer, required bytes, filled]
        decode = {}    # field → dest buffer needing decoding
        for col, f in enumerate(xyz_names):
            p = proto[f]
            if p['type'] == 'float' and p['bits'] == 32:
                dest = xyz[:, col].view(np.uint8)
            else:
                dest = np.empty((n * p['bits'] + 7) // 8, dtype=np.uint8)
                decode[f] = dest
            targets.append([stream_names.index(f), dest, len(dest), 0])
        for col, f in enumerate(rgb_names):
            p = proto[f]
            if p['type'] == 'integer' and p['bits'] == 8 and p['minimum'] == 0:
                dest = colors_u8[:, col].view(np.uint8)
            else:
                dest = np.empty((n * p['bits'] + 7) // 8, dtype=np.uint8)
                decode[f] = dest
            targets.append([stream_names.index(f), dest, len(dest), 0])

        # -- 4. Chunked CRC strip + packet walk ---------------------------
        t0 = time.time()
        CHUNK_BYTES = 256 * 1024**2         # logical bytes per chunk
        n_packets = 0
        n_runs = 0
        pos = first_pkt_log

        def _all_filled():
            return all(t[3] >= t[2] for t in targets)

        while pos < section_end_log and not _all_filled():
            c_end = min(pos + CHUNK_BYTES, section_end_log)
            c_start_page = pos // DATA_PER_PAGE
            c_end_page = min((c_end - 1) // DATA_PER_PAGE + 1, n_pages)
            c_pages = raw_arr[c_start_page * PAGE_SIZE :
                              c_end_page * PAGE_SIZE].reshape(-1, PAGE_SIZE)
            c_logical = np.ascontiguousarray(
                c_pages[:, :DATA_PER_PAGE]).ravel()
            d = c_logical[pos - c_start_page * DATA_PER_PAGE :
                          c_end - c_start_page * DATA_PER_PAGE]
            d_len = len(d)

            o = 0
            while o + 6 <= d_len:
                ptype = int(d[o])
                plen = (int(d[o + 2]) | (int(d[o + 3]) << 8)) + 1
                if o + plen > d_len:
                    break                   # continues in the next chunk
                if ptype != 1:              # index / empty packet
                    o += plen
                    continue

                bcount = int(d[o + 4]) | (int(d[o + 5]) << 8)
                hlen = 6 + 2 * bcount
                hdr = d[o : o + hlen]
                lens = struct.unpack_from(f'<{bcount}H', hdr.tobytes(), 6)
                starts = [hlen]
                for ln in lens[:-1]:
                    starts.append(starts[-1] + ln)

                # Count consecutive packets with an identical header
                # (same length and stream layout), growing the window
                k_max = (d_len - o) // plen
                run = d[o : o + k_max * plen].reshape(k_max, plen)
                k, blk = 1, 16
                while k < k_max:
                    hi = min(k_max, k + blk)
                    same = (run[k:hi, :hlen] == hdr).all(axis=1)
                    if not same.all():
                        k += int(np.argmin(same))
                        break
                    k = hi
                    blk *= 4

                for t in targets:
                    si, dest, need, filled = t
                    ln = lens[si]
                    if filled >= need or ln == 0:
                        continue
                    st = starts[si]
                    kk = min(k, (need - filled) // ln)
                    if kk:
                        dest[filled : filled + kk * ln].reshape(kk, ln)[...] = \
                            run[:kk, st : st + ln]
                        filled += kk * ln
                    if kk < k and filled < need:
                        # Final partial packet for this stream (padding beyond)
                        rem = min(need - filled, ln)
                        dest[filled : filled + rem] = run[kk, st : st + rem]
                        filled += rem
                    t[3] = filled

                o += k * plen
                n_packets += k
                n_runs += 1
                if _all_filled():
                    break

            if o == 0:
                raise ValueError(f"Cannot parse data packet at log={pos}")
            pos += o
            del c_pages, c_logical, d

        if not _all_filled():
            raise ValueError("Data packets end before all records were read")
        self._log(f"Packets: {n_packets} in {n_runs} uniform runs")

        # -- 5. Decode non-direct fields -----------------------------------
        for col, f in enumerate(xyz_names):
            if f not in decode:
                continue
            p = proto[f]
            if p['type'] == 'float':
                xyz[:, col] = decode[f].view('<f8')[:n]
            else:
                # E57 stores unsigned offset from minimum:
                #   real = (stored + minimum) * scale + offset
                raw = self._unpack_bits(decode[f], n, p['bits'])
                xyz[:, col] = ((raw.astype(np.float64) + np.float64(p['minimum']))
                               * np.float64(p['scale'])
                               + np.float64(p['offset'])).astype(np.float32)
                del raw
            self._log(f"{f}: {p['type']} {p['bits']}-bit decoded")
        for col, f in enumerate(rgb_names):
            if f not in decode:
                continue
            p = proto[f]
            vals = self._unpack_bits(decode[f], n, p['bits']).astype(np.int64) + p['minimum']
            if p['minimum'] < 0 or p['maximum'] > 255:
                # Normalize wider color ranges to 0..255
                vals = (vals - p['minimum']) * 255 // max(1, p['maximum'] - p['minimum'])
            colors_u8[:, col] = vals.astype(np.uint8)
        decode.clear()

        t_parse = time.time() - t0
        self._log(f"Chunked parse: {t_parse:.2f}s "
                  f"({n / max(t_parse, 1e-6) / 1e6:.1f} Mpts/s)")

        # Integrity check: reject if data is garbage
        sample = xyz[::max(1, n // 10000)]
        if np.isnan(sample).any() or np.isinf(sample).any():
            del raw_arr
            gc.collect()
            raise ValueError("Fast read produced NaN/Inf — falling back")

        del raw_arr
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
        # Apply E57 pose if available (transform from scanner to world coords)
        if self._e57_pose_rotation is not None or self._e57_pose_translation is not None:
            self._log("Applying E57 scan pose...")

            # Convert quaternion to rotation matrix
            if self._e57_pose_rotation is not None:
                w, x, y, z = self._e57_pose_rotation
                # Quaternion to rotation matrix
                R_pose = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
                ])
                self._e57_pose_R = R_pose  # Store for raw points
                pcd.rotate(R_pose, center=[0, 0, 0])
                self._log(f"  Rotation applied (quat: w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})")

            # Apply translation
            if self._e57_pose_translation is not None:
                t = np.array(self._e57_pose_translation)
                self._e57_pose_t = t  # Store for raw points
                pcd.translate(t)
                self._log(f"  Translation applied: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

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

    def _build_layers(self, raw_arrays, pcd_aligned, pcd_decimated,
                      surfaces, surface_clouds, measurements):
        import gc, time

        layers = []
        raw_pts = raw_arrays["points"]
        n_pts = len(raw_pts)
        
        # ------------------------------------------------------------------
        # Part 1a: Apply E57 Pose (scanner to world transform)
        # ------------------------------------------------------------------
        if not np.allclose(self._e57_pose_R, np.eye(3)) or np.any(self._e57_pose_t != 0):
            self.stage_progress.emit("Applying E57 pose to raw scan...", 2)
            t_pose = time.time()
            R_pose = self._e57_pose_R.astype(np.float32)
            t_pose_vec = self._e57_pose_t.astype(np.float32)

            # Apply rotation then translation: pts @ R_pose.T + t_pose
            CHUNK = 10_000_000
            for s in range(0, n_pts, CHUNK):
                e = min(s + CHUNK, n_pts)
                chunk = raw_pts[s:e].copy()
                raw_pts[s:e] = chunk @ R_pose.T + t_pose_vec

            self._log(f"E57 pose applied to raw: {time.time() - t_pose:.2f}s")

        # ------------------------------------------------------------------
        # Part 1b: Direct Alignment (Single-thread BLAS/AMX Native)
        # ------------------------------------------------------------------
        self.stage_progress.emit("Aligning raw scan...", 5)
        t_align = time.time()

        # Extract alignment params
        R = self._align_R.astype(np.float32)
        center = self._align_center.astype(np.float32)
        z_shift = float(self._align_z_shift)

        # Transform: (pts - center) @ R.T + shift_back
        # Simplified: pts @ R.T + (shift_back - center @ R.T)
        shift_back_vec = center - np.array([0, 0, z_shift], dtype=np.float32)
        total_offset = shift_back_vec - (center @ R.T)

        # Chunked column-wise transform: process 10M points at a time
        # so temporaries use ~160 MB instead of 5.6 GB.  F-order columns
        # are contiguous, so chunk slices (raw_pts[s:e, j]) are contiguous.
        RT = R.T.astype(np.float32)
        CHUNK = 10_000_000
        n_pts = len(raw_pts)
        for s in range(0, n_pts, CHUNK):
            e = min(s + CHUNK, n_pts)
            ox = raw_pts[s:e, 0].copy()
            oy = raw_pts[s:e, 1].copy()
            oz = raw_pts[s:e, 2].copy()
            tmp = np.empty(e - s, dtype=np.float32)
            for j in range(3):
                np.multiply(ox, RT[0, j], out=raw_pts[s:e, j])
                np.multiply(oy, RT[1, j], out=tmp)
                np.add(raw_pts[s:e, j], tmp, out=raw_pts[s:e, j])
                np.multiply(oz, RT[2, j], out=tmp)
                np.add(raw_pts[s:e, j], tmp, out=raw_pts[s:e, j])
                raw_pts[s:e, j] += total_offset[j]
        self._log(f"Raw scan aligned: {time.time() - t_align:.2f}s (column-wise)")

        # Transform ingested AABB to world space for immediate viewport fit
        lmin_w = (self._ingest_aabb_min @ R.T) + total_offset
        lmax_w = (self._ingest_aabb_max @ R.T) + total_offset
        cur_min = np.minimum(lmin_w, lmax_w)
        cur_max = np.maximum(lmin_w, lmax_w)

        # ------------------------------------------------------------------
        # Part 2: Layer Construction
        # ------------------------------------------------------------------

        # 1. Raw Scan Layer
        raw_layer = self._make_raw_layer(raw_arrays)
        raw_layer._aabb_min = cur_min
        raw_layer._aabb_max = cur_max
        layers.append(raw_layer)

        # 2. Mid-Resolution Cloud
        self.stage_progress.emit("Mid-res cloud...", 10)
        t_mid = time.time()
        stride = max(1, n_pts // 5_000_000)

        # Contiguous copies (not views) so mid-res doesn't pin the
        # full raw array in memory if raw_scan layer is later freed.
        mid_res_layer = self._make_raw_layer({
            "points": raw_pts[::stride].copy(),
            "colors_u8": (raw_arrays["colors_u8"][::stride].copy()
                          if raw_arrays.get("colors_u8") is not None
                          else None),
        })
        mid_res_layer.id = "midres"
        mid_res_layer.name = "Mid-Resolution Point Cloud"
        mid_res_layer.visible = True
        mid_res_layer.opacity = 0.6
        mid_res_layer._aabb_min = cur_min
        mid_res_layer._aabb_max = cur_max
        layers.append(mid_res_layer)
        self._log(f"Mid-res layer: {time.time() - t_mid:.4f}s (stride view)")
        self._log(f"Mid-res layer build: {time.time() - t_mid:.4f}s (stride view, 0 bytes copied)")

        # 3. Surface Layers
        t_surf = time.time()
        for i, surf in enumerate(surfaces):
            layers.append(self._make_pcd_layer(
                surface_clouds[i], f"surface_{i}",
                f"Surface {i}: {surf['orientation']}",
                visible=False, color=surf["color"], opacity=1.0))
        self._log(f"Surface layers ({len(surfaces)}): {time.time() - t_surf:.4f}s")

        # 4. Decimated Cloud (Open3D result)
        self.stage_progress.emit("Decimated cloud...", 80)
        t_dec = time.time()
        layers.append(self._make_pcd_layer(
            pcd_decimated, "decimated", "Decimated Cloud",
            visible=False, opacity=0.8))
        self._log(f"Decimated layer: {time.time() - t_dec:.4f}s")
        self._log(f"Decimated layer build: {time.time() - t_dec:.2f}s")

        self.stage_progress.emit("Done", 100)
        self._log(f"Layer building complete")
        gc.collect()
        return layers

    def _make_raw_layer(self, raw_arrays):
        layer_def = {
            "id": "raw_scan", "name": "Raw E57 Scan",
            "type": "pointcloud", "visible": False,
            "color": None, "opacity": 0.5,
        }
        layer = LayerData(layer_def, "")
        layer.points = raw_arrays["points"]
        if "colors_u8" in raw_arrays and raw_arrays["colors_u8"] is not None:
            layer.colors_u8 = raw_arrays["colors_u8"]
            layer.colors = None
        elif "colors" in raw_arrays and raw_arrays["colors"] is not None:
            layer.colors = raw_arrays["colors"]
        layer.point_count = len(layer.points)
        layer.loaded = True
        return layer

    def _make_pcd_layer(self, pcd, layer_id, name, visible=True,
                        color=None, opacity=1.0):
        layer_def = {
            "id": layer_id, "name": name, "type": "pointcloud",
            "visible": visible, "color": color, "opacity": opacity,
        }
        layer = LayerData(layer_def, "")
        n_pts = len(pcd.points)
        if n_pts == 0: return layer
        
        # Optimized chunked extraction
        CHUNK_SIZE = 50_000_000
        layer.points = np.empty((n_pts, 3), dtype=np.float32)
        pts_view = np.asarray(pcd.points)
        for start in range(0, n_pts, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_pts)
            layer.points[start:end] = pts_view[start:end].astype(np.float32)
        
        if pcd.has_colors():
            layer.colors_u8 = np.empty((n_pts, 3), dtype=np.uint8)
            colors_view = np.asarray(pcd.colors)
            for start in range(0, n_pts, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_pts)
                chunk_f = colors_view[start:end]
                scaled  = chunk_f * 255.0
                np.clip(scaled, 0, 255, out=scaled)
                layer.colors_u8[start:end] = scaled.astype(np.uint8)
            layer.colors = None
            
        if pcd.has_normals():
            layer.normals = np.empty((n_pts, 3), dtype=np.float32)
            normals_view = np.asarray(pcd.normals)
            for start in range(0, n_pts, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_pts)
                layer.normals[start:end] = normals_view[start:end].astype(np.float32)
        
        layer.point_count = n_pts
        layer.loaded = True
        return layer
    # ------------------------------------------------------------------
    # Panorama extraction (optional: libe57 + Pillow)
    # ------------------------------------------------------------------

    def _extract_panoramas(self, path: str) -> list:
        """Extract panorama images using the modular panorama subpackage.

        Delegates to rendering.panorama.extractor -- see that module for
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
    """E57 importer for the plugin system -- wraps E57ImportWorker."""

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
