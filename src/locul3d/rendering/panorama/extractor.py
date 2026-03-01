"""E57 panorama image extraction via libe57 low-level API.

Reads the ``images2D`` node from an E57 file and produces a list of
station dictionaries ready to be turned into panorama ``LayerData``
objects by the caller.

Supported E57 image representations:
  - pinholeRepresentation    (perspective camera)
  - sphericalRepresentation  (equirectangular / fisheye)
  - cylindricalRepresentation
  - visualReferenceRepresentation
  - direct blobs (jpegImage / pngImage at image-node level)
"""

from __future__ import annotations

import concurrent.futures
import io
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

# =====================================================================
# Default panorama extraction settings
# Change these to adjust how panorama images are grouped and classified.
# =====================================================================

# Tolerance (metres) for grouping images by position into one station.
# Images within this radius are considered the same camera location.
STATION_GROUP_TOLERANCE = 0.01

# Aspect-ratio threshold to auto-detect spherical (equirectangular)
# images when no explicit representation key is present.
# If  width >= SPHERICAL_ASPECT_RATIO * height  → treat as spherical.
SPHERICAL_ASPECT_RATIO = 1.8

# Default colour for panorama layers (R, G, B) — 0.0–1.0
PANORAMA_LAYER_COLOR = (1.0, 0.7, 0.0)  # orange

# Default panorama layer opacity
PANORAMA_LAYER_OPACITY = 1.0

# =====================================================================


def _process_image_task(task: dict[str, Any]) -> dict[str, Any] | None:
    """Decode and pre-process a single image (parallelizable).

    This performs the CPU-heavy JPEG/PNG decode and the horizontal
    mirror (for inside-out sphere UV convention) in a worker thread.
    Orientation correction is now handled via GL quaternion rotation
    in the renderer — pixel shifts can't represent roll.
    """
    buf = task["buf"]
    try:
        pil_img = PILImage.open(io.BytesIO(buf.tobytes()))
        pil_img.load()
    except Exception:
        return None

    w = task["w_meta"] or pil_img.width
    h = task["h_meta"] or pil_img.height
    rep_key = task["rep_key"]

    is_spherical = (
        rep_key == "sphericalRepresentation"
        or (rep_key == "direct" and w >= SPHERICAL_ASPECT_RATIO * h)
    )

    # Mirror for spherical/cylindrical (inside-out sphere UV convention)
    if is_spherical or rep_key == "cylindricalRepresentation":
        pil_img = pil_img.transpose(PILImage.FLIP_LEFT_RIGHT)

    return {
        "img": pil_img,
        "pos": task["pos"],
        "quat": task["quat"],
        "w": w,
        "h": h,
        "spherical": is_spherical,
        "rep_key": rep_key,
    }


def extract_panoramas(path: str,
                      log_fn: Optional[Callable[[str], None]] = None,
                      ) -> List[dict]:
    """Extract panorama images from an E57 file.

    Returns a list of station dictionaries::

        {
            "id":       "panorama_0",
            "name":     "Panorama 0 (spherical, 8192x4096)",
            "position": np.array([x, y, z]),
            "rotation": (qx, qy, qz, qw) or None,
            "type":     "spherical" | "cubemap",
            "equirect": PIL.Image | None,       # for spherical
            "faces":    [PIL.Image, ...] | None, # for cubemap (6 faces)
            "color":    PANORAMA_LAYER_COLOR,
            "opacity":  PANORAMA_LAYER_OPACITY,
        }

    Parameters
    ----------
    path : str
        Path to the E57 file.
    log_fn : callable, optional
        Logging callback ``log_fn(message: str)``.
    """
    try:
        import libe57
    except ImportError:
        try:
            from pye57 import libe57
        except ImportError:
            _log(log_fn, "Skipping panoramas (libe57 not available)")
            return []

    try:
        imf = libe57.ImageFile(path, mode="r")
    except Exception as e:
        _log(log_fn, f"libe57 open failed: {e}")
        return []

    root = imf.root()

    if not _has(root, "images2D"):
        imf.close()
        _log(log_fn, "No images2D node in E57")
        return []

    images2d = root["images2D"]
    n_img = images2d.childCount()
    _log(log_fn, f"Found {n_img} image entries in images2D")

    if n_img == 0:
        imf.close()
        return []

    # ------------------------------------------------------------------
    # Pass 1 — Read metadata and raw blobs sequentially (libe57 is not thread-safe)
    # ------------------------------------------------------------------
    tasks = []
    for i in range(n_img):
        sn = _struct(images2d.get(i))
        if i < 3:
            _log(log_fn, f"  image[{i}] keys: {_child_keys(sn)}")

        # Extract pose
        pos = np.zeros(3)
        quat = None
        if _has(sn, "pose"):
            pose = _struct(sn["pose"])
            if _has(pose, "translation"):
                t = _struct(pose["translation"])
                pos = np.array([_val(t, "x"), _val(t, "y"), _val(t, "z")])
            if _has(pose, "rotation"):
                r = _struct(pose["rotation"])
                quat = (_val(r, "w", 1.0), _val(r, "x"),
                        _val(r, "y"), _val(r, "z"))

        rep_key, rep = _find_representation(sn)
        if rep is None:
            continue

        blob_key = _find_blob_key(rep)
        if blob_key is None:
            continue

        blob = rep[blob_key]
        buf = np.zeros(blob.byteCount(), dtype=np.uint8)
        blob.read(buf, 0, len(buf))

        # Metadata for processing
        tasks.append({
            "idx": i,
            "buf": buf,
            "pos": pos,
            "quat": quat,
            "rep_key": rep_key,
            "w_meta": int(_val(rep, "imageWidth", 0)),
            "h_meta": int(_val(rep, "imageHeight", 0)),
        })

    imf.close()

    # ------------------------------------------------------------------
    # Pass 2 — Parallel decode and process (CPU bound)
    # ------------------------------------------------------------------
    if not tasks:
        return []

    _log(log_fn, f"Decoding {len(tasks)} images in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_image_task, tasks))

    # ------------------------------------------------------------------
    # Pass 3 — Group by rounded position and build stations
    # ------------------------------------------------------------------
    groups: Dict[Tuple[float, float, float], List[dict]] = {}
    for res in results:
        if res is None:
            continue
        
        pos = res["pos"]
        tol = STATION_GROUP_TOLERANCE
        key = (round(pos[0] / tol) * tol,
               round(pos[1] / tol) * tol,
               round(pos[2] / tol) * tol)
        groups.setdefault(key, []).append(res)

    # ------------------------------------------------------------------
    # Pass 4 — build station records (dispatched by type)
    # ------------------------------------------------------------------
    stations: List[dict] = []
    for idx, (key, imgs) in enumerate(groups.items()):
        pos = imgs[0]["pos"]
        quat = imgs[0].get("quat")
        rep_key = imgs[0].get("rep_key", "")

        pano_type = _classify_pano_type(imgs)

        # Auto-detect 180° yaw flip: only for spherical/cylindrical
        # (cubemap face sorting handles its own orientation)
        if quat is not None and pano_type in (PanoType.SPHERICAL,
                                               PanoType.CYLINDRICAL):
            w, x, y, z = quat
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            yaw_deg = math.degrees(yaw)
            if abs(yaw_deg) > 135:
                # Multiply quaternion by 180° Z rotation: q * (0,0,0,1)
                # Result: (-z, y, -x, w)
                quat = (-z, y, -x, w)

        builder = _PANO_BUILDERS[pano_type]
        station = builder(idx, imgs, pos, quat)

        stations.append(station)
        q_str = ""
        if quat:
            q_str = (f", quat=[{quat[0]:.3f},{quat[1]:.3f},"
                     f"{quat[2]:.3f},{quat[3]:.3f}]")
        _log(log_fn, f"Panorama {idx}: {station['type']} ({rep_key}), "
             f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]{q_str}")

    _log(log_fn, f"Extracted {len(stations)} panorama stations "
         f"from {n_img} images")
    return stations

# =====================================================================
# Panorama type classification
# =====================================================================

class PanoType(Enum):
    """E57 panorama representation types."""
    SPHERICAL = "spherical"
    CUBEMAP = "cubemap"
    CYLINDRICAL = "cylindrical"
    VISUAL_REF = "visual_ref"


# Map E57 representation keys → PanoType
_REP_KEY_MAP = {
    "sphericalRepresentation": PanoType.SPHERICAL,
    "pinholeRepresentation": PanoType.CUBEMAP,
    "cylindricalRepresentation": PanoType.CYLINDRICAL,
    "visualReferenceRepresentation": PanoType.VISUAL_REF,
}


def _classify_pano_type(imgs: list) -> PanoType:
    """Determine the panorama type from the image group.

    Uses the E57 representation key when available, with a heuristic
    fallback for direct blobs (no representation node).
    """
    rep_key = imgs[0].get("rep_key", "")
    ptype = _REP_KEY_MAP.get(rep_key)
    if ptype is not None:
        return ptype
    # Heuristic for direct blobs
    if imgs[0]["spherical"]:
        return PanoType.SPHERICAL
    if len(imgs) >= 6:
        return PanoType.CUBEMAP
    return PanoType.VISUAL_REF


# =====================================================================
# Station builders — one per E57 panorama type
# =====================================================================

def _build_spherical_station(idx, imgs, pos, quat) -> dict:
    """Build station from a sphericalRepresentation (equirectangular).

    Processing (now handled in parallel _process_image_task):
      1. Mirror horizontally
      2. Apply station yaw + pitch correction
    """
    equirect = imgs[0]["img"]
    return {
        "id": f"panorama_{idx}",
        "name": f"Panorama {idx} (spherical, {imgs[0]['w']}x{imgs[0]['h']})",
        "position": pos,
        "rotation": quat,
        "type": "spherical",
        "equirect": equirect,
        "faces": None,
        "color": PANORAMA_LAYER_COLOR,
        "opacity": PANORAMA_LAYER_OPACITY,
    }


def _build_cubemap_station(idx, imgs, pos, quat) -> dict:
    """Build station from pinholeRepresentation faces (cubemap).

    Used by Leica BLK2GO, BLK360, and similar scanners that produce
    6 pinhole camera images per station.

    Processing:
      1. Sort faces by quaternion look-direction into standard slots
      2. Z-axis slots swapped to correct pinhole vertical inversion
    """
    sorted_faces = _sort_cubemap_faces(imgs)
    return {
        "id": f"panorama_{idx}",
        "name": f"Panorama {idx} (cubemap, {len(imgs)} faces)",
        "position": pos,
        "rotation": quat,
        "type": "cubemap",
        "equirect": None,
        "faces": sorted_faces,
        "color": PANORAMA_LAYER_COLOR,
        "opacity": PANORAMA_LAYER_OPACITY,
    }


def _build_cylindrical_station(idx, imgs, pos, quat) -> dict:
    """Build station from a cylindricalRepresentation.

    Processing (now handled in parallel _process_image_task).
    """
    equirect = imgs[0]["img"]
    return {
        "id": f"panorama_{idx}",
        "name": f"Panorama {idx} (cylindrical, {imgs[0]['w']}x{imgs[0]['h']})",
        "position": pos,
        "rotation": quat,
        "type": "cylindrical",
        "equirect": equirect,
        "faces": None,
        "color": PANORAMA_LAYER_COLOR,
        "opacity": PANORAMA_LAYER_OPACITY,
    }


def _build_visual_ref_station(idx, imgs, pos, quat) -> dict:
    """Build station from a visualReferenceRepresentation.

    Visual reference images are typically single overview photos
    (not full 360°).  We treat them as equirectangular for display
    purposes — they'll appear stretched on the sphere but still
    provide useful context.
    """
    equirect = imgs[0]["img"]
    return {
        "id": f"panorama_{idx}",
        "name": f"Panorama {idx} (visual ref, {imgs[0]['w']}x{imgs[0]['h']})",
        "position": pos,
        "rotation": quat,
        "type": "visual_ref",
        "equirect": equirect,
        "faces": None,
        "color": PANORAMA_LAYER_COLOR,
        "opacity": PANORAMA_LAYER_OPACITY,
    }


# Dispatch table: PanoType → builder function
_PANO_BUILDERS = {
    PanoType.SPHERICAL: _build_spherical_station,
    PanoType.CUBEMAP: _build_cubemap_station,
    PanoType.CYLINDRICAL: _build_cylindrical_station,
    PanoType.VISUAL_REF: _build_visual_ref_station,
}


# =====================================================================
# libe57 helpers
# =====================================================================

def _log(fn, msg):
    if fn:
        fn(msg)


def _struct(node):
    """Cast a generic libe57 Node to StructureNode."""
    try:
        import libe57
    except ImportError:
        from pye57 import libe57
    if isinstance(node, libe57.StructureNode):
        return node
    return libe57.StructureNode(node)


def _has(node, key):
    try:
        _ = node[key]
        return True
    except Exception:
        return False


def _val(node, key, default=0.0):
    try:
        return node[key].value()
    except Exception:
        return default


def _child_keys(node):
    keys = []
    try:
        for ci in range(node.childCount()):
            pn = node.get(ci).pathName()
            keys.append(pn.rsplit("/", 1)[-1])
    except Exception:
        pass
    return keys


def _find_representation(sn):
    """Find the image representation on an E57 image struct node."""
    for rk in ("pinholeRepresentation", "sphericalRepresentation",
               "cylindricalRepresentation", "visualReferenceRepresentation"):
        if _has(sn, rk):
            return rk, _struct(sn[rk])

    # Fallback: blob sits directly on the image node
    for bk in ("jpegImage", "pngImage"):
        if _has(sn, bk):
            return "direct", sn
    return None, None


def _find_blob_key(rep):
    """Find the JPEG/PNG blob key within a representation node."""
    for bk in ("jpegImage", "pngImage", "imageMask"):
        if _has(rep, bk):
            return bk
    return None


def _apply_orientation_to_equirect(img, quat, idx=None):
    """Shift equirectangular image by the station yaw and pitch.

    Spherical panoramas from NavVis, Faro (and similar scanners) are stored
    in the scanner's local frame.  The station quaternion (w, x, y, z)
    rotates local → world.  We extract the yaw and pitch components
    and roll the image pixels so the panorama aligns with world
    coordinates.

    Yaw  → horizontal pixel shift (exact for equirectangular)
    Pitch → vertical pixel shift (good approximation for small angles)
    """
    w, x, y, z = quat

    # Yaw (rotation about world Z axis) from quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # Pitch (rotation about world Y axis) from quaternion
    sin_pitch = 2 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))  # clamp for asin safety
    pitch = math.asin(sin_pitch)

    width, height = img.size
    h_shift = int(round(-yaw / (2 * math.pi) * width))
    v_shift = int(round(pitch / math.pi * height))

    if h_shift == 0 and v_shift == 0:
        return img

    arr = np.array(img)
    if h_shift != 0:
        arr = np.roll(arr, h_shift, axis=1)
    if v_shift != 0:
        arr = np.roll(arr, v_shift, axis=0)
    return PILImage.fromarray(arr)


def _sort_cubemap_faces(imgs: list) -> list:
    """Sort cubemap faces into standard slots by per-face quaternion.

    E57 pinhole cameras look along +Z in local frame.
    Rotating [0,0,1] by each face's quaternion gives the world-space
    look direction.  The dominant axis maps to a cubemap slot:
      [+X, -X, +Y, -Y, +Z, -Z] → slots [0, 1, 2, 3, 4, 5]

    Falls back to file order for faces without quaternion data.
    """
    sorted_faces: list = [None] * 6

    for d in imgs:
        q = d.get("quat")
        if q is None:
            continue
        w, x, y, z = q
        # Rotate [0,0,1] by quaternion (q * [0,0,1] * q^-1)
        fwd = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        abs_fwd = np.abs(fwd)
        axis = int(np.argmax(abs_fwd))
        sign = 1 if fwd[axis] > 0 else -1
        # Z-axis: swap +Z/-Z slots to correct for pinhole image
        # inversion on vertical faces (E57 convention)
        if axis == 2:
            slot = 5 if sign > 0 else 4
        else:
            slot = axis * 2 + (0 if sign > 0 else 1)
        if sorted_faces[slot] is None:
            sorted_faces[slot] = d["img"]

    # Fallback: fill missing slots from original order
    orig_idx = 0
    for s in range(6):
        if sorted_faces[s] is None and orig_idx < len(imgs):
            sorted_faces[s] = imgs[orig_idx]["img"]
            orig_idx += 1

    return sorted_faces
