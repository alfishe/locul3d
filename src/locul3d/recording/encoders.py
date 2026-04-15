"""ffmpeg encoder discovery and selection.

The video recorder spawns ``ffmpeg`` as a subprocess and pipes raw
frames to it.  This module figures out *which* encoder to use given
the user's preferences (codec, hw vs sw) and the platform.

Cross-platform encoder priority order:

    H.264:
      darwin → h264_videotoolbox
      win32  → h264_nvenc, h264_qsv
      linux  → h264_nvenc, h264_vaapi, h264_qsv

    HEVC (H.265):
      darwin → hevc_videotoolbox
      win32  → hevc_nvenc, hevc_qsv
      linux  → hevc_nvenc, hevc_vaapi, hevc_qsv

    Software fallback:
      h264 → libx264
      hevc → libx265

Discovery is done once per process by parsing ``ffmpeg -encoders``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class EncoderUnavailable(RuntimeError):
    """Raised when no encoder satisfies the request."""


# ── Encoder priority tables ─────────────────────────────────────────

_HW_BY_CODEC_PLATFORM: dict = {
    "h264": {
        "darwin": ["h264_videotoolbox"],
        "win32":  ["h264_nvenc", "h264_qsv"],
        "linux":  ["h264_nvenc", "h264_vaapi", "h264_qsv"],
    },
    "hevc": {
        "darwin": ["hevc_videotoolbox"],
        "win32":  ["hevc_nvenc", "hevc_qsv"],
        "linux":  ["hevc_nvenc", "hevc_vaapi", "hevc_qsv"],
    },
}

_SW_BY_CODEC: dict = {
    "h264": "libx264",
    "hevc": "libx265",
}

# Aliases users may type on the CLI / API.
_CODEC_ALIASES: dict = {
    "h264": "h264", "h.264": "h264", "avc": "h264", "x264": "h264",
    "hevc": "hevc", "h265": "hevc", "h.265": "hevc", "x265": "hevc",
}


# ── ffmpeg discovery ────────────────────────────────────────────────

_ENCODER_CACHE: Optional[Set[str]] = None
_FFMPEG_PATH_CACHE: Optional[str] = None


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary, raising if absent.

    Honors LOCUL3D_FFMPEG env var, then PATH.
    """
    global _FFMPEG_PATH_CACHE
    if _FFMPEG_PATH_CACHE is not None:
        return _FFMPEG_PATH_CACHE
    import os
    candidate = os.environ.get("LOCUL3D_FFMPEG") or shutil.which("ffmpeg")
    if not candidate:
        raise EncoderUnavailable(
            "ffmpeg not found on PATH. Install ffmpeg or set "
            "LOCUL3D_FFMPEG to the binary path."
        )
    _FFMPEG_PATH_CACHE = candidate
    return candidate


def list_available_encoders(refresh: bool = False) -> Set[str]:
    """Return the set of video encoder names ffmpeg knows about.

    Result is cached for the process lifetime.  Pass ``refresh=True``
    to re-scan.
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None and not refresh:
        return _ENCODER_CACHE

    ffmpeg = find_ffmpeg()
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise EncoderUnavailable(f"failed to query ffmpeg encoders: {exc}")

    encoders: Set[str] = set()
    in_table = False
    for line in out.stdout.splitlines():
        # Header line precedes the table:
        #   ------
        if line.strip().startswith("------"):
            in_table = True
            continue
        if not in_table:
            continue
        # Lines look like:
        #   ' V..... libx264              libx264 H.264 / AVC ...'
        # The flag column starts with a single letter; 'V' = video.
        stripped = line.strip()
        if not stripped or len(stripped) < 8:
            continue
        flags = stripped.split(None, 1)[0]
        if not flags or flags[0] != "V":
            continue
        rest = stripped[len(flags):].strip()
        name = rest.split(None, 1)[0] if rest else ""
        if name:
            encoders.add(name)

    _ENCODER_CACHE = encoders
    return encoders


# ── Selection ───────────────────────────────────────────────────────

def normalize_codec(codec: str) -> str:
    key = (codec or "").strip().lower()
    if key not in _CODEC_ALIASES:
        raise EncoderUnavailable(
            f"unknown codec {codec!r}; expected one of: h264, hevc"
        )
    return _CODEC_ALIASES[key]


def select_encoder(
    codec: str,
    hw_pref: str = "auto",
    available: Optional[Iterable[str]] = None,
) -> Tuple[str, str, List[str]]:
    """Pick the best encoder for the requested codec.

    Args:
        codec: ``"h264"`` or ``"hevc"`` (aliases accepted).
        hw_pref: ``"auto"`` (try HW, fall back to SW),
                 ``"hw"`` (HW only — raises if unavailable),
                 ``"sw"`` (force software libx264/libx265).
        available: optional override of the encoder discovery
                   (mainly for tests).

    Returns:
        ``(encoder_name, kind, warnings)``
        ``kind`` is ``"hw"`` or ``"sw"``.

    Raises:
        EncoderUnavailable on failure.
    """
    codec = normalize_codec(codec)
    hw_pref = (hw_pref or "auto").strip().lower()
    if hw_pref not in ("auto", "hw", "sw"):
        raise EncoderUnavailable(
            f"hw_pref must be auto/hw/sw, got {hw_pref!r}"
        )

    avail = set(available) if available is not None else list_available_encoders()
    warnings: List[str] = []

    if hw_pref in ("auto", "hw"):
        candidates = _HW_BY_CODEC_PLATFORM[codec].get(sys.platform, [])
        for name in candidates:
            if name in avail:
                return name, "hw", warnings
        if hw_pref == "hw":
            raise EncoderUnavailable(
                f"no hardware {codec.upper()} encoder available on "
                f"{sys.platform}. Tried: {candidates}. ffmpeg has: "
                f"{sorted(n for n in avail if codec in n)}"
            )
        warnings.append(
            f"no hardware {codec.upper()} encoder available on "
            f"{sys.platform}; falling back to software"
        )

    sw_name = _SW_BY_CODEC[codec]
    if sw_name not in avail:
        raise EncoderUnavailable(
            f"software encoder {sw_name!r} not available in this "
            f"ffmpeg build; cannot encode {codec.upper()}"
        )
    return sw_name, "sw", warnings


def default_bitrate_kbps(width: int, height: int, fps: float, codec: str) -> int:
    """Sensible default bitrate for HW encoders that need one.

    Roughly tuned to "looks fine" presets — h264 needs ~2× HEVC for
    similar quality.  These can be overridden per recording.
    """
    codec = normalize_codec(codec)
    px = width * height
    fps_factor = max(1.0, fps / 30.0)

    # Per-pixel rate (kbps per megapixel-frame-rate-30) — picked from
    # YouTube's recommended bitrates table, halved for HEVC.
    if codec == "hevc":
        kbps_per_mpx_30 = 7500
    else:
        kbps_per_mpx_30 = 15000

    return int((px / 1_000_000.0) * fps_factor * kbps_per_mpx_30)
