"""Shared fixtures for locul3d tests."""

from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.fixture()
def tmp_video(tmp_path):
    """Return a path (inside pytest's tmp_path) for a throwaway .mp4."""
    return tmp_path / "out.mp4"


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _has_nvenc() -> bool:
    """Check if h264_nvenc is available in the local ffmpeg build."""
    if not _has_ffmpeg():
        return False
    try:
        from locul3d.recording.encoders import list_available_encoders
        return "h264_nvenc" in list_available_encoders()
    except Exception:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _has_ffmpeg(), reason="ffmpeg not on PATH"
)
requires_ffprobe = pytest.mark.skipif(
    not _has_ffprobe(), reason="ffprobe not on PATH"
)
requires_nvenc = pytest.mark.skipif(
    not _has_nvenc(), reason="h264_nvenc not available"
)


def ffprobe_json(path: str) -> dict:
    """Run ffprobe and return the parsed JSON output."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 0, f"ffprobe failed: {out.stderr}"
    import json
    return json.loads(out.stdout)
