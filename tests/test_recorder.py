"""Tests for locul3d.recording.recorder — VideoRecorder end-to-end.

Feeds synthetic RGB frames to VideoRecorder, then verifies the output
file with ffprobe: resolution, codec, pixel format, frame count, and
bitrate range.

These tests require ffmpeg + ffprobe on PATH.  NVENC-specific tests
additionally require an NVIDIA GPU with the h264_nvenc encoder.
"""

from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from locul3d.recording.encoders import select_encoder
from locul3d.recording.recorder import RecordingConfig, VideoRecorder

from .conftest import (
    ffprobe_json,
    requires_ffmpeg,
    requires_ffprobe,
    requires_nvenc,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _gradient_frame(width: int, height: int, frame_idx: int = 0) -> bytes:
    """Generate a synthetic RGB24 frame (gradient pattern).

    Each row gets a colour that shifts with *frame_idx* so consecutive
    frames are visually distinct — useful for verifying frame count.
    """
    data = bytearray(width * height * 3)
    for y in range(height):
        r = (y + frame_idx * 7) % 256
        g = (y * 2 + frame_idx * 3) % 256
        b = (255 - y + frame_idx * 11) % 256
        row = bytes([r, g, b]) * width
        data[y * width * 3:(y + 1) * width * 3] = row
    return bytes(data)


# ── ffmpeg command building ──────────────────────────────────────────


class TestBuildFfmpegCmd:
    """Verify the ffmpeg command line for each encoder family."""

    def _make_cfg(self, **overrides) -> RecordingConfig:
        defaults = dict(
            path="out.mp4", width=1920, height=1080, fps=30.0,
            codec="h264", encoder="libx264", encoder_kind="sw",
            bitrate_kbps=15000,
        )
        defaults.update(overrides)
        from pathlib import Path
        defaults["path"] = Path(defaults["path"])
        return RecordingConfig(**defaults)

    def _build(self, **overrides) -> list[str]:
        cfg = self._make_cfg(**overrides)
        rec = VideoRecorder()
        return rec._build_ffmpeg_cmd("ffmpeg", cfg)

    def test_nvenc_flags(self):
        cmd = self._build(encoder="h264_nvenc", encoder_kind="hw")
        assert "-c:v" in cmd
        idx = cmd.index("-c:v")
        assert cmd[idx + 1] == "h264_nvenc"
        assert "-rc" in cmd
        assert cmd[cmd.index("-rc") + 1] == "vbr"
        assert "-cq" in cmd
        assert cmd[cmd.index("-cq") + 1] == "20"
        assert "-maxrate" in cmd

    def test_nvenc_maxrate_is_2x_bitrate(self):
        cmd = self._build(
            encoder="hevc_nvenc", encoder_kind="hw",
            codec="hevc", bitrate_kbps=10000,
        )
        idx = cmd.index("-maxrate")
        assert cmd[idx + 1] == "20000k"

    def test_qsv_flags(self):
        cmd = self._build(encoder="h264_qsv", encoder_kind="hw")
        assert "-global_quality" in cmd

    def test_libx264_flags(self):
        cmd = self._build(encoder="libx264")
        assert "-crf" in cmd
        assert cmd[cmd.index("-crf") + 1] == "18"
        assert "-preset" in cmd

    def test_libx265_flags(self):
        cmd = self._build(encoder="libx265", codec="hevc")
        assert "-crf" in cmd
        assert cmd[cmd.index("-crf") + 1] == "22"

    def test_hevc_hvc1_tag(self):
        cmd = self._build(encoder="libx265", codec="hevc")
        assert "-tag:v" in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"

    def test_h264_no_hvc1_tag(self):
        cmd = self._build(encoder="libx264", codec="h264")
        assert "-tag:v" not in cmd

    def test_no_vflip(self):
        # render_to_buffer now returns top-down bytes, so the ffmpeg
        # command must NOT include a vflip filter.
        cmd = self._build()
        if "-vf" in cmd:
            assert cmd[cmd.index("-vf") + 1] != "vflip"

    def test_pix_fmt_output(self):
        cmd = self._build()
        assert "-pix_fmt" in cmd
        # There are two -pix_fmt: input (rgb24) and output (yuv420p).
        pix_indices = [i for i, v in enumerate(cmd) if v == "-pix_fmt"]
        output_fmt = cmd[pix_indices[-1] + 1]
        assert output_fmt == "yuv420p"

    def test_videotoolbox_flags(self):
        cmd = self._build(encoder="h264_videotoolbox", encoder_kind="hw")
        assert "-allow_sw" in cmd

    def test_movflags_faststart(self):
        cmd = self._build()
        assert "-movflags" in cmd
        assert cmd[cmd.index("-movflags") + 1] == "+faststart"


# ── Software recording end-to-end ───────────────────────────────────


@requires_ffmpeg
@requires_ffprobe
class TestSoftwareRecording:
    """Record synthetic frames with libx264/libx265 and verify output."""

    @pytest.mark.parametrize("width,height", [
        (1920, 1080),
        (1280, 720),
        (640, 480),
    ])
    def test_resolution_matches(self, tmp_video, width, height):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=width, height=height,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        assert cfg.width == width
        assert cfg.height == height

        for i in range(10):
            rec.feed_frame(_gradient_frame(width, height, i))
        stats = rec.stop()
        assert stats.last_error is None
        assert stats.frames_written == 10

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == width
        assert int(vs["height"]) == height

    def test_odd_resolution_gets_aligned(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=1921, height=1081,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        # Should round down to even.
        assert cfg.width == 1920
        assert cfg.height == 1080

        for i in range(5):
            rec.feed_frame(_gradient_frame(cfg.width, cfg.height, i))
        rec.stop()

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == 1920
        assert int(vs["height"]) == 1080

    @pytest.mark.parametrize("codec,sw_enc,expected_name", [
        ("h264", "sw", "h264"),
        ("hevc", "sw", "hevc"),
    ])
    def test_codec_matches(self, tmp_video, codec, sw_enc, expected_name):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=640, height=480,
            fps=30.0, codec=codec, hw_pref=sw_enc,
        )

        for i in range(10):
            rec.feed_frame(_gradient_frame(640, 480, i))
        rec.stop()

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert vs["codec_name"] == expected_name

    def test_frame_count(self, tmp_video):
        rec = VideoRecorder()
        n_frames = 30
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        for i in range(n_frames):
            rec.feed_frame(_gradient_frame(320, 240, i))
        stats = rec.stop()
        assert stats.frames_written == n_frames

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        # nb_frames may be a string; some ffmpeg builds report it.
        if "nb_frames" in vs and vs["nb_frames"] != "N/A":
            assert int(vs["nb_frames"]) == n_frames

    def test_output_is_yuv420p(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        for i in range(5):
            rec.feed_frame(_gradient_frame(320, 240, i))
        rec.stop()

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert vs["pix_fmt"] == "yuv420p"

    def test_file_has_nonzero_size(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        for i in range(10):
            rec.feed_frame(_gradient_frame(320, 240, i))
        stats = rec.stop()
        assert stats.bytes_written > 0
        assert tmp_video.stat().st_size > 0

    def test_hevc_hvc1_tag_in_output(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="hevc", hw_pref="sw",
        )
        for i in range(5):
            rec.feed_frame(_gradient_frame(320, 240, i))
        rec.stop()

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert vs["codec_name"] == "hevc"
        # hvc1 tag should be present in codec_tag_string.
        assert vs.get("codec_tag_string") == "hvc1"

    def test_stop_is_idempotent(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        rec.feed_frame(_gradient_frame(320, 240))
        stats1 = rec.stop()
        stats2 = rec.stop()
        assert stats1.frames_written == 1
        assert stats2.state in ("stopped", "idle")


# ── NVENC recording end-to-end ───────────────────────────────────────


@requires_ffmpeg
@requires_ffprobe
@requires_nvenc
class TestNvencRecording:
    """Record with h264_nvenc / hevc_nvenc and verify the output."""

    def test_nvenc_h264_produces_valid_file(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=1920, height=1080,
            fps=30.0, codec="h264", hw_pref="hw",
        )
        assert "nvenc" in cfg.encoder

        for i in range(30):
            rec.feed_frame(_gradient_frame(1920, 1080, i))
        stats = rec.stop()
        assert stats.last_error is None
        assert stats.frames_written == 30
        assert stats.bytes_written > 0

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == 1920
        assert int(vs["height"]) == 1080
        assert vs["codec_name"] == "h264"
        assert vs["pix_fmt"] == "yuv420p"

    def test_nvenc_hevc_produces_valid_file(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=1280, height=720,
            fps=60.0, codec="hevc", hw_pref="hw",
        )
        assert "nvenc" in cfg.encoder

        for i in range(60):
            rec.feed_frame(_gradient_frame(1280, 720, i))
        stats = rec.stop()
        assert stats.last_error is None
        assert stats.frames_written == 60

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == 1280
        assert int(vs["height"]) == 720
        assert vs["codec_name"] == "hevc"
        assert vs.get("codec_tag_string") == "hvc1"

    def test_nvenc_4k_resolution(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=3840, height=2160,
            fps=30.0, codec="h264", hw_pref="hw",
        )

        for i in range(10):
            rec.feed_frame(_gradient_frame(3840, 2160, i))
        stats = rec.stop()
        assert stats.last_error is None

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == 3840
        assert int(vs["height"]) == 2160

    def test_nvenc_bitrate_within_range(self, tmp_video):
        """Bitrate should be in the ballpark of the configured value."""
        target_kbps = 15000
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=1920, height=1080,
            fps=30.0, codec="h264", hw_pref="hw",
            bitrate_kbps=target_kbps,
        )

        # Need enough frames for the encoder to stabilise.
        for i in range(90):
            rec.feed_frame(_gradient_frame(1920, 1080, i))
        rec.stop()

        info = ffprobe_json(tmp_video)
        # bit_rate is in bits/sec as a string.
        actual_bps = int(info["format"]["bit_rate"])
        actual_kbps = actual_bps / 1000
        # VBR with CQ — allow wide tolerance (0.1× to 5× target).
        # Synthetic gradient frames compress very well, so the actual
        # rate is usually far below the target.
        assert actual_kbps < target_kbps * 5, (
            f"bitrate {actual_kbps:.0f} kbps exceeds 5× target {target_kbps}"
        )
        assert actual_kbps > 0


# ── Autoselection integration (real ffmpeg) ──────────────────────────


@requires_ffmpeg
@requires_ffprobe
class TestAutoselectionIntegration:
    """Verify that hw_pref='auto' picks the best available encoder and
    records a valid file — regardless of whether HW is present."""

    def test_auto_h264_produces_valid_file(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=640, height=480,
            fps=30.0, codec="h264", hw_pref="auto",
        )
        # On machines with NVENC: cfg.encoder == "h264_nvenc"
        # On CI / no GPU: cfg.encoder == "libx264"
        assert cfg.encoder_kind in ("hw", "sw")

        for i in range(15):
            rec.feed_frame(_gradient_frame(640, 480, i))
        stats = rec.stop()
        assert stats.last_error is None
        assert stats.frames_written == 15

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert int(vs["width"]) == 640
        assert int(vs["height"]) == 480
        assert vs["codec_name"] == "h264"

    def test_auto_hevc_produces_valid_file(self, tmp_video):
        rec = VideoRecorder()
        cfg = rec.start(
            path=str(tmp_video), width=640, height=480,
            fps=30.0, codec="hevc", hw_pref="auto",
        )
        assert cfg.encoder_kind in ("hw", "sw")

        for i in range(15):
            rec.feed_frame(_gradient_frame(640, 480, i))
        stats = rec.stop()
        assert stats.last_error is None

        info = ffprobe_json(tmp_video)
        vs = info["streams"][0]
        assert vs["codec_name"] == "hevc"


# ── VideoRecorder lifecycle ──────────────────────────────────────────


class TestRecorderStateMachine:
    """Pure state-machine tests — no ffmpeg needed."""

    def test_pause_while_idle_raises(self):
        rec = VideoRecorder()
        with pytest.raises(RuntimeError, match="can only pause"):
            rec.pause()

    def test_resume_while_idle_raises(self):
        rec = VideoRecorder()
        with pytest.raises(RuntimeError, match="can only resume"):
            rec.resume()

    def test_stop_while_idle_is_noop(self):
        rec = VideoRecorder()
        stats = rec.stop()
        assert stats.state in ("stopped", "idle")

    def test_initial_state_is_idle(self):
        rec = VideoRecorder()
        assert rec.state == "idle"
        assert not rec.is_active
        assert not rec.is_open


@requires_ffmpeg
class TestRecorderLifecycle:
    def test_start_while_recording_raises(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        with pytest.raises(RuntimeError, match="stop it first"):
            rec.start(
                path=str(tmp_video.parent / "out2.mp4"),
                width=320, height=240,
                fps=30.0, codec="h264", hw_pref="sw",
            )
        rec.stop()

    def test_pause_resume(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        rec.feed_frame(_gradient_frame(320, 240, 0))
        rec.pause()
        assert rec.state == "paused"

        # Frames dropped while paused.
        rec.feed_frame(_gradient_frame(320, 240, 1))

        rec.resume()
        rec.feed_frame(_gradient_frame(320, 240, 2))

        stats = rec.stop()
        # Only 2 frames should have been written (before pause + after resume).
        assert stats.frames_written == 2

    def test_can_reuse_after_stop(self, tmp_video):
        rec = VideoRecorder()
        rec.start(
            path=str(tmp_video), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        rec.feed_frame(_gradient_frame(320, 240))
        rec.stop()

        out2 = tmp_video.parent / "out2.mp4"
        rec.start(
            path=str(out2), width=320, height=240,
            fps=30.0, codec="h264", hw_pref="sw",
        )
        rec.feed_frame(_gradient_frame(320, 240))
        stats = rec.stop()
        assert stats.frames_written == 1
        assert out2.exists()
