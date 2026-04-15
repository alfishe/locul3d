"""Tests for locul3d.recording.encoders — encoder discovery and selection.

Verifies that NVENC is autoselected when available, fallback behaviour,
codec normalisation, and bitrate defaults.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from locul3d.recording.encoders import (
    EncoderUnavailable,
    default_bitrate_kbps,
    normalize_codec,
    select_encoder,
)


# ── normalize_codec ──────────────────────────────────────────────────


class TestNormalizeCodec:
    @pytest.mark.parametrize("alias,expected", [
        ("h264", "h264"),
        ("H.264", "h264"),
        ("avc", "h264"),
        ("x264", "h264"),
        ("hevc", "hevc"),
        ("h265", "hevc"),
        ("H.265", "hevc"),
        ("x265", "hevc"),
    ])
    def test_valid_aliases(self, alias, expected):
        assert normalize_codec(alias) == expected

    def test_unknown_codec_raises(self):
        with pytest.raises(EncoderUnavailable, match="unknown codec"):
            normalize_codec("vp9")

    def test_empty_string_raises(self):
        with pytest.raises(EncoderUnavailable):
            normalize_codec("")


# ── select_encoder — NVENC autoselection ─────────────────────────────


class TestNvencAutoselection:
    """NVENC must be the first choice on Windows and Linux when available."""

    @patch("locul3d.recording.encoders.sys")
    def test_h264_nvenc_selected_on_win32(self, mock_sys):
        mock_sys.platform = "win32"
        enc, kind, warns = select_encoder(
            "h264", "auto",
            available={"h264_nvenc", "h264_qsv", "libx264"},
        )
        assert enc == "h264_nvenc"
        assert kind == "hw"
        assert warns == []

    @patch("locul3d.recording.encoders.sys")
    def test_hevc_nvenc_selected_on_win32(self, mock_sys):
        mock_sys.platform = "win32"
        enc, kind, warns = select_encoder(
            "hevc", "auto",
            available={"hevc_nvenc", "hevc_qsv", "libx265"},
        )
        assert enc == "hevc_nvenc"
        assert kind == "hw"

    @patch("locul3d.recording.encoders.sys")
    def test_h264_nvenc_selected_on_linux(self, mock_sys):
        mock_sys.platform = "linux"
        enc, kind, warns = select_encoder(
            "h264", "auto",
            available={"h264_nvenc", "h264_vaapi", "libx264"},
        )
        assert enc == "h264_nvenc"
        assert kind == "hw"

    @patch("locul3d.recording.encoders.sys")
    def test_hevc_nvenc_selected_on_linux(self, mock_sys):
        mock_sys.platform = "linux"
        enc, kind, warns = select_encoder(
            "hevc", "auto",
            available={"hevc_nvenc", "hevc_vaapi", "libx265"},
        )
        assert enc == "hevc_nvenc"
        assert kind == "hw"

    @patch("locul3d.recording.encoders.sys")
    def test_nvenc_preferred_over_qsv(self, mock_sys):
        """Windows priority: nvenc > qsv → sw fallback."""
        mock_sys.platform = "win32"
        enc, _, _ = select_encoder(
            "h264", "auto",
            available={"h264_nvenc", "h264_qsv", "libx264"},
        )
        assert enc == "h264_nvenc"

    @patch("locul3d.recording.encoders.sys")
    def test_qsv_when_nvenc_absent(self, mock_sys):
        mock_sys.platform = "win32"
        enc, kind, _ = select_encoder(
            "h264", "auto",
            available={"h264_qsv", "libx264"},
        )
        assert enc == "h264_qsv"
        assert kind == "hw"

    @patch("locul3d.recording.encoders.sys")
    def test_sw_when_nvenc_and_qsv_absent_on_win32(self, mock_sys):
        """No AMF in the chain — falls straight to software."""
        mock_sys.platform = "win32"
        enc, kind, warns = select_encoder(
            "h264", "auto",
            available={"libx264"},
        )
        assert enc == "libx264"
        assert kind == "sw"
        assert any("falling back" in w.lower() for w in warns)


# ── select_encoder — fallback & forced modes ─────────────────────────


class TestEncoderFallback:
    @patch("locul3d.recording.encoders.sys")
    def test_sw_fallback_when_no_hw_available(self, mock_sys):
        mock_sys.platform = "win32"
        enc, kind, warns = select_encoder(
            "h264", "auto",
            available={"libx264"},
        )
        assert enc == "libx264"
        assert kind == "sw"
        assert any("falling back" in w.lower() for w in warns)

    @patch("locul3d.recording.encoders.sys")
    def test_hevc_sw_fallback(self, mock_sys):
        mock_sys.platform = "linux"
        enc, kind, warns = select_encoder(
            "hevc", "auto",
            available={"libx265"},
        )
        assert enc == "libx265"
        assert kind == "sw"
        assert len(warns) > 0

    @patch("locul3d.recording.encoders.sys")
    def test_hw_pref_hw_raises_when_no_hw(self, mock_sys):
        mock_sys.platform = "win32"
        with pytest.raises(EncoderUnavailable, match="no hardware"):
            select_encoder("h264", "hw", available={"libx264"})

    @patch("locul3d.recording.encoders.sys")
    def test_hw_pref_sw_skips_hw(self, mock_sys):
        mock_sys.platform = "win32"
        enc, kind, _ = select_encoder(
            "h264", "sw",
            available={"h264_nvenc", "libx264"},
        )
        assert enc == "libx264"
        assert kind == "sw"

    def test_sw_encoder_missing_raises(self):
        with pytest.raises(EncoderUnavailable, match="not available"):
            select_encoder("h264", "sw", available=set())

    def test_invalid_hw_pref_raises(self):
        with pytest.raises(EncoderUnavailable, match="hw_pref"):
            select_encoder("h264", "gpu", available={"libx264"})


# ── select_encoder — macOS / videotoolbox ────────────────────────────


class TestMacOSEncoder:
    @patch("locul3d.recording.encoders.sys")
    def test_videotoolbox_selected_on_darwin(self, mock_sys):
        mock_sys.platform = "darwin"
        enc, kind, _ = select_encoder(
            "h264", "auto",
            available={"h264_videotoolbox", "libx264"},
        )
        assert enc == "h264_videotoolbox"
        assert kind == "hw"

    @patch("locul3d.recording.encoders.sys")
    def test_hevc_videotoolbox_on_darwin(self, mock_sys):
        mock_sys.platform = "darwin"
        enc, kind, _ = select_encoder(
            "hevc", "auto",
            available={"hevc_videotoolbox", "libx265"},
        )
        assert enc == "hevc_videotoolbox"
        assert kind == "hw"


# ── default_bitrate_kbps ─────────────────────────────────────────────


class TestDefaultBitrate:
    def test_1080p_30fps_h264(self):
        br = default_bitrate_kbps(1920, 1080, 30.0, "h264")
        # ~31 Mbps for h264 1080p@30 is unreasonable; should be ~31k?
        # 1920*1080 = 2_073_600 px, / 1e6 = 2.0736 Mpx
        # 2.0736 * 1.0 * 15000 ≈ 31104 kbps
        assert 25_000 <= br <= 40_000

    def test_1080p_30fps_hevc_is_lower_than_h264(self):
        br_264 = default_bitrate_kbps(1920, 1080, 30.0, "h264")
        br_265 = default_bitrate_kbps(1920, 1080, 30.0, "hevc")
        assert br_265 < br_264

    def test_4k_60fps_h264(self):
        br = default_bitrate_kbps(3840, 2160, 60.0, "h264")
        # 8.29 Mpx * 2.0 fps_factor * 15000 ≈ 248_832 kbps
        assert br > 200_000

    def test_bitrate_scales_with_fps(self):
        br_30 = default_bitrate_kbps(1920, 1080, 30.0, "h264")
        br_60 = default_bitrate_kbps(1920, 1080, 60.0, "h264")
        assert br_60 > br_30

    def test_bitrate_scales_with_resolution(self):
        br_720 = default_bitrate_kbps(1280, 720, 30.0, "h264")
        br_1080 = default_bitrate_kbps(1920, 1080, 30.0, "h264")
        assert br_1080 > br_720

    def test_codec_aliases_accepted(self):
        br = default_bitrate_kbps(1920, 1080, 30.0, "x264")
        assert br > 0
