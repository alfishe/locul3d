"""VideoRecorder — drives an ffmpeg subprocess from the engine.

State machine
-------------

    idle ── start() ──► recording ─┬─ pause()  ──► paused
                                   │
                                   └─ stop()   ──► stopped
    paused ── resume() ──► recording
    paused ── stop()   ──► stopped

While ``recording`` the engine pushes one frame per virtual tick by
calling :meth:`feed_frame`.  While ``paused`` the engine still ticks
the animation but ``feed_frame`` is a no-op, so the same output file
keeps growing as more frames arrive after a resume.

Frames are handed off to a writer thread via a bounded queue; the
writer pipes them to ffmpeg's stdin.  The bound is the natural
backpressure mechanism — when ffmpeg can't keep up, the engine's call
to ``feed_frame`` blocks until a slot frees, which throttles the
render loop to encoder speed.  This is the only stall in the pipeline
and it's exactly what we want.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .encoders import (
    EncoderUnavailable,
    default_bitrate_kbps,
    find_ffmpeg,
    select_encoder,
)

log = logging.getLogger(__name__)


# Resolution presets — UHD (consumer 4K), not DCI.
RESOLUTION_PRESETS: dict = {
    "4k":     (3840, 2160),
    "uhd":    (3840, 2160),
    "1080p":  (1920, 1080),
    "fhd":    (1920, 1080),
    "720p":   (1280, 720),
    "hd":     (1280, 720),
}


@dataclass
class RecordingConfig:
    path: Path
    width: int
    height: int
    fps: float
    codec: str            # "h264" or "hevc"
    encoder: str          # actual ffmpeg -c:v value
    encoder_kind: str     # "hw" or "sw"
    bitrate_kbps: int
    pix_fmt_in: str = "rgb24"   # ffmpeg input pixel format
    pix_fmt_out: str = "yuv420p"  # widely-compatible output


@dataclass
class RecordingStats:
    state: str = "idle"
    frames_written: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    started_at: Optional[float] = None
    duration_s: float = 0.0
    last_error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class VideoRecorder:
    """Engine-driven offscreen recorder.

    The recorder is owned by the AnimationEngine.  Engine calls:

      * :meth:`start` once, before kicking off a capture-mode session
      * :meth:`feed_frame` once per emitted frame
      * :meth:`pause` / :meth:`resume` to halt frame writes without
        closing the file
      * :meth:`stop` to flush ffmpeg and close the file
    """

    # Bounded queue depth (frames).  Memory cost = depth × W × H × 3.
    # 4 frames at 4K rgb24 ≈ 95 MB.
    QUEUE_DEPTH = 4

    def __init__(self) -> None:
        self._cfg: Optional[RecordingConfig] = None
        self._proc: Optional[subprocess.Popen] = None
        self._writer: Optional[threading.Thread] = None
        self._queue: Optional[queue.Queue] = None
        self._stderr_buf: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._state = "idle"
        self._stats = RecordingStats()
        self._lock = threading.Lock()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        """True if frames are being written (excludes paused/stopped)."""
        return self._state == "recording"

    @property
    def is_open(self) -> bool:
        """True if the file is open (recording or paused)."""
        return self._state in ("recording", "paused")

    @property
    def config(self) -> Optional[RecordingConfig]:
        return self._cfg

    @property
    def stats(self) -> RecordingStats:
        # Refresh duration if open.
        if self._stats.started_at is not None and self.is_open:
            self._stats.duration_s = (
                self._stats.frames_written / max(self._cfg.fps, 1.0)
                if self._cfg else 0.0
            )
        return self._stats

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        *,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc",
        hw_pref: str = "auto",
        bitrate_kbps: Optional[int] = None,
    ) -> RecordingConfig:
        """Spawn ffmpeg and prepare to receive frames.

        Raises :class:`EncoderUnavailable` (or other exceptions) on
        failure — the caller decides whether to abort the session.
        """
        with self._lock:
            if self._state != "idle":
                raise RuntimeError(
                    f"recorder is in state {self._state!r}; stop it first"
                )

            ffmpeg = find_ffmpeg()
            encoder, kind, warnings = select_encoder(codec, hw_pref)

            # ── Format-specific alignment ───────────────────────
            # yuv420p (our output pix_fmt) requires even width AND
            # height. HEVC/H.264 hardware encoders additionally
            # *prefer* (and some require) multiples of 2 for both
            # axes; we round DOWN to the nearest even number.  This
            # applies to every resolution path uniformly so callers
            # don't have to think about it.
            req_w, req_h = int(width), int(height)
            aligned_w = req_w - (req_w % 2)
            aligned_h = req_h - (req_h % 2)
            if aligned_w < 2 or aligned_h < 2:
                raise EncoderUnavailable(
                    f"resolution {req_w}x{req_h} too small after "
                    f"alignment (need ≥ 2×2)"
                )
            if aligned_w != req_w or aligned_h != req_h:
                warnings.append(
                    f"resolution rounded {req_w}x{req_h} → "
                    f"{aligned_w}x{aligned_h} (yuv420p needs even sides)"
                )

            if bitrate_kbps is None:
                bitrate_kbps = default_bitrate_kbps(
                    aligned_w, aligned_h, fps, codec
                )

            out_path = Path(path)
            # Force .mp4 extension if the user gave a bare name.
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cfg = RecordingConfig(
                path=out_path,
                width=aligned_w,
                height=aligned_h,
                fps=float(fps),
                codec=codec,
                encoder=encoder,
                encoder_kind=kind,
                bitrate_kbps=int(bitrate_kbps),
            )

            cmd = self._build_ffmpeg_cmd(ffmpeg, cfg)
            log.info("recording → %s (%s, %s @ %dkbps)",
                     out_path, encoder, f"{width}x{height}@{fps}",
                     bitrate_kbps)
            log.debug("ffmpeg cmd: %s", " ".join(cmd))

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except OSError as exc:
                raise EncoderUnavailable(f"failed to spawn ffmpeg: {exc}")

            self._cfg = cfg
            self._proc = proc
            self._queue = queue.Queue(maxsize=self.QUEUE_DEPTH)
            self._stats = RecordingStats(
                state="recording",
                started_at=time.perf_counter(),
                warnings=list(warnings),
            )
            self._stderr_buf = []
            self._stderr_thread = threading.Thread(
                target=self._stderr_drain, name="locul3d-rec-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            self._writer = threading.Thread(
                target=self._writer_loop, name="locul3d-rec-writer",
                daemon=True,
            )
            self._writer.start()
            self._state = "recording"
            return cfg

    def pause(self) -> None:
        with self._lock:
            if self._state != "recording":
                raise RuntimeError(
                    f"can only pause from 'recording', not {self._state!r}"
                )
            self._state = "paused"
            self._stats.state = "paused"

    def resume(self) -> None:
        with self._lock:
            if self._state != "paused":
                raise RuntimeError(
                    f"can only resume from 'paused', not {self._state!r}"
                )
            self._state = "recording"
            self._stats.state = "recording"

    def stop(self) -> RecordingStats:
        """Flush and close.  Idempotent."""
        with self._lock:
            if self._state == "idle" or self._state == "stopped":
                return self._stats
            self._state = "stopped"
            self._stats.state = "stopping"

        # Sentinel tells the writer to drain and exit.
        try:
            self._queue.put(None, timeout=5.0)
        except Exception:
            pass

        if self._writer is not None:
            self._writer.join(timeout=30.0)

        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5.0)
            ret = self._proc.returncode
            if ret != 0:
                tail = "".join(self._stderr_buf[-20:])
                self._stats.last_error = (
                    f"ffmpeg exited with code {ret}\n{tail}"
                )
                log.error("ffmpeg exited with code %d:\n%s", ret, tail)

        self._stats.state = "stopped"
        if self._cfg is not None:
            try:
                self._stats.bytes_written = self._cfg.path.stat().st_size
            except OSError:
                pass

        # Reset for a future start().
        self._cfg = None
        self._proc = None
        self._writer = None
        self._queue = None
        self._stderr_thread = None
        # Move state back to idle so a new recording can begin.
        self._state = "idle"
        return self._stats

    def abort(self, reason: str) -> RecordingStats:
        """Mark as failed and stop. Used by the engine on render errors."""
        log.error("recording aborted: %s", reason)
        self._stats.last_error = reason
        return self.stop()

    # ── Frame ingestion ──────────────────────────────────────────────

    def feed_frame(self, rgb_bytes: bytes) -> None:
        """Push one frame.

        No-op while paused or stopped.  Blocks if the writer queue is
        full — that's the backpressure mechanism that throttles the
        engine to encoder speed.
        """
        if self._state != "recording":
            return
        if self._queue is None:
            return
        try:
            self._queue.put(rgb_bytes, timeout=60.0)
            self._stats.frames_written += 1
        except queue.Full:
            self._stats.frames_dropped += 1
            log.warning(
                "recorder queue full for 60s — dropping frame %d",
                self._stats.frames_written + self._stats.frames_dropped,
            )

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_ffmpeg_cmd(self, ffmpeg: str, cfg: RecordingConfig) -> List[str]:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", cfg.pix_fmt_in,
            "-s", f"{cfg.width}x{cfg.height}",
            "-r", f"{cfg.fps:g}",
            "-i", "-",
        ]

        # Vertical flip: GL framebuffer is bottom-up; ffmpeg/mp4 wants
        # top-down, so we add a vflip filter.  Cheap on the CPU at
        # 4K because libavfilter is SIMD-vectorised.
        cmd += ["-vf", "vflip"]

        # Encoder selection + sensible defaults per encoder family.
        cmd += ["-c:v", cfg.encoder]

        # Bitrate / quality knobs vary by encoder.
        if cfg.encoder.endswith("_videotoolbox"):
            cmd += ["-b:v", f"{cfg.bitrate_kbps}k",
                    "-allow_sw", "1"]
        elif cfg.encoder.endswith("_nvenc"):
            cmd += ["-rc", "vbr",
                    "-cq", "20",
                    "-b:v", f"{cfg.bitrate_kbps}k",
                    "-maxrate", f"{cfg.bitrate_kbps * 2}k"]
        elif cfg.encoder.endswith("_qsv"):
            cmd += ["-global_quality", "22",
                    "-b:v", f"{cfg.bitrate_kbps}k"]
        elif cfg.encoder.endswith("_amf"):
            cmd += ["-quality", "quality",
                    "-rc", "vbr_peak",
                    "-b:v", f"{cfg.bitrate_kbps}k"]
        elif cfg.encoder.endswith("_vaapi"):
            cmd += ["-b:v", f"{cfg.bitrate_kbps}k"]
        elif cfg.encoder == "libx264":
            cmd += ["-preset", "medium", "-crf", "18"]
        elif cfg.encoder == "libx265":
            cmd += ["-preset", "medium", "-crf", "22",
                    "-x265-params", "log-level=error"]

        # HEVC in mp4: tag the stream as 'hvc1' instead of the
        # libavformat default 'hev1'. QuickTime / Apple Photos /
        # iOS players refuse 'hev1' but accept 'hvc1'.  Other
        # players accept both.  Applies regardless of HW or SW.
        if cfg.codec == "hevc":
            cmd += ["-tag:v", "hvc1"]

        cmd += ["-pix_fmt", cfg.pix_fmt_out,
                "-movflags", "+faststart",
                str(cfg.path)]
        return cmd

    def _writer_loop(self) -> None:
        """Writer thread: pop frames from the queue, write to stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                stdin.write(item)
            except (BrokenPipeError, OSError) as exc:
                self._stats.last_error = f"pipe write failed: {exc}"
                log.error("recorder writer: %s", exc)
                break
        try:
            stdin.flush()
        except Exception:
            pass

    def _stderr_drain(self) -> None:
        """Drain ffmpeg stderr so the pipe doesn't block on a full
        buffer."""
        assert self._proc is not None and self._proc.stderr is not None
        for raw in self._proc.stderr:
            try:
                line = raw.decode("utf-8", "replace")
            except Exception:
                continue
            self._stderr_buf.append(line)
            # Keep tail bounded — encoder warnings can be noisy.
            if len(self._stderr_buf) > 200:
                del self._stderr_buf[: len(self._stderr_buf) - 200]
            log.debug("ffmpeg: %s", line.rstrip())
