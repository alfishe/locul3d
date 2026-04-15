"""POC: Qt QOpenGLWidget + reliable frame capture + video encoding.

Renders a simple animated 3D scene (spinning colored cube on a grid)
in a QOpenGLWidget, captures every frame via grabFramebuffer(), and
pipes raw RGB24 to an ffmpeg subprocess to produce a real video.

Run one encoder:
    python capture_poc.py --encoder h264_nvenc
    python capture_poc.py --encoder libx264 --seconds 3

Run ALL encoders sequentially (prints a summary table):
    python capture_poc.py --all

Requirements: PySide6, PyOpenGL, ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING, GL_LINES,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_PROJECTION, GL_QUADS,
    GL_SRC_ALPHA, GL_AMBIENT_AND_DIFFUSE,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glColorMaterial, glDisable, glEnable, glEnd, glLightfv, glLoadIdentity,
    glMatrixMode, glNormal3f, glPopMatrix, glPushMatrix, glRotatef,
    glTranslatef, glVertex3f, glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective

OUTPUT_DIR = Path(__file__).parent / "out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── The animated OpenGL widget ──────────────────────────────────────


class SceneWidget(QOpenGLWidget):
    """Draws a rotating colored cube on a grid at a fixed camera."""

    def __init__(self) -> None:
        # QSurfaceFormat with no MSAA — matches what grabFramebuffer reads
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)  # widget MSAA is fine; grabFramebuffer handles it
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__()
        self._angle = 0.0
        self.setMinimumSize(640, 480)

    def tick(self, dt_frame: float) -> None:
        """Advance animation by one deterministic frame step."""
        self._angle = (self._angle + dt_frame * 90.0) % 360.0

    def initializeGL(self) -> None:
        glClearColor(0.08, 0.10, 0.14, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, 0x1200, [0.3, 0.3, 0.3, 1.0])  # GL_AMBIENT
        glLightfv(GL_LIGHT0, 0x1201, [0.9, 0.9, 0.9, 1.0])  # GL_DIFFUSE

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w, h = max(1, self.width()), max(1, self.height())
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(6, 4, 8, 0, 0, 0, 0, 1, 0)

        glLightfv(GL_LIGHT0, 0x1203, [4.0, 6.0, 8.0, 1.0])  # GL_POSITION

        self._draw_grid()
        glPushMatrix()
        glRotatef(self._angle, 0, 1, 0)
        self._draw_cube()
        glPopMatrix()

    def _draw_grid(self) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.4)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, 0, -10)
            glVertex3f(i, 0, 10)
            glVertex3f(-10, 0, i)
            glVertex3f(10, 0, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_cube(self) -> None:
        # Six faces, each a different color.
        faces = [
            ((1.0, 0.3, 0.3), ( 1, 0, 0), [( 1,-1,-1), ( 1, 1,-1), ( 1, 1, 1), ( 1,-1, 1)]),
            ((0.3, 1.0, 0.3), (-1, 0, 0), [(-1,-1, 1), (-1, 1, 1), (-1, 1,-1), (-1,-1,-1)]),
            ((0.3, 0.3, 1.0), ( 0, 1, 0), [(-1, 1,-1), (-1, 1, 1), ( 1, 1, 1), ( 1, 1,-1)]),
            ((1.0, 1.0, 0.3), ( 0,-1, 0), [(-1,-1, 1), (-1,-1,-1), ( 1,-1,-1), ( 1,-1, 1)]),
            ((1.0, 0.3, 1.0), ( 0, 0, 1), [(-1,-1, 1), ( 1,-1, 1), ( 1, 1, 1), (-1, 1, 1)]),
            ((0.3, 1.0, 1.0), ( 0, 0,-1), [( 1,-1,-1), (-1,-1,-1), (-1, 1,-1), ( 1, 1,-1)]),
        ]
        glBegin(GL_QUADS)
        for color, normal, verts in faces:
            glColor4f(*color, 1.0)
            glNormal3f(*normal)
            for v in verts:
                glVertex3f(*v)
        glEnd()


# ── Frame capture: the reliable path ─────────────────────────────────


def grab_rgb24(widget: SceneWidget, out_w: int, out_h: int) -> bytes:
    """Capture the widget's current frame as RGB24 bytes at *out_w × out_h*.

    Uses ``grabFramebuffer()`` which triggers a real paintGL cycle
    inside Qt's internal FBO — the rendering that the user actually
    sees on screen.  No custom FBO, no double render.

    CRITICAL: QImage rows are padded to 4-byte alignment.  For
    ``Format_RGB888`` with an odd width, ``bytesPerLine()`` exceeds
    ``3 * width``, so reading ``constBits()`` as a flat ``w*h*3``
    byte buffer yields misaligned garbage after the first row.  We
    pass the stride to Pillow via the ``raw`` decoder.

    The returned bytes are TOP-DOWN (QImage convention); feed them
    to ffmpeg WITHOUT a ``vflip`` filter.
    """
    from PIL import Image

    qimg = widget.grabFramebuffer()
    qimg = qimg.convertToFormat(qimg.Format.Format_RGB888)
    iw, ih = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()  # stride incl. padding
    ptr = qimg.constBits()
    raw = ptr.tobytes() if hasattr(ptr, "tobytes") else bytes(ptr)

    # frombuffer with explicit stride handles the row padding.
    img = Image.frombuffer("RGB", (iw, ih), raw, "raw", "RGB", bpl, 1)
    if iw != out_w or ih != out_h:
        img = img.resize((out_w, out_h), Image.Resampling.BOX)
    return img.tobytes()


# ── ffmpeg encoder management ────────────────────────────────────────


def list_encoders() -> set:
    """Parse ``ffmpeg -encoders`` and return the set of video encoder
    names available in the current ffmpeg build.
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, timeout=10,
    )
    encoders = set()
    in_table = False
    for line in out.stdout.splitlines():
        if line.strip().startswith("------"):
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        flags = stripped.split(None, 1)[0]
        if not flags or flags[0] != "V":
            continue
        rest = stripped[len(flags):].strip()
        name = rest.split(None, 1)[0] if rest else ""
        if name:
            encoders.add(name)
    return encoders


def build_ffmpeg_cmd(
    encoder: str, width: int, height: int, fps: int,
    bitrate_kbps: int, out_path: Path,
) -> List[str]:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", encoder,
    ]

    if encoder.endswith("_nvenc"):
        cmd += ["-rc", "vbr", "-cq", "20",
                "-b:v", f"{bitrate_kbps}k",
                "-maxrate", f"{bitrate_kbps * 2}k"]
    elif encoder.endswith("_qsv"):
        cmd += ["-global_quality", "22",
                "-b:v", f"{bitrate_kbps}k"]
    elif encoder == "libx264":
        cmd += ["-preset", "medium", "-crf", "18"]
    elif encoder == "libx265":
        cmd += ["-preset", "medium", "-crf", "22",
                "-x265-params", "log-level=error"]

    if encoder.startswith("hevc") or encoder == "libx265":
        cmd += ["-tag:v", "hvc1"]

    cmd += ["-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path)]
    return cmd


# ── Recording driver ────────────────────────────────────────────────


def record_with_encoder(
    encoder: str, width: int, height: int, fps: int, seconds: float,
) -> dict:
    """Run one recording session and return stats."""

    out_path = OUTPUT_DIR / f"poc_{encoder}.mp4"
    n_frames = int(round(fps * seconds))

    # Rough bitrate target: 15 Mbps per Mpx at 30fps.
    mpx = (width * height) / 1_000_000
    bitrate_kbps = int(mpx * (fps / 30.0) * 15000)

    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    widget = SceneWidget()
    window.setCentralWidget(widget)
    window.resize(width, height)
    window.show()

    # Pump the event loop until the widget is exposed and GL is ready.
    deadline = time.perf_counter() + 3.0
    while not widget.isValid() and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)
    # One initial paint to settle the framebuffer size.
    widget.update()
    app.processEvents()

    cmd = build_ffmpeg_cmd(encoder, width, height, fps, bitrate_kbps, out_path)
    print(f"  ffmpeg cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    stderr_lines: List[str] = []
    import threading

    def drain_stderr() -> None:
        for raw in proc.stderr:
            try:
                stderr_lines.append(raw.decode("utf-8", "replace"))
            except Exception:
                pass

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    frame_dt = 1.0 / fps
    t_start = time.perf_counter()

    try:
        for i in range(n_frames):
            widget.tick(frame_dt)
            # Let Qt repaint the widget at its real size.
            widget.update()
            app.processEvents()

            # Capture at target resolution.
            rgb = grab_rgb24(widget, width, height)
            if len(rgb) != width * height * 3:
                return {
                    "encoder": encoder, "ok": False,
                    "error": f"bad frame size {len(rgb)} "
                             f"(expected {width*height*3})",
                    "frames": i,
                }

            try:
                proc.stdin.write(rgb)
            except BrokenPipeError as e:
                return {
                    "encoder": encoder, "ok": False,
                    "error": f"broken pipe at frame {i}: {e}",
                    "frames": i,
                    "ffmpeg_err": "".join(stderr_lines[-5:]),
                }
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=30)

    t_end = time.perf_counter()
    wall_s = t_end - t_start
    file_size = out_path.stat().st_size if out_path.exists() else 0
    ret = proc.returncode

    window.close()
    app.processEvents()

    return {
        "encoder": encoder,
        "ok": ret == 0 and file_size > 0,
        "returncode": ret,
        "frames": n_frames,
        "file_size": file_size,
        "wall_s": wall_s,
        "path": str(out_path),
        "bitrate_target_kbps": bitrate_kbps,
        "ffmpeg_err": "".join(stderr_lines[-10:]) if ret != 0 else "",
    }


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None,
                    help="Specific encoder (e.g. h264_nvenc, libx264).")
    ap.add_argument("--all", action="store_true",
                    help="Try all encoders in order.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not on PATH")
        return 1

    avail = list_encoders()
    print(f"ffmpeg encoders found: {len(avail)}")

    if args.all:
        # Priority order matching locul3d: NVENC → QSV → software.
        candidates = [
            "h264_nvenc", "hevc_nvenc",
            "h264_qsv", "hevc_qsv",
            "libx264", "libx265",
        ]
        encoders = [e for e in candidates if e in avail]
    elif args.encoder:
        if args.encoder not in avail:
            print(f"ERROR: encoder {args.encoder!r} not in this "
                  f"ffmpeg build")
            return 1
        encoders = [args.encoder]
    else:
        encoders = ["libx264"]

    print(f"Recording {args.width}x{args.height} @ {args.fps}fps "
          f"for {args.seconds}s with: {encoders}\n")

    results = []
    for enc in encoders:
        print(f"═══ {enc} ═══")
        res = record_with_encoder(
            enc, args.width, args.height, args.fps, args.seconds,
        )
        results.append(res)
        mb = res.get("file_size", 0) / 1024 / 1024
        if res["ok"]:
            print(f"  OK: {res['frames']} frames, {mb:.2f} MB, "
                  f"{res['wall_s']:.1f}s wall")
        else:
            print(f"  FAILED: rc={res.get('returncode')} "
                  f"err={res.get('error')}")
            if res.get("ffmpeg_err"):
                print(f"  ffmpeg stderr tail:\n{res['ffmpeg_err']}")
        print()

    # Summary
    print("═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"{'Encoder':<18} {'OK':<4} {'Frames':<8} {'Size(MB)':<10} {'Wall(s)':<8}")
    print("-" * 70)
    for r in results:
        ok = "YES" if r["ok"] else "NO"
        mb = r.get("file_size", 0) / 1024 / 1024
        print(f"{r['encoder']:<18} {ok:<4} {r.get('frames',0):<8} "
              f"{mb:<10.2f} {r.get('wall_s',0):<8.1f}")

    return 0 if all(r["ok"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
