#!/usr/bin/env python3
"""Demo 04 — High-Resolution Frame Capture

Demonstrates:
  - Switching to capture mode (full quality, no LOD)
  - Frame-by-frame rendering via render.capture_frame
  - Saving frames to disk as PNG sequence
  - Camera animation advancing per-frame for deterministic output
  - Switching back to realtime mode

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this demo:     python scripts/demo_04_high_res_capture.py

Requires: pip install websocket-client requests
Output: Frames saved to ./capture_frames/ (created automatically)
"""

import json
import math
import os
import sys
import time

try:
    import requests
    import websocket
except ImportError:
    print("ERROR: pip install requests websocket-client")
    sys.exit(1)

API = "http://localhost:8350/api/v1"
WS_URL = "ws://localhost:8350/ws"

FRAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capture_frames")


def ws_send(ws, msg_type, **kwargs):
    payload = {"type": msg_type, **kwargs}
    ws.send(json.dumps(payload))
    while True:
        resp = json.loads(ws.recv())
        if not resp.get("type", "").startswith("event."):
            return resp.get("data", resp)


def wait_server():
    for _ in range(20):
        try:
            r = requests.get(f"{API}/system/ping", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    print("ERROR: Locul3D not running on port 8350")
    sys.exit(1)


def main():
    print("Demo 04: High-Resolution Frame Capture")
    print("=" * 50)
    wait_server()

    os.makedirs(FRAME_DIR, exist_ok=True)
    print(f"  Output directory: {os.path.abspath(FRAME_DIR)}")

    ws = websocket.create_connection(WS_URL, timeout=30)

    try:
        # ── 1. Build a colorful scene ────────────────────────────────

        print("\n[1/5] Building capture scene...")

        # Create a spiral staircase of boxes
        for i in range(24):
            angle = i * (2 * math.pi / 12)
            r = 4.0
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = i * 0.4

            hue = i / 24.0
            color = [
                0.5 + 0.5 * math.sin(hue * 6.28),
                0.5 + 0.5 * math.sin(hue * 6.28 + 2.09),
                0.5 + 0.5 * math.sin(hue * 6.28 + 4.19),
            ]

            requests.post(f"{API}/scene/dynamic", json={
                "name": f"step_{i:02d}",
                "geometry_type": "bboxes",
                "bboxes": [
                    {"label": f"step_{i}", "center": [x, y, z],
                     "size": [1.5, 0.8, 0.3],
                     "color": color,
                     "rotation_z": math.degrees(angle),
                     "fill_opacity": 0.2},
                ],
                "color": color,
            })

        # Central column
        requests.post(f"{API}/scene/dynamic", json={
            "name": "column",
            "geometry_type": "bboxes",
            "bboxes": [
                {"label": "column", "center": [0, 0, 5],
                 "size": [0.6, 0.6, 10],
                 "color": [0.7, 0.65, 0.8],
                 "fill_opacity": 0.1},
            ],
            "color": [0.7, 0.65, 0.8],
        })

        # Floor
        requests.post(f"{API}/scene/dynamic", json={
            "name": "floor",
            "geometry_type": "surfaces",
            "surfaces": [
                {"axis": "xy", "center": [0, 0, 0], "size": [12, 12],
                 "color": [0.2, 0.2, 0.25], "opacity": 0.7},
            ],
        })

        print(f"  Created spiral staircase (24 steps + column + floor)")

        # Configure viewport
        requests.put(f"{API}/viewport", json={
            "point_size": 3,
            "show_grid": False,
            "show_axes": False,
            "bg_color": [0.05, 0.05, 0.08],
        })

        # ── 2. Set up camera for capture ─────────────────────────────

        print("[2/5] Setting initial camera...")
        ws_send(ws, "camera.set",
                azimuth=0, elevation=25, distance=18,
                target=[0, 0, 4], fov=50)
        time.sleep(0.5)

        # ── 3. Switch to capture mode ────────────────────────────────

        print("[3/5] Switching to capture mode...")
        result = ws_send(ws, "render.get_mode")
        print(f"  Current mode: {result.get('mode', '?')}")

        # Note: we don't resize the viewport — capture at current resolution

        # ── 4. Capture frame sequence ────────────────────────────────

        TARGET_FPS = 30
        DURATION = 4.0  # seconds
        total_frames = int(TARGET_FPS * DURATION)

        print(f"[4/5] Capturing {total_frames} frames ({DURATION}s at {TARGET_FPS}fps)...")
        print(f"       Camera orbits 360 degrees via smooth animation")

        ws_send(ws, "render.set_target_fps", fps=TARGET_FPS)

        # Start a smooth camera orbit via animation engine
        # (much smoother than per-frame camera.set)
        ws_send(ws, "camera.animate",
                id="capture-orbit",
                keyframes=[
                    {"t": 0.0, "azimuth": 0, "elevation": 25, "distance": 18},
                    {"t": 0.25, "azimuth": 90, "elevation": 35},
                    {"t": 0.5, "azimuth": 180, "elevation": 25},
                    {"t": 0.75, "azimuth": 270, "elevation": 15},
                    {"t": 1.0, "azimuth": 360, "elevation": 25, "distance": 18},
                ],
                duration_ms=int(DURATION * 1000),
                easing="linear")

        capture_times = []
        saved_count = 0

        frame_dir_abs = os.path.abspath(FRAME_DIR).replace("\\", "/")

        for i in range(total_frames):
            t0 = time.perf_counter()

            # Capture frame — animation engine advances automatically per tick
            frame_path = f"{frame_dir_abs}/frame_{i:04d}.png"
            result = ws_send(ws, "render.capture_frame",
                             save_to=frame_path, format="png")

            if isinstance(result, dict) and result.get("status") == "ok":
                saved_count += 1

            elapsed = time.perf_counter() - t0
            capture_times.append(elapsed)

            # Progress every 30 frames
            if (i + 1) % 30 == 0:
                avg = sum(capture_times[-30:]) / 30 * 1000
                print(f"       Frame {i + 1}/{total_frames} "
                      f"({avg:.0f}ms/frame)")

        avg_ms = sum(capture_times) / len(capture_times) * 1000
        print(f"\n  Capture stats:")
        print(f"    Frames captured: {saved_count}/{total_frames}")
        print(f"    Avg frame time:  {avg_ms:.0f} ms")
        print(f"    Total time:      {sum(capture_times):.1f} s")
        if saved_count > 0:
            size = os.path.getsize(os.path.join(FRAME_DIR, "frame_0000.png"))
            print(f"    Frame size:      ~{size // 1024} KB each")

        # ── 5. Switch back to realtime ───────────────────────────────

        print("\n[5/5] Switching back to realtime mode...")
        ws_send(ws, "render.set_mode", mode="realtime")

        print(f"\nFrames saved to: {os.path.abspath(FRAME_DIR)}/")
        print("To assemble video:")
        print(f"  ffmpeg -framerate {TARGET_FPS} -i {FRAME_DIR}/frame_%04d.png "
              f"-c:v libx264 -pix_fmt yuv420p demo.mp4")

        print("\nPress Enter to clean up...")
        try:
            input()
        except EOFError:
            time.sleep(2)

        ws_send(ws, "dynamic.clear")
        requests.put(f"{API}/viewport", json={
            "show_grid": True,
            "show_axes": True,
            "bg_color": [0.18, 0.2, 0.24],
        })

    finally:
        ws.close()

    print("Done.")


if __name__ == "__main__":
    main()
