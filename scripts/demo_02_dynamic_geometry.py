#!/usr/bin/env python3
"""Demo 02 — Dynamic Geometry & Real-Time Streaming

Demonstrates:
  - Creating STL-like triangle mesh cubes dynamically
  - 60 Hz geometry updates from a Python loop (simulating live data)
  - Continuous camera orbit running server-side at 125 Hz
  - Multiple objects animating independently
  - Binary point streaming for high-throughput cloud updates

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this demo:     python scripts/demo_02_dynamic_geometry.py

Requires: pip install websocket-client requests numpy
"""

import json
import math
import struct
import sys
import time

try:
    import requests
    import websocket
    import numpy as np
except ImportError:
    print("ERROR: pip install requests websocket-client numpy")
    sys.exit(1)

API = "http://localhost:8350/api/v1"
WS_URL = "ws://localhost:8350/ws"


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


def make_cube_verts(cx, cy, cz, sx, sy, sz):
    """Generate 8 vertices and 12 triangles for a box."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ]
    tris = [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5], [2, 7, 3], [2, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ]
    return verts, tris


def main():
    print("Demo 02: Dynamic Geometry & Real-Time Streaming")
    print("=" * 50)
    wait_server()

    # Dark background for contrast
    requests.put(f"{API}/viewport", json={
        "point_size": 4,
        "show_grid": True,
        "show_axes": True,
        "bg_color": [0.08, 0.09, 0.12],
    })

    ws = websocket.create_connection(WS_URL, timeout=10)

    try:
        # ── 1. Create base scene — a grid of pillars ─────────────────

        print("\n[1/5] Creating pillar grid...")
        pillars = []
        for ix in range(4):
            for iy in range(4):
                pillars.append({
                    "label": f"pillar_{ix}_{iy}",
                    "center": [ix * 4, iy * 4, 1.5],
                    "size": [0.6, 0.6, 3.0],
                    "color": [0.45, 0.42, 0.55],
                    "fill_opacity": 0.1,
                })
        requests.post(f"{API}/scene/dynamic", json={
            "name": "pillars",
            "geometry_type": "bboxes",
            "bboxes": pillars,
            "color": [0.45, 0.42, 0.55],
        })

        # Floor surface
        requests.post(f"{API}/scene/dynamic", json={
            "name": "ground",
            "geometry_type": "surfaces",
            "surfaces": [
                {"axis": "xy", "center": [8, 8, 0], "size": [20, 20],
                 "color": [0.25, 0.25, 0.3], "opacity": 0.5},
            ],
        })

        # ── 2. Create the animated cube ──────────────────────────────

        print("[2/5] Creating animated mesh cube...")
        verts, tris = make_cube_verts(6, 6, 1, 2, 2, 2)
        requests.post(f"{API}/scene/dynamic", json={
            "name": "stretching_cube",
            "geometry_type": "mesh",
            "vertices": verts,
            "triangles": tris,
            "color": [1.0, 0.35, 0.1],
            "opacity": 0.85,
        })

        # Set camera
        ws_send(ws, "camera.set",
                azimuth=30, elevation=25, distance=25,
                target=[6, 6, 2], fov=50)

        # ── 3. Start camera orbit ────────────────────────────────────

        print("[3/5] Starting 10 deg/sec camera orbit (server-side 125Hz)...")
        ws_send(ws, "camera.transform_continuous",
                id="orbit", property="azimuth", rate=10.0, duration_ms=0)

        # ── 4. 60Hz geometry updates — stretching cube ───────────────

        print("[4/5] Streaming 60Hz geometry updates for 10 seconds...")
        print("       (cube stretches along Z, color shifts)")

        frame_count = 600  # 10 seconds at 60fps
        target_dt = 1.0 / 60.0
        times = []

        for frame in range(frame_count):
            t0 = time.perf_counter()
            t = frame * target_dt

            # Oscillating Z scale
            z_scale = 2.0 + 3.0 * (0.5 + 0.5 * math.sin(t * 1.5))
            z_offset = z_scale / 2.0

            # Generate cube with dynamic Z
            verts, tris = make_cube_verts(6, 6, z_offset, 2, 2, z_scale)

            # Color shifts with time
            r = 0.5 + 0.5 * math.sin(t * 0.8)
            g = 0.5 + 0.5 * math.sin(t * 0.8 + 2.1)
            b = 0.5 + 0.5 * math.sin(t * 0.8 + 4.2)

            ws_send(ws, "dynamic.update",
                    layer_id="dyn_stretching_cube",
                    vertices=verts,
                    triangles=tris,
                    color=[r, g, b])

            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Progress indicator every 2 seconds
            if frame > 0 and frame % 120 == 0:
                avg = sum(times[-120:]) / 120 * 1000
                print(f"       frame {frame}/{frame_count}, avg {avg:.1f}ms/frame")

        avg_ms = sum(times) / len(times) * 1000
        print(f"\n  Geometry update stats:")
        print(f"    Frames:       {frame_count}")
        print(f"    Avg latency:  {avg_ms:.1f} ms")
        print(f"    Max latency:  {max(times) * 1000:.1f} ms")

        # ── 5. Binary point streaming burst ──────────────────────────

        print("\n[5/5] Binary point streaming — 50K points in 5 bursts...")

        layer_name = "stream_cloud"
        name_bytes = layer_name.encode("utf-8")

        # Create initial cloud
        pts = np.random.randn(10_000, 3).astype(np.float32) * 2.0
        pts[:, 0] += 0  # centered at origin
        pts[:, 2] = np.abs(pts[:, 2])  # above ground

        header = struct.pack("<II", 1, len(name_bytes))
        ws.send(header + name_bytes + pts.tobytes(),
                opcode=websocket.ABNF.OPCODE_BINARY)
        while True:
            resp = json.loads(ws.recv())
            if not resp.get("type", "").startswith("event."):
                break
        print(f"    Created: 10,000 points")

        for burst in range(4):
            pts = np.random.randn(10_000, 3).astype(np.float32) * 2.0
            pts[:, 2] = np.abs(pts[:, 2]) + burst * 1.5

            header = struct.pack("<II", 2, len(name_bytes))
            ws.send(header + name_bytes + pts.tobytes(),
                    opcode=websocket.ABNF.OPCODE_BINARY)
            while True:
                resp = json.loads(ws.recv())
                if not resp.get("type", "").startswith("event."):
                    count = resp.get("data", {}).get("point_count", "?")
                    break
            print(f"    Appended burst {burst + 1}: total = {count}")
            time.sleep(0.2)

        # Let it orbit for a few more seconds
        print("\n  Orbiting for 5 more seconds...")
        time.sleep(5)

        # Stop orbit
        ws_send(ws, "transform.stop", id="orbit")

        print("\nPress Enter to clean up...")
        try:
            input()
        except EOFError:
            time.sleep(2)

        ws_send(ws, "dynamic.clear")
        requests.put(f"{API}/viewport", json={
            "bg_color": [0.18, 0.2, 0.24],
        })

    finally:
        ws.close()

    print("Done.")


if __name__ == "__main__":
    main()
