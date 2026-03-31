#!/usr/bin/env python3
"""Demo 03 — Color Pulse & Easing Showcase

Demonstrates:
  - Multiple easing curves applied to different objects
  - Spring physics for bouncy bbox appearance
  - Simultaneous camera + object animations
  - Color/opacity keyframe animation on multiple layers
  - Layer visibility toggling for staggered reveal

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this demo:     python scripts/demo_03_color_pulse_and_easing.py

Requires: pip install websocket-client requests
"""

import json
import math
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


# Color palette
CORAL     = [1.0, 0.42, 0.33]
TEAL      = [0.2, 0.84, 0.75]
GOLD      = [1.0, 0.82, 0.2]
VIOLET    = [0.68, 0.35, 0.95]
LIME      = [0.55, 0.95, 0.2]
SKY       = [0.3, 0.65, 1.0]
ROSE      = [0.95, 0.3, 0.55]
MINT      = [0.3, 0.95, 0.65]


def main():
    print("Demo 03: Color Pulse & Easing Showcase")
    print("=" * 50)
    wait_server()

    requests.put(f"{API}/viewport", json={
        "point_size": 3,
        "show_grid": True,
        "show_axes": False,
        "bg_color": [0.06, 0.07, 0.1],
    })

    ws = websocket.create_connection(WS_URL, timeout=10)

    try:
        # ── 1. Create a ring of colored boxes ────────────────────────

        print("\n[1/6] Creating ring of 8 colored boxes...")

        colors = [CORAL, TEAL, GOLD, VIOLET, LIME, SKY, ROSE, MINT]
        names = ["coral", "teal", "gold", "violet", "lime", "sky", "rose", "mint"]

        for i, (name, color) in enumerate(zip(names, colors)):
            angle = i * (2 * math.pi / 8)
            x = 6 * math.cos(angle)
            y = 6 * math.sin(angle)
            requests.post(f"{API}/scene/dynamic", json={
                "name": f"box_{name}",
                "geometry_type": "bboxes",
                "bboxes": [
                    {"label": name, "center": [x, y, 0.75],
                     "size": [1.5, 1.5, 1.5], "color": color,
                     "fill_opacity": 0.15},
                ],
                "color": color,
                "visible": False,  # start hidden
            })

        # Center sphere (point cloud)
        pts = []
        pt_colors = []
        for i in range(3000):
            # Fibonacci sphere
            phi = math.acos(1 - 2 * (i + 0.5) / 3000)
            theta = math.pi * (1 + 5**0.5) * i
            r = 1.5
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi) + 2.0
            pts.append([x, y, z])
            # Rainbow based on angle
            hue = (theta / (2 * math.pi)) % 1.0
            pt_colors.append([
                int(255 * (0.5 + 0.5 * math.sin(hue * 6.28))),
                int(255 * (0.5 + 0.5 * math.sin(hue * 6.28 + 2.09))),
                int(255 * (0.5 + 0.5 * math.sin(hue * 6.28 + 4.19))),
            ])

        requests.post(f"{API}/scene/dynamic", json={
            "name": "center_sphere",
            "geometry_type": "pointcloud",
            "points": pts,
            "colors": pt_colors,
            "color": [0.8, 0.7, 1.0],
        })

        # Set camera
        ws_send(ws, "camera.set",
                azimuth=0, elevation=30, distance=20,
                target=[0, 0, 1.5], fov=50)
        time.sleep(0.5)

        # ── 2. Staggered reveal with different easings ───────────────

        print("[2/6] Staggered box reveal (different easing per box)...")

        easings = [
            "ease_out",
            "ease_out_back",        # overshoots!
            {"spring": {"damping": 0.4, "stiffness": 150}},  # bouncy
            "ease_out_cubic",
            "ease_out_bounce",      # bounce settle
            {"spring": {"damping": 0.6, "stiffness": 100}},
            "ease_in_out",
            "ease_out_back",
        ]

        for i, name in enumerate(names):
            layer_id = f"dyn_box_{name}"
            # Make visible
            requests.patch(f"{API}/scene/dynamic/{layer_id}", json={
                "visible": True, "opacity": 1.0,
            })
            time.sleep(0.25)

        print("  All 8 boxes revealed")

        # ── 3. Start camera orbit ────────────────────────────────────

        print("[3/6] Starting slow camera orbit...")
        ws_send(ws, "camera.transform_continuous",
                id="slow-orbit", property="azimuth",
                rate=8.0, duration_ms=0)

        time.sleep(2)

        # ── 4. Color pulse animation on all boxes ────────────────────

        print("[4/6] Starting color/opacity pulse on each box...")

        for i, (name, color) in enumerate(zip(names, colors)):
            layer_id = f"dyn_box_{name}"
            # Each box pulses between its color and white
            ws_send(ws, "dynamic.animate",
                    id=f"pulse-{name}",
                    layer_id=layer_id,
                    keyframes=[
                        {"t": 0.0, "color": color, "opacity": 0.9},
                        {"t": 0.5, "color": [1.0, 1.0, 1.0], "opacity": 0.4},
                        {"t": 1.0, "color": color, "opacity": 0.9},
                    ],
                    duration_ms=2000 + i * 300,  # staggered periods
                    loop=True,
                    ping_pong=True,
                    easing="ease_in_out")

        time.sleep(6)

        # ── 5. Camera zoom with spring easing ────────────────────────

        print("[5/6] Spring-eased camera zoom burst...")

        ws_send(ws, "camera.animate",
                id="zoom-spring",
                keyframes=[
                    {"t": 0.0, "distance": 20, "elevation": 30},
                    {"t": 1.0, "distance": 10, "elevation": 15},
                ],
                duration_ms=2000,
                easing={"spring": {"damping": 0.5, "stiffness": 120}})

        time.sleep(3)

        # Pull back with ease_out_back
        ws_send(ws, "camera.animate",
                id="zoom-back",
                keyframes=[
                    {"t": 0.0, "distance": 10, "elevation": 15},
                    {"t": 1.0, "distance": 22, "elevation": 35},
                ],
                duration_ms=3000,
                easing="ease_out_back")

        time.sleep(4)

        # ── 6. Continuous color transitions ──────────────────────────

        print("[6/6] Continuous color shifts for 8 seconds...")

        # Fade center sphere color through spectrum
        ws_send(ws, "dynamic.transform_continuous",
                id="sphere-color", layer_id="dyn_center_sphere",
                property="color", target=[1.0, 0.2, 0.8],
                duration_ms=4000)

        time.sleep(4)

        ws_send(ws, "dynamic.transform_continuous",
                id="sphere-color2", layer_id="dyn_center_sphere",
                property="color", target=[0.2, 1.0, 0.5],
                duration_ms=4000)

        time.sleep(5)

        # Stop everything
        print("\nStopping all animations...")
        ws_send(ws, "transform.stop_all")

        print("\nPress Enter to clean up...")
        try:
            input()
        except EOFError:
            time.sleep(2)

        ws_send(ws, "dynamic.clear")
        requests.put(f"{API}/viewport", json={
            "show_axes": True,
            "bg_color": [0.18, 0.2, 0.24],
        })

    finally:
        ws.close()

    print("Done.")


if __name__ == "__main__":
    main()
