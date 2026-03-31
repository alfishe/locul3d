#!/usr/bin/env python3
"""Demo 01 — Warehouse Walkthrough

Demonstrates:
  - Realistic warehouse scene with aligned rack rows, aisles, loading bay
  - Scene correction (rotate + shift to level it)
  - Scripted 25-second camera fly-through with smooth ease_in_out
  - Zone highlight bboxes overlaying the warehouse
  - Viewport settings adjustments (point size, grid, background)

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this demo:     python scripts/demo_01_warehouse_walkthrough.py

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


# ── Warehouse Layout Constants ───────────────────────────────────────

# Warehouse is 40m x 24m, origin at bottom-left corner
W, D = 40.0, 24.0   # width (X), depth (Y)
WALL_H = 6.0         # wall height
RACK_H = 4.0         # rack height
RACK_W = 1.2         # rack width (depth in Y)
RACK_L = 5.0         # rack length (in X)
AISLE_W = 3.0        # aisle width between rack pairs
PAIR_GAP = 0.0       # gap between back-to-back racks

# 3 rack rows, each pair of back-to-back racks
# Row centers at Y = 6, 13, 20  (giving 3m aisle between rows)
ROW_Y = [6.0, 13.0, 20.0]
# Each row has 6 rack segments along X
RACK_X_START = 4.0
RACK_X_GAP = 0.6     # gap between segments

LOADING_BAY_X = (0, 3.5)  # left strip is loading bay
OFFICE_X = (36.5, 40)     # right strip is office area


def main():
    print("Demo 01: Warehouse Walkthrough")
    print("=" * 50)
    wait_server()

    ws = websocket.create_connection(WS_URL, timeout=10)

    try:
        # ── 1. Build the warehouse ───────────────────────────────────

        print("\n[1/6] Building warehouse structure...")

        # Floor — concrete slab (center = midpoint of the quad)
        requests.post(f"{API}/scene/dynamic", json={
            "name": "floor",
            "geometry_type": "surfaces",
            "surfaces": [
                {"axis": "xy", "center": [W / 2, D / 2, 0], "size": [W, D],
                 "color": [0.42, 0.40, 0.37], "opacity": 0.95},
            ],
        })

        # Walls — 4 perimeter walls (center = midpoint of each wall)
        requests.post(f"{API}/scene/dynamic", json={
            "name": "walls",
            "geometry_type": "surfaces",
            "surfaces": [
                # Front wall (Y=0)
                {"axis": "xz", "center": [W / 2, 0, WALL_H / 2], "size": [W, WALL_H],
                 "color": [0.72, 0.70, 0.65], "opacity": 0.45},
                # Back wall (Y=D)
                {"axis": "xz", "center": [W / 2, D, WALL_H / 2], "size": [W, WALL_H],
                 "color": [0.72, 0.70, 0.65], "opacity": 0.45},
                # Left wall (X=0)
                {"axis": "yz", "center": [0, D / 2, WALL_H / 2], "size": [D, WALL_H],
                 "color": [0.68, 0.66, 0.62], "opacity": 0.45},
                # Right wall (X=W)
                {"axis": "yz", "center": [W, D / 2, WALL_H / 2], "size": [D, WALL_H],
                 "color": [0.68, 0.66, 0.62], "opacity": 0.45},
            ],
        })

        # ── 2. Rack rows ────────────────────────────────────────────

        print("[2/6] Placing rack rows (3 rows x 6 segments)...")

        rack_bboxes = []
        rack_colors = [
            [0.30, 0.42, 0.62],  # blue-steel for row 1
            [0.30, 0.50, 0.40],  # green-steel for row 2
            [0.50, 0.38, 0.30],  # rust for row 3
        ]

        for row_idx, row_y in enumerate(ROW_Y):
            color = rack_colors[row_idx]
            for seg in range(6):
                x_center = RACK_X_START + seg * (RACK_L + RACK_X_GAP) + RACK_L / 2

                # Front rack of the pair
                rack_bboxes.append({
                    "label": f"rack_r{row_idx}_s{seg}_front",
                    "center": [x_center, row_y - RACK_W / 2 - PAIR_GAP / 2, RACK_H / 2],
                    "size": [RACK_L, RACK_W, RACK_H],
                    "color": color,
                    "fill_opacity": 0.08,
                })
                # Back rack of the pair
                rack_bboxes.append({
                    "label": f"rack_r{row_idx}_s{seg}_back",
                    "center": [x_center, row_y + RACK_W / 2 + PAIR_GAP / 2, RACK_H / 2],
                    "size": [RACK_L, RACK_W, RACK_H],
                    "color": [c + 0.05 for c in color],
                    "fill_opacity": 0.08,
                })

        requests.post(f"{API}/scene/dynamic", json={
            "name": "racks",
            "geometry_type": "bboxes",
            "bboxes": rack_bboxes,
            "color": [0.35, 0.45, 0.55],
        })
        print(f"  Placed {len(rack_bboxes)} rack segments")

        # ── 3. Loading bay + office area ─────────────────────────────

        print("[3/6] Adding loading bay and office markers...")

        # Loading bay — yellow floor stripe at X=0..3.5
        requests.post(f"{API}/scene/dynamic", json={
            "name": "loading_bay_floor",
            "geometry_type": "surfaces",
            "surfaces": [
                {"axis": "xy", "center": [1.75, D / 2, 0.01], "size": [3.5, D],
                 "color": [0.75, 0.68, 0.25], "opacity": 0.3},
            ],
        })

        # Office area — subtle blue floor stripe at X=36.5..40
        requests.post(f"{API}/scene/dynamic", json={
            "name": "office_floor",
            "geometry_type": "surfaces",
            "surfaces": [
                {"axis": "xy", "center": [38.25, D / 2, 0.01], "size": [3.5, D],
                 "color": [0.25, 0.35, 0.55], "opacity": 0.25},
            ],
        })

        # Pillars — interior structural columns on a 10m grid
        # Placed between rack rows, not on walls
        pillar_bboxes = []
        for px in [10, 20, 30]:
            for py in [3, 9.5, 16.5, 21]:
                pillar_bboxes.append({
                    "label": "pillar",
                    "center": [px, py, WALL_H / 2],
                    "size": [0.4, 0.4, WALL_H],
                    "color": [0.55, 0.53, 0.50],
                    "fill_opacity": 0.15,
                })

        requests.post(f"{API}/scene/dynamic", json={
            "name": "pillars",
            "geometry_type": "bboxes",
            "bboxes": pillar_bboxes,
            "color": [0.55, 0.53, 0.50],
        })

        # ── 4. Simulated LiDAR point cloud on floor/walls ───────────

        print("[4/6] Generating simulated LiDAR scan (8000 points)...")

        pts = []
        colors = []
        # Floor points — dense grid
        for i in range(5000):
            x = (i * 7.31 % W)
            y = (i * 11.73 % D)
            z = 0.0 + 0.02 * math.sin(x * 0.5 + y * 0.3)  # slight noise
            pts.append([x, y, z])
            # Concrete gray with subtle variation
            g = 0.40 + 0.06 * math.sin(x * 0.4) + 0.04 * math.cos(y * 0.5)
            colors.append([int(g * 255), int((g - 0.02) * 255), int((g - 0.05) * 255)])

        # Wall points — scattered on perimeter
        for i in range(3000):
            side = i % 4
            t = (i * 3.71 % 1.0)
            z = (i * 5.17 % WALL_H)
            if side == 0:    # front
                pts.append([t * W, 0.05, z])
            elif side == 1:  # back
                pts.append([t * W, D - 0.05, z])
            elif side == 2:  # left
                pts.append([0.05, t * D, z])
            else:            # right
                pts.append([W - 0.05, t * D, z])
            # Wall color — lighter gray
            g = 0.60 + 0.08 * math.sin(z * 1.2)
            colors.append([int(g * 255), int((g - 0.01) * 255), int((g - 0.03) * 255)])

        requests.post(f"{API}/scene/dynamic", json={
            "name": "lidar_scan",
            "geometry_type": "pointcloud",
            "points": pts,
            "colors": colors,
            "color": [0.5, 0.48, 0.44],
        })

        # ── 5. Zone highlight overlays ───────────────────────────────

        print("[5/6] Adding zone highlights...")

        # Zones tightly wrap around their respective rack rows
        # Each row has racks from X=4 to X=4+6*(5+0.6)-0.6=37.4
        rack_x_min = RACK_X_START - 0.5
        rack_x_max = RACK_X_START + 6 * (RACK_L + RACK_X_GAP) - RACK_X_GAP + 0.5
        rack_zone_w = rack_x_max - rack_x_min
        rack_zone_cx = (rack_x_min + rack_x_max) / 2

        requests.post(f"{API}/scene/dynamic", json={
            "name": "zones",
            "geometry_type": "bboxes",
            "bboxes": [
                {"label": "Loading Bay", "center": [1.75, D / 2, WALL_H / 2],
                 "size": [3.5, D, WALL_H], "color": [1.0, 0.85, 0.15],
                 "fill_opacity": 0.02},
                {"label": "Storage Zone A",
                 "center": [rack_zone_cx, ROW_Y[0], RACK_H / 2],
                 "size": [rack_zone_w, RACK_W * 2 + PAIR_GAP + 1.5, RACK_H + 0.2],
                 "color": [0.15, 0.75, 0.45], "fill_opacity": 0.015},
                {"label": "Storage Zone B",
                 "center": [rack_zone_cx, ROW_Y[1], RACK_H / 2],
                 "size": [rack_zone_w, RACK_W * 2 + PAIR_GAP + 1.5, RACK_H + 0.2],
                 "color": [0.25, 0.55, 0.95], "fill_opacity": 0.015},
                {"label": "Storage Zone C",
                 "center": [rack_zone_cx, ROW_Y[2], RACK_H / 2],
                 "size": [rack_zone_w, RACK_W * 2 + PAIR_GAP + 1.5, RACK_H + 0.2],
                 "color": [0.85, 0.45, 0.15], "fill_opacity": 0.015},
                {"label": "Office", "center": [38.25, D / 2, WALL_H / 2],
                 "size": [3.5, D, WALL_H], "color": [0.4, 0.55, 0.85],
                 "fill_opacity": 0.02},
            ],
        })

        # Configure viewport
        requests.put(f"{API}/viewport", json={
            "point_size": 2,
            "show_grid": False,
            "show_axes": False,
            "bg_color": [0.10, 0.11, 0.14],
        })

        # ── 6. Scripted fly-through ──────────────────────────────────

        print("[6/6] Starting 25-second guided fly-through...")
        print("       (watch the viewer window)\n")

        # Phase A: Establishing shot — high overview descending
        ws_send(ws, "camera.set",
                azimuth=220, elevation=55, distance=55,
                target=[20, 12, 0], fov=45)
        time.sleep(1)

        ws_send(ws, "camera.animate",
                id="flythrough-a",
                keyframes=[
                    {"t": 0.0, "azimuth": 220, "elevation": 55, "distance": 55,
                     "target": [20, 12, 0], "fov": 45},
                    {"t": 0.5, "azimuth": 250, "elevation": 40, "distance": 40,
                     "target": [20, 12, 1], "fov": 48},
                    {"t": 1.0, "azimuth": 270, "elevation": 25, "distance": 28,
                     "target": [15, 9, 2], "fov": 52},
                ],
                duration_ms=7000,
                easing="ease_in_out")
        print("  Phase A: Establishing overview (7s)")
        time.sleep(7.5)

        # Phase B: Fly into aisle between row 1 and row 2
        ws_send(ws, "camera.animate",
                id="flythrough-b",
                keyframes=[
                    {"t": 0.0, "azimuth": 270, "elevation": 25, "distance": 28,
                     "target": [15, 9, 2], "fov": 52},
                    {"t": 0.3, "azimuth": 275, "elevation": 12, "distance": 15,
                     "target": [10, 9.5, 2], "fov": 60},
                    {"t": 0.7, "azimuth": 280, "elevation": 8, "distance": 10,
                     "target": [25, 9.5, 2], "fov": 65},
                    {"t": 1.0, "azimuth": 290, "elevation": 15, "distance": 14,
                     "target": [32, 9.5, 2], "fov": 58},
                ],
                duration_ms=8000,
                easing="ease_in_out")
        print("  Phase B: Fly through aisle (8s)")
        time.sleep(8.5)

        # Phase C: Sweep across loading bay
        ws_send(ws, "camera.animate",
                id="flythrough-c",
                keyframes=[
                    {"t": 0.0, "azimuth": 290, "elevation": 15, "distance": 14,
                     "target": [32, 9.5, 2], "fov": 58},
                    {"t": 0.4, "azimuth": 330, "elevation": 20, "distance": 20,
                     "target": [20, 12, 1], "fov": 50},
                    {"t": 0.7, "azimuth": 10, "elevation": 18, "distance": 18,
                     "target": [5, 12, 2], "fov": 55},
                    {"t": 1.0, "azimuth": 40, "elevation": 35, "distance": 30,
                     "target": [2, 12, 1], "fov": 48},
                ],
                duration_ms=6000,
                easing="ease_in_out")
        print("  Phase C: Sweep to loading bay (6s)")
        time.sleep(6.5)

        # Phase D: Pull back to final overview
        ws_send(ws, "camera.animate",
                id="flythrough-d",
                keyframes=[
                    {"t": 0.0, "azimuth": 40, "elevation": 35, "distance": 30,
                     "target": [2, 12, 1], "fov": 48},
                    {"t": 0.5, "azimuth": 80, "elevation": 42, "distance": 42},
                    {"t": 1.0, "azimuth": 120, "elevation": 50, "distance": 52,
                     "target": [20, 12, 0], "fov": 42},
                ],
                duration_ms=5000,
                easing="ease_out")
        print("  Phase D: Pull back to overview (5s)")
        time.sleep(5.5)

        print("\nFly-through complete!")
        print("\nPress Enter to clean up and exit...")
        try:
            input()
        except EOFError:
            time.sleep(2)

        # Cleanup
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
