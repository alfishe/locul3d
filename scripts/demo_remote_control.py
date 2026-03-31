#!/usr/bin/env python3
"""Remote Control API Demo — dynamic geometry, animation, and high-FPS streaming.

Demonstrates:
  1. Dynamic layer CRUD (pointcloud, mesh, bboxes, surfaces)
  2. Real-time geometry updates at 60 Hz via WebSocket
  3. Camera animation and continuous transforms
  4. Color/opacity animation on dynamic layers
  5. Binary point streaming (high-throughput)
  6. Event broadcasting (listening for server-pushed events)

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this demo:     python scripts/demo_remote_control.py

Requires:
    pip install websocket-client requests
"""

import json
import math
import struct
import sys
import threading
import time

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

try:
    import websocket
except ImportError:
    print("ERROR: pip install websocket-client")
    sys.exit(1)

API = "http://localhost:8350/api/v1"
WS_URL = "ws://localhost:8350/ws"

# ── Helpers ──────────────────────────────────────────────────────────


def ws_send(ws, msg_type, **kwargs):
    """Send a WS command and return the response data.

    Drains any server-pushed events that arrive before the actual response.
    """
    payload = {"type": msg_type, **kwargs}
    ws.send(json.dumps(payload))
    while True:
        resp = json.loads(ws.recv())
        # Skip server-pushed events, wait for the actual result/error
        if resp.get("type", "").startswith("event."):
            continue
        if resp.get("type") == "error":
            print(f"  WS error: {resp.get('message')}")
        return resp.get("data", resp)


def ws_recv_result(ws):
    """Receive the next non-event response from WS."""
    while True:
        resp = json.loads(ws.recv())
        if resp.get("type", "").startswith("event."):
            continue
        return resp


def wait_server():
    """Wait for the API server to be ready."""
    for _ in range(20):
        try:
            r = requests.get(f"{API}/system/ping", timeout=1)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    print("ERROR: Could not connect to Locul3D on port 8350")
    print("       Start the viewer first: python -m locul3d")
    sys.exit(1)


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Phase 1: REST dynamic layer CRUD ────────────────────────────────


def demo_dynamic_layers_rest():
    section("Phase 2: Dynamic Layer CRUD (REST)")

    # Create a point cloud layer — rainbow gradient
    print("  Creating dynamic point cloud layer (rainbow)...")
    r = requests.post(f"{API}/scene/dynamic", json={
        "name": "demo_points",
        "geometry_type": "pointcloud",
        "points": [[i * 0.1, math.sin(i * 0.1) * 0.5, math.cos(i * 0.05) * 0.3]
                    for i in range(100)],
        "colors": [[int(255 * (0.5 + 0.5 * math.sin(i * 0.06))),
                     int(255 * (0.5 + 0.5 * math.sin(i * 0.06 + 2.1))),
                     int(255 * (0.5 + 0.5 * math.sin(i * 0.06 + 4.2)))]
                    for i in range(100)],
        "color": [1.0, 0.4, 0.1],
        "opacity": 1.0,
    })
    assert r.status_code == 200, f"Create failed: {r.text}"
    info = r.json()
    print(f"    Created: {info['layer_id']} ({info['point_count']} points)")

    # Create a triangle mesh (cube) — bright cyan
    print("  Creating dynamic mesh (cube)...")
    verts = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    tris = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 7, 3], [2, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ]
    r = requests.post(f"{API}/scene/dynamic", json={
        "name": "demo_cube",
        "geometry_type": "mesh",
        "vertices": verts,
        "triangles": tris,
        "color": [0.2, 0.6, 1.0],
        "opacity": 0.8,
    })
    assert r.status_code == 200
    info = r.json()
    print(f"    Created: {info['layer_id']} ({info['tri_count']} triangles)")

    # Create bbox collection
    print("  Creating bbox collection layer...")
    r = requests.post(f"{API}/scene/dynamic", json={
        "name": "demo_boxes",
        "geometry_type": "bboxes",
        "bboxes": [
            {"label": "rack_A", "center": [3, 0, 1], "size": [1, 1, 2],
             "color": [1, 0.3, 0]},
            {"label": "rack_B", "center": [3, 3, 1], "size": [1, 1, 2],
             "color": [0, 0.8, 0.3]},
        ],
    })
    assert r.status_code == 200
    print(f"    Created: {r.json()['layer_id']}")

    # Create surface layer
    print("  Creating surface layer...")
    r = requests.post(f"{API}/scene/dynamic", json={
        "name": "demo_floor",
        "geometry_type": "surfaces",
        "surfaces": [
            {"axis": "xy", "center": [2, 2, 0], "size": [8, 8],
             "color": [0.4, 0.4, 0.6], "opacity": 0.2},
        ],
    })
    assert r.status_code == 200
    print(f"    Created: {r.json()['layer_id']}")

    # List all dynamic layers
    r = requests.get(f"{API}/scene/dynamic")
    layers = r.json()
    print(f"\n  Dynamic layers in scene: {len(layers)}")
    for l in layers:
        print(f"    - {l['layer_id']} ({l['geometry_type']}, "
              f"{l['point_count']} pts, {l['tri_count']} tris)")

    # Patch a layer (property-only, no VBO rebuild)
    print("\n  Patching demo_cube opacity to 0.4...")
    r = requests.patch(f"{API}/scene/dynamic/dyn_demo_cube", json={
        "opacity": 0.4,
        "color": [1.0, 0.2, 0.8],
    })
    assert r.status_code == 200

    # Duplicate name should return 409
    r = requests.post(f"{API}/scene/dynamic", json={
        "name": "demo_points",
        "geometry_type": "pointcloud",
        "points": [[0, 0, 0]],
    })
    assert r.status_code == 409, f"Expected 409 for duplicate, got {r.status_code}"
    print("  Duplicate name correctly returned 409")

    # Fit camera to see everything
    requests.post(f"{API}/camera/fit")
    print("\n  Camera fitted to scene.")


# ── Phase 2: WebSocket real-time animation ──────────────────────────


def demo_ws_animation(ws):
    section("Phase 3: WebSocket Animation & Real-Time Updates")

    # Create a stretching cube for animation
    print("  Creating stretching cube...")
    ws_send(ws, "dynamic.create",
            name="stretching_cube",
            geometry_type="mesh",
            vertices=[
                [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
                [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1],
            ],
            triangles=[
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 5, 1], [0, 4, 5],
                [2, 7, 3], [2, 6, 7],
                [0, 3, 7], [0, 7, 4],
                [1, 5, 6], [1, 6, 2],
            ],
            color=[1.0, 0.3, 0.1],
            opacity=0.9)

    # Camera orbit via continuous transform
    print("  Starting 15 deg/sec camera orbit...")
    ws_send(ws, "camera.transform_continuous",
            id="demo-orbit", property="azimuth", rate=15.0, duration_ms=0)

    # Keyframed camera zoom pulse
    print("  Starting camera zoom pulse...")
    ws_send(ws, "camera.animate",
            id="zoom-pulse",
            keyframes=[
                {"t": 0.0, "distance": 12, "fov": 45},
                {"t": 1.0, "distance": 6, "fov": 55},
            ],
            duration_ms=4000,
            loop=True,
            ping_pong=True,
            easing="ease_in_out")

    # Keyframed object animation (color + opacity)
    print("  Starting keyframed object animation (color + opacity pulse)...")
    ws_send(ws, "dynamic.animate",
            id="cube-anim",
            layer_id="dyn_stretching_cube",
            keyframes=[
                {"t": 0.0, "color": [1.0, 0.3, 0.1], "opacity": 1.0},
                {"t": 0.5, "color": [0.1, 0.8, 1.0], "opacity": 0.6},
                {"t": 1.0, "color": [1.0, 0.3, 0.1], "opacity": 1.0},
            ],
            duration_ms=3000,
            loop=True,
            ping_pong=True,
            easing="ease_in_out")

    # Continuous color fade on bbox layer
    print("  Starting continuous color fade on bboxes...")
    ws_send(ws, "dynamic.transform_continuous",
            id="bbox-fade", layer_id="dyn_demo_boxes",
            property="color", target=[0.0, 0.3, 1.0], duration_ms=5000)

    print("  Animations running for 8 seconds...")
    time.sleep(8)

    # Stop all animations
    print("  Stopping all animations...")
    ws_send(ws, "transform.stop_all")
    print("  Animation demo complete.")


# ── Phase 3: High-FPS geometry updates ──────────────────────────────


def demo_high_fps_updates(ws):
    section("Phase 3: High-FPS Geometry Updates (60 Hz)")

    # Create a wave point cloud that updates every frame
    print("  Creating animated wave point cloud...")
    ws_send(ws, "dynamic.create",
            name="wave",
            geometry_type="pointcloud",
            points=[[0, 0, 0]],
            color=[0.0, 1.0, 0.5])

    frame_count = 180  # 3 seconds at 60fps
    target_dt = 1.0 / 60.0
    times = []

    print(f"  Streaming {frame_count} geometry updates at 60 Hz...")

    # Set a good camera view
    ws_send(ws, "camera.set", azimuth=45, elevation=35, distance=15,
            target=[5, 5, 0], fov=45)

    for frame in range(frame_count):
        t0 = time.perf_counter()
        t = frame * 0.05

        # Generate wave surface as point cloud
        pts = []
        for ix in range(40):
            for iy in range(40):
                x = ix * 0.25
                y = iy * 0.25
                z = math.sin(x * 0.5 + t) * math.cos(y * 0.5 + t * 0.7) * 1.5
                pts.append([x, y, z])

        # Update geometry via WS
        ws_send(ws, "dynamic.update",
                layer_id="dyn_wave",
                points=pts)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Throttle to ~60fps
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    avg_ms = sum(times) / len(times) * 1000
    max_ms = max(times) * 1000
    effective_fps = 1.0 / (sum(times) / len(times) + target_dt) if times else 0
    print(f"\n  Results:")
    print(f"    Frames sent:     {frame_count}")
    print(f"    Avg round-trip:  {avg_ms:.1f} ms")
    print(f"    Max round-trip:  {max_ms:.1f} ms")
    print(f"    Effective rate:  ~{min(60, effective_fps):.0f} fps")


# ── Phase 4: Binary point streaming ─────────────────────────────────


def demo_binary_streaming(ws):
    section("Phase 3: Binary Point Streaming (High Throughput)")

    import numpy as np

    # Create a layer via binary message (message_type=1)
    layer_name = "binary_cloud"
    name_bytes = layer_name.encode("utf-8")

    # Generate 10,000 random points
    n_points = 10_000
    points = np.random.randn(n_points, 3).astype(np.float32) * 3.0
    points[:, 0] += 10  # offset to not overlap with other demo layers

    header = struct.pack("<II", 1, len(name_bytes))  # type=1 (create)
    binary_msg = header + name_bytes + points.tobytes()

    print(f"  Creating point cloud via binary message ({n_points} points, "
          f"{len(binary_msg)} bytes)...")
    ws.send(binary_msg, opcode=websocket.ABNF.OPCODE_BINARY)
    resp = ws_recv_result(ws)
    data = resp.get("data", resp)
    assert data.get("status") == "ok" or data.get("layer_id"), \
        f"Binary create failed: {resp}"
    print(f"    Created via binary: dyn_{layer_name}")

    # Append 5 batches of 5,000 points each
    print("  Appending 5 batches of 5,000 points each...")
    for batch in range(5):
        pts = np.random.randn(5_000, 3).astype(np.float32) * 3.0
        pts[:, 0] += 10
        pts[:, 2] += batch * 2  # stack vertically

        header = struct.pack("<II", 2, len(name_bytes))  # type=2 (append)
        binary_msg = header + name_bytes + pts.tobytes()

        ws.send(binary_msg, opcode=websocket.ABNF.OPCODE_BINARY)
        resp = ws_recv_result(ws)
        count = resp.get("data", {}).get("point_count", "?")
        print(f"    Batch {batch + 1}: total points now = {count}")

    # Fit camera
    ws_send(ws, "camera.fit")
    print("  Binary streaming complete.")


# ── Phase 5: Event listener ─────────────────────────────────────────


def demo_event_listener(ws_main):
    section("Phase 3: Event Broadcasting")

    events_received = []

    def listener():
        """Background thread to listen for server-pushed events."""
        ws2 = websocket.create_connection(WS_URL, timeout=5)
        try:
            while True:
                try:
                    msg = ws2.recv()
                    data = json.loads(msg)
                    if data.get("type", "").startswith("event."):
                        events_received.append(data)
                except websocket.WebSocketTimeoutException:
                    break
                except Exception:
                    break
        finally:
            ws2.close()

    # Start listener on background thread
    t = threading.Thread(target=listener, daemon=True)
    t.start()
    time.sleep(0.3)

    # Trigger some events
    print("  Creating and deleting a layer to trigger events...")
    ws_send(ws_main, "dynamic.create",
            name="event_test",
            geometry_type="pointcloud",
            points=[[0, 0, 0], [1, 1, 1]])

    time.sleep(0.3)

    ws_send(ws_main, "dynamic.delete", layer_id="dyn_event_test")

    time.sleep(0.5)
    t.join(timeout=1)

    print(f"  Events received by listener: {len(events_received)}")
    for evt in events_received:
        print(f"    - {evt['type']}: {json.dumps({k: v for k, v in evt.items() if k != 'type'})}")


# ── Cleanup ──────────────────────────────────────────────────────────


def cleanup(ws):
    section("Cleanup")
    print("  Clearing all dynamic layers...")
    ws_send(ws, "dynamic.clear")
    print("  Done.")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("Locul3D Remote Control API Demo")
    print("=" * 60)

    wait_server()
    status = requests.get(f"{API}/system/status").json()
    print(f"  Connected to Locul3D ({status['mode']} mode)")
    print(f"  Layers: {status['layers_count']}, "
          f"Points: {status['total_points']:,}")

    # Phase 2: REST CRUD
    demo_dynamic_layers_rest()

    # Connect WebSocket for remaining demos
    ws = websocket.create_connection(WS_URL, timeout=10)

    try:
        # Phase 3: Animation
        demo_ws_animation(ws)

        # Phase 3: High-FPS updates
        demo_high_fps_updates(ws)

        # Phase 3: Binary streaming
        demo_binary_streaming(ws)

        # Phase 3: Event broadcasting
        demo_event_listener(ws)

        # Cleanup
        cleanup(ws)

    finally:
        ws.close()

    section("Demo Complete")
    print("  All phases demonstrated successfully!")
    print()


if __name__ == "__main__":
    main()
