#!/usr/bin/env python3
"""Smoke test for the Locul3D Remote Control API (Phase 1).

Usage:
    1. Start the viewer:  python -m locul3d
    2. Run this script:   python scripts/test_api_smoke.py

Tests all Phase 1 REST endpoints:
  - System: ping, status, screenshot
  - Camera: GET/PUT full state, individual params, preset, fit, look_at
  - Viewport: settings, correction, clip
  - Scene: layers, bounds
  - WebSocket: connect, camera.set, camera.fit
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: `requests` not installed.  pip install requests")
    sys.exit(1)

API = "http://localhost:8350/api/v1"
WS_URL = "ws://localhost:8350/ws"

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


def assert_ok(resp, expected_status=200):
    assert resp.status_code == expected_status, (
        f"Expected {expected_status}, got {resp.status_code}: {resp.text[:200]}"
    )


# ── Wait for server ──────────────────────────────────────────────────

print("\n🔌 Connecting to Remote API...")
for attempt in range(10):
    try:
        r = requests.get(f"{API}/system/ping", timeout=1)
        if r.status_code == 200:
            print(f"   Connected on attempt {attempt + 1}\n")
            break
    except requests.ConnectionError:
        pass
    time.sleep(0.5)
else:
    print("   ✗ Could not connect to API. Is the viewer running?")
    print("     Start with:  python -m locul3d")
    sys.exit(1)


# ── System ────────────────────────────────────────────────────────────

print("📡 System endpoints")


def test_ping():
    r = requests.get(f"{API}/system/ping")
    assert_ok(r)
    data = r.json()
    assert data["pong"] is True, f"Expected pong=True, got {data}"


def test_status():
    r = requests.get(f"{API}/system/status")
    assert_ok(r)
    data = r.json()
    assert "mode" in data
    assert "layers_count" in data
    assert "api_version" in data
    print(f"       mode={data['mode']}, layers={data['layers_count']}")


def test_screenshot():
    r = requests.get(f"{API}/system/screenshot")
    assert_ok(r)
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 100, f"Screenshot too small: {len(r.content)} bytes"
    print(f"       {len(r.content)} bytes PNG")


test("ping", test_ping)
test("status", test_status)
test("screenshot", test_screenshot)


# ── Camera ────────────────────────────────────────────────────────────

print("\n📷 Camera endpoints")


def test_get_camera():
    r = requests.get(f"{API}/camera")
    assert_ok(r)
    data = r.json()
    for key in ("azimuth", "elevation", "distance", "target", "fov"):
        assert key in data, f"Missing key: {key}"
    print(f"       az={data['azimuth']:.1f} el={data['elevation']:.1f} dist={data['distance']:.1f}")


def test_set_camera_full():
    r = requests.put(f"{API}/camera", json={
        "azimuth": 90, "elevation": 45, "distance": 60, "fov": 50
    })
    assert_ok(r)
    data = r.json()
    assert abs(data["azimuth"] - 90) < 0.01
    assert abs(data["elevation"] - 45) < 0.01


def test_set_azimuth():
    r = requests.put(f"{API}/camera/azimuth", json={"value": 180})
    assert_ok(r)
    data = r.json()
    assert abs(data["azimuth"] - 180) < 0.01


def test_set_elevation():
    r = requests.put(f"{API}/camera/elevation", json={"value": 15})
    assert_ok(r)
    assert abs(r.json()["elevation"] - 15) < 0.01


def test_set_distance():
    r = requests.put(f"{API}/camera/distance", json={"value": 100})
    assert_ok(r)
    assert abs(r.json()["distance"] - 100) < 0.01


def test_set_fov():
    r = requests.put(f"{API}/camera/fov", json={"value": 60})
    assert_ok(r)
    assert abs(r.json()["fov"] - 60) < 0.01


def test_set_target():
    r = requests.put(f"{API}/camera/target", json={"value": [1, 2, 3]})
    assert_ok(r)
    t = r.json()["target"]
    assert abs(t[0] - 1) < 0.01 and abs(t[1] - 2) < 0.01 and abs(t[2] - 3) < 0.01


def test_camera_preset():
    r = requests.post(f"{API}/camera/preset", json={"preset": "Top"})
    assert_ok(r)
    data = r.json()
    assert abs(data["elevation"] - 89) < 0.1, f"Top preset elevation should be ~89, got {data['elevation']}"


def test_camera_fit():
    r = requests.post(f"{API}/camera/fit")
    assert_ok(r)
    assert r.json().get("status") == "ok"


def test_camera_look_at():
    r = requests.post(f"{API}/camera/look_at", json={"target": [5, 5, 0], "distance": 30})
    assert_ok(r)
    data = r.json()
    t = data["target"]
    assert abs(t[0] - 5) < 0.01


test("GET  /camera", test_get_camera)
test("PUT  /camera (full state)", test_set_camera_full)
test("PUT  /camera/azimuth", test_set_azimuth)
test("PUT  /camera/elevation", test_set_elevation)
test("PUT  /camera/distance", test_set_distance)
test("PUT  /camera/fov", test_set_fov)
test("PUT  /camera/target", test_set_target)
test("POST /camera/preset (Top)", test_camera_preset)
test("POST /camera/fit", test_camera_fit)
test("POST /camera/look_at", test_camera_look_at)


# ── Viewport ──────────────────────────────────────────────────────────

print("\n🖥️  Viewport endpoints")


def test_get_viewport():
    r = requests.get(f"{API}/viewport")
    assert_ok(r)
    data = r.json()
    assert "point_size" in data
    assert "show_axes" in data
    print(f"       point_size={data['point_size']} axes={data['show_axes']} grid={data['show_grid']}")


def test_set_viewport():
    r = requests.put(f"{API}/viewport", json={"point_size": 4, "show_grid": False})
    assert_ok(r)
    # Read back
    r2 = requests.get(f"{API}/viewport")
    data = r2.json()
    assert data["point_size"] == 4
    assert data["show_grid"] is False
    # Restore
    requests.put(f"{API}/viewport", json={"point_size": 2, "show_grid": True})


def test_get_correction():
    r = requests.get(f"{API}/viewport/correction")
    assert_ok(r)
    data = r.json()
    assert "rotate_x" in data


def test_set_correction():
    r = requests.put(f"{API}/viewport/correction", json={
        "rotate_x": -90, "shift_z": -1.0
    })
    assert_ok(r)
    # Read back
    r2 = requests.get(f"{API}/viewport/correction")
    data = r2.json()
    assert abs(data["rotate_x"] - (-90)) < 0.01
    # Restore
    requests.put(f"{API}/viewport/correction", json={
        "rotate_x": 0, "rotate_y": 0, "rotate_z": 0,
        "shift_x": 0, "shift_y": 0, "shift_z": 0,
    })


def test_clip():
    # Set clip
    r = requests.put(f"{API}/viewport/clip", json={
        "x_min": -10, "x_max": 10, "y_min": -10, "y_max": 10,
        "z_min": -5, "z_max": 5,
    })
    assert_ok(r)
    # Read back
    r2 = requests.get(f"{API}/viewport/clip")
    data = r2.json()
    assert data["active"] is True
    assert data["x_min"] == -10
    # Clear
    r3 = requests.delete(f"{API}/viewport/clip")
    assert_ok(r3)
    r4 = requests.get(f"{API}/viewport/clip")
    assert r4.json()["active"] is False


test("GET  /viewport", test_get_viewport)
test("PUT  /viewport (point_size + grid)", test_set_viewport)
test("GET  /viewport/correction", test_get_correction)
test("PUT  /viewport/correction", test_set_correction)
test("clip  (set → read → clear → verify)", test_clip)


# ── Scene ─────────────────────────────────────────────────────────────

print("\n🌍 Scene endpoints")


def test_get_layers():
    r = requests.get(f"{API}/scene/layers")
    assert_ok(r)
    data = r.json()
    assert isinstance(data, list)
    print(f"       {len(data)} layers loaded")


def test_get_bounds():
    r = requests.get(f"{API}/scene/bounds")
    assert_ok(r)
    data = r.json()
    assert "center" in data
    assert "radius" in data
    print(f"       center={data['center']}, radius={data['radius']:.2f}")


test("GET  /scene/layers", test_get_layers)
test("GET  /scene/bounds", test_get_bounds)


# ── WebSocket ─────────────────────────────────────────────────────────

print("\n🔗 WebSocket")

try:
    import websocket

    def test_ws_ping():
        ws = websocket.create_connection(WS_URL, timeout=3)
        ws.send(json.dumps({"type": "camera.fit", "id": "test-1"}))
        resp = json.loads(ws.recv())
        assert resp.get("type") == "result", f"Expected result, got {resp}"
        assert resp.get("id") == "test-1"
        assert resp.get("status") == "ok"
        ws.close()

    def test_ws_camera_set():
        ws = websocket.create_connection(WS_URL, timeout=3)
        ws.send(json.dumps({
            "type": "camera.set",
            "id": "test-2",
            "azimuth": 270,
            "elevation": 10,
        }))
        resp = json.loads(ws.recv())
        assert resp["status"] == "ok"
        assert abs(resp["data"]["azimuth"] - 270) < 0.01
        ws.close()

    def test_ws_camera_preset():
        ws = websocket.create_connection(WS_URL, timeout=3)
        ws.send(json.dumps({
            "type": "camera.preset",
            "id": "test-3",
            "preset": "Isometric",
        }))
        resp = json.loads(ws.recv())
        assert resp["status"] == "ok"
        assert abs(resp["data"]["azimuth"] - 45) < 0.1
        ws.close()

    test("WS camera.fit", test_ws_ping)
    test("WS camera.set", test_ws_camera_set)
    test("WS camera.preset", test_ws_camera_preset)

except ImportError:
    print("  ⚠ `websocket-client` not installed — WS tests skipped")
    print("    Install with:  pip install websocket-client")


# ── Summary ───────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
total = passed + failed
if failed == 0:
    print(f"  ✅ All {passed} tests passed!")
else:
    print(f"  ❌ {failed}/{total} tests FAILED, {passed} passed")
print(f"{'═' * 50}\n")

sys.exit(1 if failed else 0)
