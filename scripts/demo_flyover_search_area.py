#!/usr/bin/env python3
"""Demo Flyover — Orbit camera around the `search_region` bbox.

Usage:
    1. Start Locul3D and load an .e57 with a scene yaml that defines a
       bbox annotation labeled `search_region`.
    2. Run:  python scripts/demo_flyover_search_area.py
             python scripts/demo_flyover_search_area.py --duration 20 --elevation 30

The script reads the search_region bbox via REST, frames the camera on
it, then starts a continuous azimuth rotation via the animation engine.
Press Ctrl-C (or re-run with --stop) to halt the rotation.
"""

import argparse
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
TRACK_ID = "flyover-search-region"
SEARCH_LABEL = "search_region"


def ws_send(ws, msg_type, **kwargs):
    ws.send(json.dumps({"type": msg_type, **kwargs}))
    while True:
        resp = json.loads(ws.recv())
        if not resp.get("type", "").startswith("event."):
            return resp.get("data", resp)


def wait_server():
    for _ in range(20):
        try:
            if requests.get(f"{API}/system/ping", timeout=1).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    print("ERROR: Locul3D not running on port 8350")
    sys.exit(1)


def find_search_bbox():
    r = requests.get(f"{API}/shapes/bboxes", timeout=5)
    if r.status_code != 200:
        print(f"ERROR: GET /shapes/bboxes -> {r.status_code} {r.text}")
        sys.exit(1)
    bboxes = r.json()
    for b in bboxes:
        if b.get("label") == SEARCH_LABEL:
            return b
    labels = ", ".join(b.get("label", "?") for b in bboxes) or "(none)"
    print(f"ERROR: no bbox labeled '{SEARCH_LABEL}'. Found: {labels}")
    sys.exit(1)


def bbox_center_size(b):
    if "center" in b and "size" in b:
        c = list(b["center"])
        s = list(b["size"])
    else:
        mn, mx = b["min"], b["max"]
        c = [(mn[i] + mx[i]) / 2 for i in range(3)]
        s = [mx[i] - mn[i] for i in range(3)]
    return c, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=15.0,
                    help="Seconds for one full 360° revolution")
    ap.add_argument("--elevation", type=float, default=25.0,
                    help="Camera elevation angle in degrees")
    ap.add_argument("--margin", type=float, default=1.6,
                    help="Distance multiplier on the bbox bounding sphere")
    ap.add_argument("--fov", type=float, default=None,
                    help="Override camera FOV (degrees)")
    ap.add_argument("--clip", choices=["none", "z", "box"], default="z",
                    help="Scene clipping: none, z=clip ceiling only, "
                         "box=clip to bbox horizontally too")
    ap.add_argument("--clip-pad-z", type=float, default=0.2,
                    help="Vertical padding (m) added above/below the bbox")
    ap.add_argument("--clip-pad-xy", type=float, default=2.0,
                    help="Horizontal padding (m) for --clip box")
    ap.add_argument("--stop", action="store_true",
                    help="Stop the flyover and exit")
    args = ap.parse_args()

    wait_server()
    ws = websocket.create_connection(WS_URL, timeout=10)

    if args.stop:
        # Try targeted stop; fall back to stop_all for safety.
        try:
            ws_send(ws, "transform.stop", track_id=TRACK_ID)
        except Exception:
            pass
        ws_send(ws, "transform.stop_all")
        try:
            requests.delete(f"{API}/viewport/clip", timeout=5)
        except Exception:
            pass
        print("Flyover stopped.")
        return

    bbox = find_search_bbox()
    center, size = bbox_center_size(bbox)
    bb_min = [center[i] - size[i] / 2 for i in range(3)]
    bb_max = [center[i] + size[i] / 2 for i in range(3)]
    # Bounding sphere radius of the bbox
    radius = 0.5 * math.sqrt(sum(s * s for s in size))
    print(f"search_region: center={center}, size={size}, radius={radius:.2f}")

    # Frame the bbox: distance so the bsphere fits the vertical FOV
    cam = requests.get(f"{API}/camera", timeout=5).json()
    fov_deg = args.fov if args.fov is not None else cam.get("fov", 60.0)
    fov_rad = math.radians(fov_deg)
    distance = (radius * args.margin) / math.tan(fov_rad / 2)

    if args.fov is not None:
        requests.put(f"{API}/camera/fov", json={"value": fov_deg})

    # Scene clip — hide the ceiling (and optionally outer walls)
    if args.clip != "none":
        BIG = 1e6
        if args.clip == "z":
            clip = {
                "x_min": -BIG, "x_max": BIG,
                "y_min": -BIG, "y_max": BIG,
                "z_min": bb_min[2] - args.clip_pad_z,
                "z_max": bb_max[2] + args.clip_pad_z,
            }
        else:  # box
            clip = {
                "x_min": bb_min[0] - args.clip_pad_xy,
                "x_max": bb_max[0] + args.clip_pad_xy,
                "y_min": bb_min[1] - args.clip_pad_xy,
                "y_max": bb_max[1] + args.clip_pad_xy,
                "z_min": bb_min[2] - args.clip_pad_z,
                "z_max": bb_max[2] + args.clip_pad_z,
            }
        r = requests.put(f"{API}/viewport/clip", json=clip, timeout=5)
        if r.status_code != 200:
            print(f"WARN: clip failed: {r.status_code} {r.text}")
        else:
            print(f"Scene clip applied ({args.clip}): "
                  f"z=[{clip['z_min']:.2f}, {clip['z_max']:.2f}]")

    # Aim camera at bbox center
    requests.put(f"{API}/camera/target", json={"value": center})
    requests.put(f"{API}/camera/elevation", json={"value": args.elevation})
    requests.put(f"{API}/camera/distance", json={"value": distance})
    requests.put(f"{API}/camera/azimuth", json={"value": 0.0})

    # Continuous azimuth rotation: 360° over `duration` seconds
    rate = 360.0 / max(args.duration, 0.1)
    ws_send(ws, "camera.transform_continuous",
            track_id=TRACK_ID, property="azimuth", rate=rate)
    print(f"Flyover started: {rate:.1f} deg/s "
          f"(one revolution every {args.duration:.1f}s)")
    print(f"Stop with: python {sys.argv[0]} --stop")


if __name__ == "__main__":
    main()
