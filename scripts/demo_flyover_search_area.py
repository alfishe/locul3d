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
    ap.add_argument("--fade", action="store_true",
                    help="Enable shader fade for points between camera "
                         "and target (50%% alpha by default)")
    ap.add_argument("--fade-mul", type=float, default=0.4,
                    help="Alpha multiplier for faded points (0..1)")
    ap.add_argument("--fade-band", type=float, default=0.8,
                    help="Smoothstep half-band around target (m)")
    ap.add_argument("--max-fps", type=float, default=125.0,
                    help="Realtime FPS CEILING (default 125 — matches "
                         "the engine tick rate). The adaptive controller "
                         "drops below this only when paint time exceeds "
                         "the budget.")
    ap.add_argument("--min-fps", type=float, default=5.0,
                    help="Realtime FPS FLOOR — never throttle below this.")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="Disable the closed-loop FPS controller and "
                         "render at exactly --max-fps regardless of load.")
    ap.add_argument("--preview", action="store_true",
                    help="Preview mode: hold target FPS by adapting LOD "
                         "(stride decimation) instead of full-res "
                         "rendering. Use to scout camera paths smoothly.")
    ap.add_argument("--preview-fps", type=float, default=60.0,
                    help="Target FPS for --preview mode (default 60).")
    ap.add_argument("--no-time-scale", action="store_true",
                    help="Disable automatic animation slowdown. By "
                         "default the animation clock decelerates by "
                         "eff_fps/nominal_fps so that each rendered "
                         "frame represents 1/nominal_fps of motion, "
                         "even at 1 FPS — slow but smooth instead of "
                         "fast and jumpy.")
    ap.add_argument("--nominal-fps", type=float, default=60.0,
                    help="Frame rate the animation is *authored* for "
                         "(default 60). The auto time-scale slows "
                         "playback by eff_fps / nominal_fps.")
    ap.add_argument("--stop", action="store_true",
                    help="Stop the flyover and exit")
    args = ap.parse_args()

    wait_server()
    ws = websocket.create_connection(WS_URL, timeout=10)

    # Warn if vsync is on — it'll spoof the adaptive FPS controller.
    try:
        vp_settings = requests.get(f"{API}/viewport", timeout=5).json()
        if vp_settings.get("vsync"):
            print("WARN: vsync is ON. Adaptive FPS will misread paint "
                  "cost as display-refresh wait. Restart the editor "
                  "with --no-vsync (or omit the --vsync flag).")
    except Exception:
        pass

    if args.stop:
        # Try targeted stop; fall back to stop_all for safety.
        try:
            ws_send(ws, "transform.stop", track_id=TRACK_ID)
        except Exception:
            pass
        ws_send(ws, "transform.stop_all")
        try:
            requests.put(f"{API}/viewport/fade", json={"enable": False},
                         timeout=5)
        except Exception:
            pass
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

    # Optional shader fade for points occluding the AoI.  Fade is a
    # cone from the camera through the bbox's bounding sphere — only
    # points inside that cone AND in front of the AoI's near edge are
    # dimmed, so the AoI itself stays full-opacity.
    if args.fade:
        r = requests.put(f"{API}/viewport/fade", json={
            "enable": True,
            "alpha_mul": args.fade_mul,
            "band": args.fade_band,
            "aoi_center": center,
            "aoi_radius": radius,
        }, timeout=5)
        if r.status_code == 200:
            info = r.json()
            if info.get("available"):
                print(f"Point fade ON: alpha_mul={args.fade_mul}, "
                      f"band={args.fade_band}")
            else:
                print("WARN: shader unavailable on this driver — fade off")
        else:
            print(f"WARN: fade endpoint failed: {r.status_code} {r.text}")

    # Configure the animation time-scale. Default AUTO slows the
    # wall-clock playback so each rendered frame represents one
    # nominal_fps step, even when paint is heavy and eff_fps drops
    # to 1-2 — slow-but-smooth instead of fast-and-jumpy.
    ws_send(ws, "animation.set_time_scale",
            auto=(not args.no_time_scale),
            nominal_fps=args.nominal_fps)
    if args.no_time_scale:
        print("Animation time-scale: FIXED 1.0× (will jump at low FPS)")
    else:
        print(f"Animation time-scale: AUTO (nominal {args.nominal_fps:.0f} FPS)")

    # Configure realtime/preview render mode.
    if args.preview:
        prev_resp = ws_send(ws, "animation.set_preview_mode",
                            enable=True,
                            target_fps=args.preview_fps)
        # Disable full-res adaptive so the two controllers don't fight.
        ws_send(ws, "animation.set_realtime_fps",
                fps=args.preview_fps, adaptive=False)
        print(f"PREVIEW mode: target={prev_resp['target_fps']:.0f} FPS, "
              f"start budget={prev_resp['budget_pts']:,} pts "
              f"(LOD adapts to hold the rate)")
    else:
        ws_send(ws, "animation.set_preview_mode", enable=False)
        fps_resp = ws_send(ws, "animation.set_realtime_fps",
                           fps=args.max_fps,
                           min_fps=args.min_fps,
                           adaptive=(not args.no_adaptive))
        mode = "fixed" if args.no_adaptive else "adaptive"
        print(f"FULL-RES [{mode}]: ceiling={fps_resp['max_fps']:.0f}, "
              f"floor={fps_resp['min_fps']:.0f}")

    # Continuous azimuth rotation: 360° over `duration` seconds
    rate = 360.0 / max(args.duration, 0.1)
    ws_send(ws, "camera.transform_continuous",
            track_id=TRACK_ID, property="azimuth", rate=rate)
    print(f"Flyover started: {rate:.1f} deg/s "
          f"(one revolution every {args.duration:.1f}s)")
    print(f"Stop with: python {sys.argv[0]} --stop")


if __name__ == "__main__":
    main()
