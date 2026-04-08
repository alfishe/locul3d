"""Recording REST endpoints — start/stop/pause/resume/status."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from aiohttp import web

from ..dispatcher import CommandDispatcher

log = logging.getLogger(__name__)


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    app.router.add_get("/api/v1/recording/status", _status)
    app.router.add_post("/api/v1/recording/start", _start)
    app.router.add_post("/api/v1/recording/stop", _stop)
    app.router.add_post("/api/v1/recording/pause", _pause)
    app.router.add_post("/api/v1/recording/resume", _resume)
    app.router.add_get("/api/v1/recording/encoders", _list_encoders)


def _config_to_dict(cfg) -> dict:
    if cfg is None:
        return {}
    return {
        "path": str(cfg.path),
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "codec": cfg.codec,
        "encoder": cfg.encoder,
        "encoder_kind": cfg.encoder_kind,
        "bitrate_kbps": cfg.bitrate_kbps,
    }


def _stats_to_dict(stats) -> dict:
    return {
        "state": stats.state,
        "frames_written": stats.frames_written,
        "frames_dropped": stats.frames_dropped,
        "bytes_written": stats.bytes_written,
        "duration_s": stats.duration_s,
        "started_at": stats.started_at,
        "last_error": stats.last_error,
        "warnings": list(stats.warnings),
    }


async def _status(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    rec = dispatcher.recorder
    return web.json_response({
        "state": rec.state,
        "is_open": rec.is_open,
        "is_active": rec.is_active,
        "config": _config_to_dict(rec.config),
        "stats": _stats_to_dict(rec.stats),
    })


async def _list_encoders(request: web.Request) -> web.Response:
    from locul3d.recording.encoders import (
        EncoderUnavailable,
        list_available_encoders,
        find_ffmpeg,
        select_encoder,
    )
    try:
        ffmpeg = find_ffmpeg()
        all_encs = sorted(list_available_encoders())
    except EncoderUnavailable as exc:
        return web.json_response({"error": str(exc)}, status=503)

    out = {"ffmpeg": ffmpeg, "all": all_encs, "selection": {}}
    for codec in ("h264", "hevc"):
        sel = {}
        for pref in ("auto", "hw", "sw"):
            try:
                enc, kind, warns = select_encoder(codec, pref)
                sel[pref] = {
                    "encoder": enc, "kind": kind, "warnings": warns,
                }
            except EncoderUnavailable as exc:
                sel[pref] = {"error": str(exc)}
        out["selection"][codec] = sel
    return web.json_response(out)


async def _start(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()

    # Resolution: either explicit width/height, a preset name, or
    # the special 'viewport' value meaning "current widget size".
    from locul3d.recording.recorder import RESOLUTION_PRESETS
    width = data.get("width")
    height = data.get("height")
    if width is None or height is None:
        preset = (data.get("resolution") or "viewport").strip().lower()
        if preset in ("viewport", "native", "current"):
            # Read live viewport size on the Qt thread.
            def _read_size():
                vp = dispatcher._viewport
                # Account for HiDPI: pixel size is logical * DPR
                try:
                    dpr = float(vp.devicePixelRatioF())
                except Exception:
                    dpr = 1.0
                w = max(2, int(vp.width() * dpr))
                h = max(2, int(vp.height() * dpr))
                # H.264/HEVC need even dimensions for yuv420p output.
                w -= w % 2
                h -= h % 2
                return w, h
            width, height = await dispatcher._bridge.invoke_on_qt(_read_size)
        elif preset in RESOLUTION_PRESETS:
            width, height = RESOLUTION_PRESETS[preset]
        else:
            return web.json_response(
                {"error": f"unknown resolution preset {preset!r}; "
                          f"valid: {sorted(RESOLUTION_PRESETS)} + viewport"},
                status=400,
            )

    fps = float(data.get("fps", 60.0))
    codec = (data.get("codec") or "hevc").strip().lower()
    hw_pref = (data.get("hw") or "auto").strip().lower()
    bitrate_kbps = data.get("bitrate_kbps")

    # Output path resolution:
    #   - absolute path → use as-is
    #   - relative path → resolved under <repo>/video/
    #   - missing      → <repo>/video/locul3d_YYYYMMDD_HHMMSS.mp4
    from locul3d import __file__ as pkg_init
    repo_root = Path(pkg_init).resolve().parents[2]
    video_dir = repo_root / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data.get("path")
    if not raw_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_str = str(video_dir / f"locul3d_{ts}.mp4")
    else:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = video_dir / p
        path_str = str(p)

    try:
        result = await dispatcher.start_recording(
            path=path_str,
            width=int(width),
            height=int(height),
            fps=fps,
            codec=codec,
            hw_pref=hw_pref,
            bitrate_kbps=int(bitrate_kbps) if bitrate_kbps else None,
        )
    except Exception as exc:
        log.exception("recording.start failed")
        return web.json_response({"error": str(exc)}, status=400)

    return web.json_response(result)


async def _stop(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    try:
        result = await dispatcher.stop_recording()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result)


async def _pause(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    try:
        result = await dispatcher.pause_recording()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result)


async def _resume(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    try:
        result = await dispatcher.resume_recording()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result)
