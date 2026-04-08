# Locul3D Remote Control API

> Architecture design for HTTP REST + WebSocket remote control of the Locul3D viewer and editor.

## 1. Objectives

| Goal | Description |
|------|-------------|
| **External scripting** | Control the viewer programmatically from Python scripts, Jupyter notebooks, CI/CD pipelines, or any HTTP/WS client |
| **Demo recording** | Script camera movements, file loading, shape creation, and layer toggling to produce repeatable presentation sequences |
| **Real-time streaming** | Push point clouds, shapes, and annotations into the live viewport at interactive rates via WebSocket |
| **OpenAPI manifest** | Auto-generated OpenAPI 3.1 spec served at `/openapi.json` for client codegen and documentation |
| **DOM/scene control** | Full access to the scene graph: layers, visibility, opacity, bounding boxes, planes, clipping, correction |
| **Minimal footprint** | Zero new heavyweight dependencies — built on Python stdlib + one lightweight ASGI library |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    External Clients                      │
│  (curl / Python requests / browser / Jupyter / scripts)  │
└──────────┬──────────────────────────────────┬────────────┘
           │ HTTP REST (JSON)                 │ WebSocket (JSON)
           │ :8350/api/v1/*                   │ :8350/ws
           ▼                                  ▼
┌──────────────────────────────────────────────────────────┐
│              ASGI Server (aiohttp)                       │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ REST Router │   │ WS Handler   │   │ OpenAPI Spec │   │
│  │ /api/v1/*   │   │ /ws          │   │ /openapi.json│   │
│  └──────┬──────┘   └──────┬───────┘   └──────────────┘   │
│         │                 │                              │
│         ▼                 ▼                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Command Dispatcher (thread-safe)          │ │
│  │  Validates → Queues → Invokes on Qt main thread     │ │
│  └──────────────────────────────┬──────────────────────┘ │
└─────────────────────────────────┼────────────────────────┘
                                  │ QMetaObject.invokeMethod
                                  │ (Qt::QueuedConnection)
                                  ▼
┌──────────────────────────────────────────────────────────┐
│                  Qt Main Thread                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ ViewerWindow │  │ EditorWindow │  │ BaseGLViewport│   │
│  │              │  │              │  │  + Camera     │   │
│  │ LayerManager │  │ Annotations  │  │  + Correction │   │
│  │ LayerPanel   │  │ BBoxPanel    │  │  + Clip Planes│   │
│  └──────────────┘  └──────────────┘  └───────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Threading Model

The viewer runs on the **Qt main thread** — all OpenGL and widget operations must execute there. The HTTP/WS server runs on a **separate asyncio thread**. Commands cross the boundary via `QMetaObject.invokeMethod` with `Qt.QueuedConnection`, which safely enqueues a callable onto the Qt event loop. Results are returned via `asyncio.Future` bridging.

---

## 3. Package Layout & Detachable Architecture

The remote control and animation subsystems are **fully optional plugins** — the viewer and editor work identically without them. No import, no overhead, no coupling.

```
src/locul3d/
├── remote/                       ← NEW — optional, detachable
│   ├── __init__.py               ← start_server() / stop_server()
│   ├── server.py                 ← aiohttp Application, routes, WS handler
│   ├── dispatcher.py             ← CommandDispatcher: validate + queue + invoke
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── scene.py              ← /api/v1/scene/* (layers, load, clear)
│   │   ├── camera.py             ← /api/v1/camera/* (position, FOV, presets)
│   │   ├── dynamic.py            ← /api/v1/scene/dynamic/* (dynamic geometry layers)
│   │   ├── shapes.py             ← /api/v1/shapes/* (editor bbox, plane overlays)
│   │   ├── viewport.py           ← /api/v1/viewport/* (clip, correction, render)
│   │   └── system.py             ← /api/v1/system/* (status, screenshot, ping)
│   ├── schemas.py                ← Pydantic models for request/response validation
│   ├── openapi.py                ← OpenAPI 3.1 spec generator
│   └── bridge.py                 ← Qt ↔ asyncio thread bridge utilities
├── animation/                    ← NEW — optional, detachable
│   ├── __init__.py               ← AnimationEngine factory
│   ├── engine.py                 ← AnimationEngine: QTimer tick loop, track management
│   ├── tracks.py                 ← CameraTrack, TransformTrack, PropertyTrack, ColorTrack
│   ├── easing.py                 ← Timing functions: named, cubic-bezier, spring, steps
│   └── interpolation.py          ← Lerp, slerp, shortest-arc, color space interpolation
```

### Detachment Strategy

**Zero coupling** — the viewer/editor never imports `remote` or `animation` directly. Both are activated via **lazy, guarded imports**:

```python
# In ViewerWindow / EditorWindow — startup hook
def _maybe_start_remote(self):
    if self._api_disabled:
        return  # --no-api flag
    try:
        from locul3d.remote import start_server
        self._remote_server = start_server(window=self, port=self._api_port)
    except ImportError:
        pass  # remote package not installed / stripped from dist

def _maybe_create_animation_engine(self):
    try:
        from locul3d.animation import create_engine
        self._animation_engine = create_engine(self.viewport)
    except ImportError:
        self._animation_engine = None  # animations unavailable
```

**Consequences:**

| Scenario | Behavior |
|----------|----------|
| Normal viewer use | `remote/` and `animation/` never imported, zero overhead |
| `--no-api` flag | Server skipped, animation engine still available for local scripting |
| `aiohttp` not installed | `remote/` import fails gracefully, viewer works normally |
| Unit testing viewer | No network deps needed, mock `_animation_engine = None` |
| CI/headless testing of API | Import `remote` + `animation` with mocked Qt window |

**Dependency isolation** — `aiohttp` and `pydantic` are listed as optional extras:

```toml
# pyproject.toml
[project.optional-dependencies]
remote = ["aiohttp>=3.9.0", "pydantic>=2.0.0"]
```

```bash
pip install locul3d[remote]   # with API support
pip install locul3d            # without — viewer works fine
```

---

## 4. Dependency Choice: `aiohttp`

| Criterion | aiohttp | Rationale |
|-----------|---------|-----------|
| **Async** | Native asyncio | Non-blocking alongside Qt event loop |
| **WebSocket** | First-class WS support | `aiohttp.web.WebSocketResponse` — no extra dep |
| **Weight** | Single pip dep | `pip install aiohttp` — no framework bloat |
| **Maturity** | 10+ years, 14k★ | Battle-tested, well-documented |
| **Compatibility** | Python 3.9+ | Matches Locul3D's `requires-python` |

**Alternative considered:** `FastAPI` + `uvicorn` — heavier dependency tree (starlette, pydantic v2, uvloop), more opinionated. We use Pydantic for schemas only, not as a web framework.

---

## 5. REST API Design (OpenAPI 3.1)

Base URL: `http://localhost:8350/api/v1`

### 5.1 System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/system/status` | Server health, viewer mode, loaded file count, FPS |
| `GET` | `/system/ping` | Heartbeat — returns `{"pong": true}` |
| `GET` | `/system/screenshot` | Capture viewport as PNG (returns binary) |
| `GET` | `/openapi.json` | OpenAPI 3.1 specification |

### 5.2 Scene / Layers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/scene/layers` | List all layers with id, name, type, visible, opacity, point_count |
| `POST` | `/scene/load` | Load file(s) by path — `{"paths": ["/path/to/scan.ply"]}` |
| `POST` | `/scene/load_folder` | Load all files from a folder — `{"path": "/folder"}` |
| `DELETE` | `/scene/clear` | Clear all layers from scene |
| `PUT` | `/scene/layers/{layer_id}` | Update layer properties — `{"visible": false, "opacity": 0.5}` |
| `GET` | `/scene/bounds` | Scene AABB — `{x_min, x_max, y_min, y_max, z_min, z_max}` |

### 5.3 Camera

**Full-state access** — read/write the entire camera state in one call:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/camera` | Full camera state — azimuth, elevation, distance, target, FOV |
| `PUT` | `/camera` | Set camera (all fields optional, partial update) — `{"azimuth": 45, "elevation": 30, "distance": 50, "target": [0,0,0], "fov": 45}` |

**Individual parameter setters** — convenience endpoints for tweaking one value:

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/camera/azimuth` | Set azimuth only — `{"value": 90}` |
| `PUT` | `/camera/elevation` | Set elevation only — `{"value": 30}` |
| `PUT` | `/camera/distance` | Set distance only — `{"value": 50}` |
| `PUT` | `/camera/fov` | Set FOV only — `{"value": 60}` |
| `PUT` | `/camera/target` | Set target only — `{"value": [1, 2, 3]}` |

**Actions:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/camera/preset` | Apply named preset — `{"preset": "Top"}` |
| `POST` | `/camera/fit` | Fit camera to scene bounds |
| `POST` | `/camera/look_at` | Look at point — `{"target": [x,y,z], "distance": 20}` |

All individual setters return the full `CameraState` in the response, so callers can chain without extra GETs.

### 5.4 Shapes & Annotations (Editor Overlays)

These operate on the **Editor's annotation list** — bounding boxes and planes drawn as overlays on top of the scene. They're lightweight, editor-specific, and don't create scene layers.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/shapes/bboxes` | List all annotation bboxes |
| `POST` | `/shapes/bboxes` | Create annotation bbox — `{"label": "column", "center": [1,2,3], "size": [1,1,3], "color": [1, 0.5, 0]}` |
| `DELETE` | `/shapes/bboxes/{index}` | Delete annotation bbox by index |
| `PUT` | `/shapes/bboxes/{index}` | Update annotation bbox properties |
| `GET` | `/shapes/planes` | List all annotation planes |
| `POST` | `/shapes/planes` | Create annotation plane — `{"axis": "xy", "center": [0,0,0], "size": [10,10]}` |
| `DELETE` | `/shapes/planes/{index}` | Delete annotation plane by index |

### 5.5 Dynamic Geometry Layers

Dynamic layers are **first-class scene layers** created via the API. They appear in the layer panel alongside layers loaded from files, support visibility/opacity control, and can hold any geometry type: point clouds, triangle meshes (like STL), bounding boxes, or surface quads.

This is the backbone for real-time demos: load a base scene from E57 + folders, then create/modify/animate dynamic layers on top.

**Layer ID generation:** `layer_id = f"dyn_{name}"`. Names must be unique among dynamic layers. Creating with a duplicate name returns HTTP 409 Conflict. Example: `name="stretching_cube"` → `layer_id="dyn_stretching_cube"`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/scene/dynamic` | List all dynamic layers |
| `POST` | `/scene/dynamic` | Create a dynamic layer — see payload below |
| `GET` | `/scene/dynamic/{layer_id}` | Get dynamic layer geometry and properties |
| `PUT` | `/scene/dynamic/{layer_id}` | Update geometry and/or layer properties |
| `PATCH` | `/scene/dynamic/{layer_id}` | Partial property update (visibility, opacity, color) |
| `DELETE` | `/scene/dynamic/{layer_id}` | Remove dynamic layer from scene |
| `DELETE` | `/scene/dynamic` | Remove ALL dynamic layers |

#### Create Payload — Point Cloud

```json
{
  "name": "live_scan",
  "geometry_type": "pointcloud",
  "points": [[1,2,3], [4,5,6], ...],
  "colors": [[255,0,0], [0,255,0], ...],
  "color": [0.2, 0.8, 1.0],
  "visible": true,
  "opacity": 1.0
}
```

#### Create Payload — Triangle Mesh (STL-like)

```json
{
  "name": "demo_cube",
  "geometry_type": "mesh",
  "vertices": [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
  "triangles": [[0,1,2], [0,2,3], [4,5,6], [4,6,7], ...],
  "color": [1.0, 0.5, 0.0],
  "opacity": 0.8
}
```

#### Create Payload — BBox Layer (collection of boxes as a layer)

```json
{
  "name": "detected_objects",
  "geometry_type": "bboxes",
  "bboxes": [
    {"label": "rack", "center": [1,2,1.5], "size": [2,1,3], "color": [1,0.5,0], "fill_opacity": 0.1},
    {"label": "column", "center": [5,3,1.5], "size": [0.5,0.5,3], "color": [0,1,0]}
  ],
  "visible": true
}
```

#### Create Payload — Surface Quads Layer

```json
{
  "name": "detected_walls",
  "geometry_type": "surfaces",
  "surfaces": [
    {"axis": "xz", "center": [0,5,0], "size": [10,3], "color": [0.5,0.5,0.8], "opacity": 0.3}
  ],
  "visible": true
}
```

#### Create Payload — From File (load STL/OBJ/PLY as a dynamic layer)

```json
{
  "name": "reference_model",
  "geometry_type": "file",
  "path": "C:/models/bracket.stl",
  "color": [0.3, 0.9, 0.3],
  "opacity": 0.6
}
```

#### Update Payload (PUT — full geometry replace)

Same structure as create. Replaces all geometry, rebuilds VBOs.

#### Patch Payload (PATCH — property-only update, no geometry rebuild)

```json
{
  "visible": false,
  "opacity": 0.5,
  "color": [1, 0, 0]
}
```

The distinction between PUT (geometry replace) and PATCH (property tweak) matters for performance: PATCH never rebuilds VBOs and is suitable for 60Hz property animation, while PUT re-uploads geometry to the GPU.

### 5.6 Viewport Settings

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/viewport` | Current settings — `point_size`, `show_axes`, `show_grid`, `bg_color`, `vsync`, `fps_movement`, `point_attenuation`, `use_layer_colors` |
| `PUT` | `/viewport` | Update render settings — `{"point_size": 3, "show_grid": false}`. `vsync` is locked into the GL context at creation; PUT returns `vsync_restart_required` if a different value is requested. |
| `GET` | `/viewport/correction` | Scene correction state |
| `PUT` | `/viewport/correction` | Set correction — `{"rotate_x": -90, "shift_z": -1.2}` |
| `GET` | `/viewport/clip` | Current clip planes |
| `PUT` | `/viewport/clip` | Set AABB clip — `{"x_min": -5, "x_max": 5, ...}` |
| `DELETE` | `/viewport/clip` | Remove clipping |
| `GET` | `/viewport/fade` | Current cone-shadow shader fade state — see §5.7 |
| `PUT` | `/viewport/fade` | Configure / enable / disable the fade — see §5.7 |
| `GET` | `/viewport/render_mode` | Current render mode (`realtime` or `capture`) |
| `PUT` | `/viewport/render_mode` | Switch render mode (auto-managed by recording.start/stop in normal use) |

### 5.7 Cone-Shadow Shader Fade

A GLSL 1.20 fragment shader that **fades only point cloud vertices that
lie inside a cone swept from the camera through an "area of interest"
bounding sphere AND in front of the AoI's near edge**. Vertices off to
the side of the cone or behind the AoI keep their full alpha. Used by
the flyover demo to expose a search-region annotation as the camera
orbits without dimming the rest of the scene.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/viewport/fade` | `{enable, alpha_mul, band, aoi_center, aoi_radius, available}` |
| `PUT` | `/viewport/fade` | Update one or more fields |

```json
PUT /api/v1/viewport/fade
{
  "enable": true,
  "alpha_mul": 0.4,
  "band": 0.8,
  "aoi_center": [-1.5, -8.8, 2.8],
  "aoi_radius": 7.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `enable` | bool | When `false`, the shader is bypassed and the renderer falls back to the fixed-function path |
| `alpha_mul` | float `0..1` | Alpha multiplier for occluding points (default `0.5`; `0` = invisible) |
| `band` | float (m) | Smoothstep half-band around the AoI's near edge in world units (default `0.5`) |
| `aoi_center` | `[x,y,z]` | World-space center of the area of interest |
| `aoi_radius` | float (m) | Bounding-sphere radius of the AoI |
| `available` | bool _(read-only)_ | `true` if the shader compiled successfully on the current driver |

The shader uses only OpenGL 2.1 compatibility-profile built-ins
(`gl_ModelViewMatrix`, `gl_Vertex`, `gl_Color`, `gl_FragColor`) and
requires no GL upgrade. If compilation fails on the driver, the renderer
silently uses the fixed-function path and `available` reports `false`.

### 5.8 Video Recording

Records the GL viewport to mp4 with platform-native HW encoding,
software fallback, and explicit pause/resume on the same file.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/recording/encoders` | Probe ffmpeg + report which encoders the server will pick for each `(codec, hw_pref)` combination |
| `GET` | `/recording/status` | Live state (`idle` / `recording` / `paused` / `stopped`), config, frames written, bytes |
| `POST` | `/recording/start` | Spawn ffmpeg, switch engine into capture mode, lock UI input |
| `POST` | `/recording/pause` | Stop writing frames; the file stays open |
| `POST` | `/recording/resume` | Resume writing to the same file |
| `POST` | `/recording/stop` | Flush, finalize the file, restore engine, unlock UI |

#### `POST /recording/start` body

| Field | Default | Description |
|-------|---------|-------------|
| `path` | `<repo>/video/locul3d_<ts>.mp4` | Output. Relative paths placed under `<repo>/video/`; absolute paths used as-is. `.mp4` extension auto-added. |
| `resolution` | `"viewport"` | Preset name. Recognized: `viewport` (current widget size, HiDPI-aware, even-aligned), `4k`/`uhd` (3840×2160), `1080p`/`fhd`, `720p`/`hd`. |
| `width`, `height` | from `resolution` | Explicit dimensions. Always rounded down to even — yuv420p needs even sides. |
| `fps` | `60.0` | Output frame rate |
| `codec` | `"hevc"` | `"hevc"` or `"h264"`. Aliases accepted: `h265`, `avc`, `x264`, `x265`. |
| `hw` | `"auto"` | `"auto"` (try HW, fall back to SW), `"hw"` (HW only — fail if unavailable), `"sw"` (force libx264/libx265). |
| `bitrate_kbps` | computed | Override the auto-derived bitrate (~15 kbps/Mpx-30s for H.264, ~7.5 for HEVC). |
| `grid` | `null` (inherit) | `true`/`false` to override the viewer's `show_grid` for the recording. Restored on stop. |
| `axes` | `null` (inherit) | Same for `show_axes`. |
| `bg_color` | `null` (inherit) | `[r,g,b]` or `[r,g,b,a]` floats 0..1 to force a background color. Restored to the viewer theme on stop. |

#### Response (`/recording/start` and `/recording/status`)

```json
{
  "status": "ok",
  "config": {
    "path": "/repo/video/fly.mp4",
    "width": 3840, "height": 2160,
    "fps": 60.0, "codec": "hevc",
    "encoder": "hevc_videotoolbox",
    "encoder_kind": "hw",
    "bitrate_kbps": 124416,
    "show_grid": true,
    "show_axes": true,
    "bg_color": [0.08, 0.08, 0.12, 1.0]
  },
  "warnings": [
    "no hardware HEVC encoder available on linux; falling back to software"
  ]
}
```

`/recording/status` adds a `stats` block:

```json
{
  "stats": {
    "state": "recording",
    "frames_written": 137,
    "frames_dropped": 0,
    "bytes_written": 25812352,
    "duration_s": 2.28,
    "started_at": 12345.67,
    "last_error": null,
    "warnings": []
  }
}
```

#### `GET /recording/encoders`

```json
{
  "ffmpeg": "/opt/homebrew/bin/ffmpeg",
  "all": ["h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265", ...],
  "selection": {
    "h264": {
      "auto": {"encoder": "h264_videotoolbox", "kind": "hw", "warnings": []},
      "hw":   {"encoder": "h264_videotoolbox", "kind": "hw", "warnings": []},
      "sw":   {"encoder": "libx264",           "kind": "sw", "warnings": []}
    },
    "hevc": {
      "auto": {"encoder": "hevc_videotoolbox", "kind": "hw", "warnings": []},
      "hw":   {"encoder": "hevc_videotoolbox", "kind": "hw", "warnings": []},
      "sw":   {"encoder": "libx265",           "kind": "sw", "warnings": []}
    }
  }
}
```

#### Encoder priority

| Codec | macOS | Windows | Linux | SW fallback |
|-------|-------|---------|-------|-------------|
| H.264 | `h264_videotoolbox` | `h264_nvenc` → `h264_qsv` → `h264_amf` | `h264_nvenc` → `h264_vaapi` → `h264_qsv` | `libx264` |
| HEVC  | `hevc_videotoolbox` | `hevc_nvenc` → `hevc_qsv` → `hevc_amf` | `hevc_nvenc` → `hevc_vaapi` → `hevc_qsv` | `libx265` |

ffmpeg discovery: `LOCUL3D_FFMPEG` env var → `which ffmpeg` on PATH.

#### Pipeline

```
Engine._capture_tick (Qt main thread)
   │  advance virtual_clock by 1/fps
   │  tick all tracks at virtual_clock
   │
   ├─► viewport.render_to_buffer(W, H)        ── offscreen FBO render
   │     │
   │     └── glReadPixels → rgb24 bytes        ── samples=0 (no MSAA)
   │
   ├─► recorder.feed_frame(bytes)              ── bounded queue.put (4)
   │     │                                        backpressure: blocks
   │     │                                        when encoder is slow
   │     │
   │     └── writer thread → ffmpeg stdin
   │            │
   │            └── ffmpeg ── HW or SW encoder ── mp4
   │
   └─► viewport.update()                       ── widget repaint so the
                                                  operator sees a live
                                                  preview while UI is
                                                  locked
```

#### HEVC + QuickTime

HEVC outputs are tagged `hvc1` (not the libavformat default `hev1`)
unconditionally. QuickTime / Photos / iOS only accept `hvc1`; other
players accept both.

#### Failure semantics

- HW encoder request that can't be satisfied raises `EncoderUnavailable`
  and `recording.start` returns HTTP 400. Recording is not started; the
  viewport stays in its prior state.
- A render or write failure during recording calls `recorder.abort()`
  which closes ffmpeg with the partial output and `_stop_capture_session`
  unlocks the UI. The error is reported in `stats.last_error`.

---

## 6. WebSocket Protocol

Endpoint: `ws://localhost:8350/ws`

The WebSocket channel is designed for **low-latency, high-frequency** operations: real-time point cloud streaming, camera animation, and event notifications.

### 6.1 Message Format

All messages are JSON with a required `type` field:

```json
{
  "type": "command_type",
  "id": "optional-correlation-id",
  ...payload fields
}
```

Responses echo the `id` for correlation:

```json
{
  "type": "result",
  "id": "correlation-id",
  "status": "ok",
  "data": { ... }
}
```

Errors:

```json
{
  "type": "error",
  "id": "correlation-id",
  "code": "INVALID_PARAM",
  "message": "Human-readable error"
}
```

### 6.2 Command Types (Client → Server)

**Camera** — full state and individual params:

| Type | Payload | Description |
|------|---------|-------------|
| `camera.set` | `{azimuth?, elevation?, distance?, target?, fov?}` | Set camera (partial update — any subset of fields) |
| `camera.set_azimuth` | `{value: 90}` | Set azimuth only |
| `camera.set_elevation` | `{value: 30}` | Set elevation only |
| `camera.set_distance` | `{value: 50}` | Set distance only |
| `camera.set_fov` | `{value: 60}` | Set FOV only |
| `camera.set_target` | `{value: [x,y,z]}` | Set target only |
| `camera.preset` | `{preset: "Top"}` | Apply named preset |
| `camera.fit` | `{}` | Fit to scene |

**Layers** — visibility/opacity of any layer (file-loaded or dynamic):

| Type | Payload | Description |
|------|---------|-------------|
| `layer.set` | `{layer_id, visible?, opacity?, color?}` | Update layer property |
| `layer.set_all` | `{visible: bool}` | Show/hide all layers |

**Dynamic Geometry Layers** — create/modify scene geometry at runtime:

| Type | Payload | Description |
|------|---------|-------------|
| `dynamic.create` | `{name, geometry_type, ...geometry_data}` | Create dynamic layer (pointcloud/mesh/bboxes/surfaces/file) |
| `dynamic.update` | `{layer_id, ...full_geometry_data}` | Replace geometry (triggers VBO rebuild) |
| `dynamic.patch` | `{layer_id, visible?, opacity?, color?}` | Property-only update (no VBO rebuild, 60Hz safe) |
| `dynamic.delete` | `{layer_id}` | Remove dynamic layer |
| `dynamic.clear` | `{}` | Remove all dynamic layers |
| `dynamic.create_b64` | `{name, geometry_type, points_b64, colors_b64?}` | Create from Base64 float32 (high perf) |
| `dynamic.append_b64` | `{layer_id, points_b64, colors_b64?}` | Append points to existing layer |

**Annotations** — editor overlay shapes (not layers):

| Type | Payload | Description |
|------|---------|-------------|
| `bbox.create` | `{label, center, size, color?, rotation_z?, fill_opacity?}` | Create annotation bbox |
| `bbox.update` | `{index, ...fields}` | Update annotation bbox |
| `bbox.delete` | `{index}` | Delete annotation bbox |

**Animation** — camera and object animations (see §7 for full details):

> ⚠️ **Track ID field**: pass the user-supplied track id as `track_id`, *not*
> as `id`. The WebSocket transport uses the top-level `id` field as a
> request-correlation identifier and pops it before the payload reaches
> the animation engine. Existing scripts that pass `id` keep working
> via a fallback in the engine, but new code should use `track_id`.

| Type | Payload | Description |
|------|---------|-------------|
| `camera.animate` | `{track_id, keyframes, duration_ms, easing?, loop?, ping_pong?, repeat_count?}` | Keyframed camera animation |
| `camera.transform_continuous` | `{track_id, property, rate, duration_ms?}` | Continuous camera property change (e.g., 10°/sec orbit) |
| `dynamic.animate` | `{track_id, layer_id, keyframes, duration_ms, easing?, loop?, ping_pong?}` | Keyframed object animation (position, scale, rotation, color, opacity) |
| `dynamic.transform_continuous` | `{track_id, layer_id, property, rate, duration_ms?}` | Continuous layer property change |
| `dynamic.transform` | `{layer_id, position?, rotation_z?, scale?, color?, opacity?}` | One-shot instant transform |
| `animation.stop` | `{track_id}` | Stop a specific animation/transform by ID |
| `transform.stop` | `{track_id}` | Stop a specific continuous transform by ID |
| `transform.stop_all` | `{}` | Stop all continuous transforms |

**Adaptive realtime FPS controller** — closed-loop GPU-cost-aware
playback gate (see §8.6):

| Type | Payload | Description |
|------|---------|-------------|
| `animation.set_realtime_fps` | `{fps?, min_fps?, adaptive?}` | Set the realtime FPS ceiling, floor, and enable/disable adaptive control |
| `animation.get_realtime_fps` | `{}` | Returns `{max_fps, min_fps, effective_fps, adaptive, paint_peak_ms, paint_p80_ms, paint_ema_ms, paint_last_ms, paint_last_cpu_ms, time_scale_*}` |
| `animation.set_preview_mode` | `{enable, target_fps?, min_pts?, max_pts?}` | Hold target FPS by adapting LOD (global vertex stride budget) instead of dropping FPS |
| `animation.set_time_scale` | `{auto?, scale?, nominal_fps?}` | Slow the virtual animation clock when `effective_fps < nominal_fps` so each rendered frame represents one nominal-fps step (slow but smooth, instead of fast and jumpy) |

**Render Modes** — realtime vs capture (see §8 for full details):

| Type | Payload | Description |
|------|---------|-------------|
| `render.set_mode` | `{mode: "realtime"\|"capture", width?, height?}` | Switch render mode |
| `render.get_mode` | `{}` | Query current render mode |
| `render.capture_frame` | `{save_to?, format?: "png"}` | Render one frame (capture mode only) |
| `render.set_target_fps` | `{fps: 60}` | Set capture-mode step size (1/fps seconds) |

**Viewport & Scene:**

| Type | Payload | Description |
|------|---------|-------------|
| `viewport.set` | `{point_size?, show_axes?, show_grid?, bg_color?}` | Update render settings |
| `correction.set` | `{rotate_x?, rotate_y?, rotate_z?, shift_x?, shift_y?, shift_z?}` | Update scene correction |
| `clip.set` | `{x_min, x_max, y_min, y_max, z_min, z_max}` | Set AABB clipping |
| `clip.clear` | `{}` | Remove clipping |
| `scene.load` | `{paths: [...]}` | Load files |
| `scene.clear` | `{}` | Clear scene |
| `screenshot.capture` | `{format?: "png"}` | Capture screenshot (returned as base64) |

### 6.3 Event Types (Server → Client)

The server pushes events to all connected WS clients:

| Type | Payload | Trigger |
|------|---------|---------|
| `event.fps` | `{fps: float}` | Every 1s — current FPS |
| `event.camera` | `{azimuth, elevation, distance, target}` | Camera position changed (throttled to 30Hz) |
| `event.layer_changed` | `{layer_id, visible, opacity}` | Layer visibility/opacity changed in UI |
| `event.scene_loaded` | `{layers: [{id, name, type, point_count}]}` | Files loaded |
| `event.scene_cleared` | `{}` | Scene cleared |
| `event.dynamic_created` | `{layer_id, name, geometry_type}` | Dynamic layer created |
| `event.dynamic_updated` | `{layer_id}` | Dynamic layer geometry replaced |
| `event.dynamic_deleted` | `{layer_id}` | Dynamic layer removed |
| `event.bbox_created` | `{index, bbox}` | Annotation added |
| `event.bbox_deleted` | `{index}` | Annotation removed |
| `event.animation_started` | `{id, type}` | Animation/transform track started |
| `event.animation_done` | `{id}` | Animation completed (on final cycle if looping) |
| `event.transform_started` | `{id, property}` | Continuous transform started |
| `event.transform_stopped` | `{id, reason: "manual"\|"expired"}` | Continuous transform stopped |
| `event.render_mode_changed` | `{mode, width?, height?}` | Render mode switched |

### 6.4 Binary Mode (High-Performance Point Streaming)

For maximum throughput (millions of points/sec), the WS channel supports a **binary message mode**:

```
Byte layout:
  [0..3]   uint32  message_type (1=create_points, 2=append_points)
  [4..7]   uint32  name_length (N)
  [8..8+N] utf8    layer_name
  [8+N..]  float32 interleaved XYZ XYZ ... (points)
```

This avoids JSON serialization overhead for large point arrays. The REST endpoint `POST /scene/dynamic` with `geometry_type: "pointcloud"` remains available for smaller payloads using JSON arrays.

---

## 7. Server-Side Animation Engine

The animation engine runs entirely on the **Qt main thread** via a high-frequency `QTimer`. This ensures animations are buttery smooth regardless of network latency — the client sends a *declaration* of what to animate, and the server ticks it locally.

### 7.1 Architecture

```
┌────────────────────────────────────────────────────────┐
│                  AnimationEngine                        │
│                                                        │
│  QTimer (8ms / 120Hz)                                  │
│   ├── tick() ─┬── camera_tracks[]  → update viewport   │
│   │           ├── layer_tracks[]   → update layer props │
│   │           └── emit frame_ready signal               │
│   │                                                    │
│  Active tracks:                                        │
│   • CameraTrack:  keyframe interpolation for cam       │
│   • ContinuousRotation: constant °/sec camera orbit    │
│   • LayerPropertyTrack: animate opacity/color/visible  │
│   • TransformTrack: per-layer position/rotation anim   │
│                                                        │
│  Render Mode:                                          │
│   • REALTIME: LOD active, FPS counted, timer-driven    │
│   • CAPTURE:  full quality, frame-by-frame, no LOD     │
└────────────────────────────────────────────────────────┘
```
*Note: Every animation requires an `id`. Starting a new animation with an existing `id` automatically cancels and replaces the previous one. This prevents unintended stacking of transforms.*

### 7.2 Timer Design

```python
class AnimationEngine(QObject):
    frame_ready = Signal(int)  # frame number (for capture mode)

    def __init__(self, viewport):
        super().__init__()
        self._viewport = viewport
        self._tracks: list[AnimationTrack] = []
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(8)  # 125Hz base tick (renders at vsync)
        self._frame_number = 0
        self._render_mode = "realtime"  # or "capture"

    def _tick(self):
        t = time.perf_counter()
        dirty = False
        for track in self._tracks:
            if track.tick(t):
                dirty = True
        # Remove completed tracks
        self._tracks = [t for t in self._tracks if not t.done]

        if dirty:
            if self._render_mode == "capture":
                # Full quality render — disable LOD
                self._viewport._interacting = False
                self._viewport.update()
                self._viewport.repaint()  # force synchronous paint
                self.frame_ready.emit(self._frame_number)
                self._frame_number += 1
            else:
                self._viewport.update()  # async, vsync-driven
```

The timer runs at **125Hz** (8ms interval) using `Qt.PreciseTimer` for accurate timing. The actual render rate is determined by vsync or explicit `repaint()` in capture mode. Animation math uses `time.perf_counter()` deltas — never frame counting — so animations stay time-accurate even if frames drop.

### 7.3 Camera Keyframe Animation

```json
{
  "type": "camera.animate",
  "id": "demo-orbit",
  "keyframes": [
    {"t": 0.0,  "azimuth": 0,   "elevation": 30, "distance": 50},
    {"t": 0.5,  "azimuth": 180, "elevation": 45, "distance": 40},
    {"t": 1.0,  "azimuth": 360, "elevation": 30, "distance": 50}
  ],
  "duration_ms": 5000,
  "easing": "ease_in_out",
  "loop": false,
  "ping_pong": false
}
```

Zoom in/out loop example:
```json
{
  "type": "camera.animate",
  "id": "zoom-pulse",
  "keyframes": [
    {"t": 0.0, "distance": 80, "fov": 45},
    {"t": 1.0, "distance": 30, "fov": 60}
  ],
  "duration_ms": 3000,
  "easing": "ease_in_out",
  "loop": true,
  "ping_pong": true
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `loop` | `false` | Repeat the animation indefinitely |
| `ping_pong` | `false` | Reverse direction each cycle (A→B→A→B...) instead of snapping back (A→B, A→B...) |
| `repeat_count` | `0` | If `loop=true`, stop after N repeats. `0` = infinite |

- Keyframe `t` values are normalized (0.0–1.0), interpolated with the selected easing
- Camera parameters interpolated independently; angles use shortest-arc
- `duration_ms: 0` instantly jumps to the last keyframe
- Events: `event.animation_started`, `event.animation_done` (on final cycle if looping)
- Stop a looping animation: `{"type": "animation.stop", "id": "zoom-pulse"}`

### 7.4 Continuous Transforms

Any camera or layer property can be continuously changed at a fixed rate — one-shot, for a specific duration, or forever until stopped.

**Camera continuous transform:**

```json
{
  "type": "camera.transform_continuous",
  "id": "turntable",
  "property": "azimuth",
  "rate": 10.0,
  "duration_ms": 0
}
```

| Field | Description |
|-------|-------------|
| `property` | Any camera property: `"azimuth"`, `"elevation"`, `"distance"`, `"fov"` |
| `rate` | Change per second (degrees for angles, units for distance/FOV). Negative = reverse |
| `duration_ms` | `0` = run forever, `>0` = stop after N ms |

**Dynamic layer continuous transform:**

```json
{
  "type": "dynamic.transform_continuous",
  "id": "spin_cube",
  "layer_id": "stretching_cube",
  "property": "rotation_z",
  "rate": 45.0,
  "duration_ms": 5000
}
```

| `property` values | Description |
|-------------------|-------------|
| `"rotation_z"` | Rotate around Z axis (°/sec) |
| `"position_x"`, `"position_y"`, `"position_z"` | Translate (units/sec) |
| `"scale_x"`, `"scale_y"`, `"scale_z"` | Scale (factor/sec) |
| `"opacity"` / `"alpha"` | Fade (0–1 range, per second) |
| `"color_r"`, `"color_g"`, `"color_b"` | Individual color channel (0–1, per second) |
| `"color"` | Full RGB transition — uses `target` field instead of `rate` (see below) |

For `property: "color"`, use `target` instead of `rate` — the color interpolates linearly from current to target over `duration_ms`:
```json
{
  "type": "dynamic.transform_continuous",
  "id": "fade_red",
  "layer_id": "cube",
  "property": "color",
  "target": [1, 0, 0],
  "duration_ms": 2000
}
```
When `duration_ms` expires, `event.transform_stopped` fires with `reason: "expired"`.

**Stop a continuous transform:**
```json
{"type": "transform.stop", "id": "turntable"}
```

**Stop all:**
```json
{"type": "transform.stop_all"}
```

**One-shot instant transform** (no animation, immediate):
```json
{
  "type": "dynamic.transform",
  "layer_id": "stretching_cube",
  "position": [1, 2, 0],
  "rotation_z": 45.0,
  "scale": [1, 1, 2],
  "color": [1.0, 0.3, 0.0],
  "opacity": 0.8
}
```

All continuous transforms use wall-clock `dt` from `time.perf_counter()` — speed is frame-rate independent. Multiple continuous transforms can run simultaneously on different properties (e.g., camera orbits while a cube spins, changes color, and another object translates).

### 7.5 Independent Object Animation (Keyframed)

Dynamic layers can be independently animated via keyframed tracks — including position, rotation, scale, color, and opacity:

```json
{
  "type": "dynamic.animate",
  "layer_id": "stretching_cube",
  "keyframes": [
    {"t": 0.0, "scale": [1,1,1], "position": [0,0,0], "rotation_z": 0,
              "color": [1.0, 0.3, 0.1], "opacity": 1.0},
    {"t": 0.5, "scale": [1,1,3], "position": [0,0,1], "rotation_z": 45,
              "color": [0.1, 0.8, 1.0], "opacity": 0.6},
    {"t": 1.0, "scale": [1,1,1], "position": [0,0,0], "rotation_z": 90,
              "color": [1.0, 0.3, 0.1], "opacity": 1.0}
  ],
  "duration_ms": 3000,
  "loop": true,
  "ping_pong": true,
  "easing": "ease_in_out"
}
```

**Animatable properties per keyframe:**

| Property | Type | Description |
|----------|------|-------------|
| `position` | `[x,y,z]` | GL-level translation |
| `rotation_z` | `float` | Z-axis rotation (degrees) |
| `scale` | `[x,y,z]` | GL-level scale |
| `color` | `[r,g,b]` | Layer color (0–1, interpolated in linear RGB or sRGB) |
| `opacity` | `float` | Alpha channel (0–1) |
| `point_size` | `float` | Point size override for point cloud layers |

**Key design:** Transform + color animation applies GL-level transforms and uniform updates — it does **not** modify vertex data or rebuild VBOs. This means:
- Zero GPU re-upload cost per frame
- Multiple objects can animate independently at 120Hz
- Color/opacity transitions are per-uniform, not per-vertex (instant)
- The camera can orbit simultaneously via its own track

**Per-vertex color animation** (for advanced use):
To animate individual point/vertex colors (e.g., heatmap pulse), use `dynamic.update` with new color arrays. This triggers a VBO rebuild — not suitable for 120Hz but fine for periodic updates (1–10 Hz).

Combined with continuous camera rotation, this enables scenes like:
- Camera orbits at 10°/sec
- Cube stretches, rotates, and pulses color
- Point cloud layers fade in/out via opacity tracks
- All at 60–120 FPS with LOD active

### 7.6 Timing & Easing System

The animation timing API is modeled after **Core Animation** (iOS/macOS) and the **Web Animations API** — the same concepts used in `CAMediaTimingFunction` and CSS `animation-timing-function`. All animation commands (`camera.animate`, `dynamic.animate`, continuous transforms) accept an `easing` field.

#### Named Presets

| Name | Equivalent | Curve |
|------|------------|-------|
| `"linear"` | `cubic-bezier(0, 0, 1, 1)` | Constant speed |
| `"ease"` | `cubic-bezier(0.25, 0.1, 0.25, 1.0)` | CSS default — gentle S-curve |
| `"ease_in"` | `cubic-bezier(0.42, 0, 1.0, 1.0)` | Slow start, fast end |
| `"ease_out"` | `cubic-bezier(0, 0, 0.58, 1.0)` | Fast start, slow end |
| `"ease_in_out"` | `cubic-bezier(0.42, 0, 0.58, 1.0)` | Smooth S-curve (default) |
| `"ease_in_cubic"` | `cubic-bezier(0.32, 0, 0.67, 0)` | Aggressive slow start |
| `"ease_out_cubic"` | `cubic-bezier(0.33, 1, 0.68, 1)` | Aggressive slow end |
| `"ease_in_out_cubic"` | `cubic-bezier(0.65, 0, 0.35, 1)` | Pronounced S-curve |
| `"ease_in_back"` | `cubic-bezier(0.36, 0, 0.66, -0.56)` | Pulls back before starting |
| `"ease_out_back"` | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Overshoots before settling |
| `"ease_out_bounce"` | Custom | Bouncing settle effect |

#### Custom Cubic Bézier

Exact CSS `cubic-bezier()` syntax for precise control:

```json
{"easing": {"cubic_bezier": [0.17, 0.67, 0.83, 0.67]}}
```

This is the workhorse — Core Animation’s `CAMediaTimingFunction(controlPoints:)` and CSS both use this exact 4-parameter form.

#### Spring Physics

Natural spring-based easing for organic motion (like `CASpringAnimation` / `UIView.animate(usingSpringWithDamping:)`):

```json
{
  "easing": {
    "spring": {
      "damping": 0.6,
      "stiffness": 100,
      "mass": 1.0,
      "initial_velocity": 0.0
    }
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `damping` | `0.7` | Damping ratio (0–1, lower = more bouncy) |
| `stiffness` | `100` | Spring constant (higher = snappier) |
| `mass` | `1.0` | Mass of the animated object |
| `initial_velocity` | `0.0` | Starting velocity |

Use cases: camera "snap" to preset, bbox appearing with overshoot, zoom bounce.

#### Step Functions

For frame-by-frame or staggered animations (like CSS `steps()`):

```json
{"easing": {"steps": {"count": 10, "position": "end"}}}
```

| `position` | Behavior |
|------------|----------|
| `"start"` | Jump at the beginning of each step |
| `"end"` | Jump at the end of each step (default) |

Use cases: layer-by-layer reveal, text ticker, discrete visibility toggling.

#### Per-Property Easing

For keyframed animations, different easing can be applied to each property — for example, position eases out while color transitions linearly:

```json
{
  "type": "dynamic.animate",
  "layer_id": "indicator",
  "keyframes": [
    {"t": 0.0, "position": [0,0,0], "color": [1,0,0]},
    {"t": 1.0, "position": [5,0,0], "color": [0,1,0]}
  ],
  "duration_ms": 2000,
  "easing": {
    "position": "ease_out",
    "color": "linear"
  }
}
```

#### Implementation

```python
# animation/easing.py

def resolve_easing(spec) -> Callable[[float], float]:
    """Convert easing spec (string, dict, or callable) to a t->t function."""
    if isinstance(spec, str):
        return NAMED_PRESETS[spec]
    if isinstance(spec, dict):
        if "cubic_bezier" in spec:
            return cubic_bezier(*spec["cubic_bezier"])
        if "spring" in spec:
            return spring_timing(**spec["spring"])
        if "steps" in spec:
            return step_timing(**spec["steps"])
    return linear  # fallback

def cubic_bezier(x1, y1, x2, y2) -> Callable[[float], float]:
    """Standard cubic Bézier curve — same algo as CSS/Core Animation."""
    ...

def spring_timing(damping, stiffness, mass, initial_velocity) -> Callable[[float], float]:
    """Critically-damped spring solver — matches CASpringAnimation."""
    ...
```

---

## 8. Render Modes

The API exposes two render modes to support both interactive use and high-quality recording.

### 8.1 Realtime Mode (Default)

Standard interactive rendering with all optimizations active:

- **LOD/decimation**: Stride-based point decimation during interaction (existing `INTERACTIVE_BUDGET`)
- **VSync-driven**: Renders at display refresh rate
- **FPS counting**: Status bar and `event.fps` reports accurate FPS
- **Animation timer**: 125Hz tick, renders asynchronously

### 8.2 Capture Mode

Full-quality frame-by-frame rendering for video recording:

```json
{"type": "render.set_mode", "mode": "capture", "width": 1920, "height": 1080}
```

| Property | Behavior |
|----------|----------|
| **LOD** | Disabled — all points rendered at full density |
| **Stride** | Always 1 — no interactive decimation |
| **FPS** | Not counted — each frame renders on explicit request |
| **Resolution** | Viewport resized to specified width × height |
| **Frame export** | Each frame saved as PNG or returned via WS |

In capture mode, the animation engine switches from timer-driven to **stepped**:

```json
{"type": "render.capture_frame"}
```

Each `capture_frame` command:
1. Advances the animation engine by exactly `1/target_fps` seconds (e.g., 1/60s)
2. Renders the frame at full quality (no LOD, no stride)
3. Returns the frame as base64 PNG via WS response
4. Optionally saves to disk: `{"type": "render.capture_frame", "save_to": "C:/frames/frame_%04d.png"}`

This ensures **deterministic, frame-perfect** output regardless of actual rendering speed.

### 8.3 Capture Sequence

Full workflow for recording a scripted demo to video frames:

```python
ws = websocket.create_connection(WS)

# 1. Switch to capture mode
ws.send(json.dumps({
    "type": "render.set_mode",
    "mode": "capture",
    "width": 1920,
    "height": 1080
}))

# 2. Start camera animation (won't auto-advance in capture mode)
ws.send(json.dumps({
    "type": "camera.animate",
    "id": "flythrough",
    "keyframes": [...],
    "duration_ms": 10000,
    "easing": "ease_in_out"
}))

# 3. Render frame by frame at 60fps
TARGET_FPS = 60
DURATION = 10.0  # seconds
for i in range(int(TARGET_FPS * DURATION)):
    ws.send(json.dumps({
        "type": "render.capture_frame",
        "save_to": f"C:/frames/frame_{i:04d}.png"
    }))
    # Wait for frame confirmation
    msg = json.loads(ws.recv())
    assert msg["type"] == "result"

# 4. Switch back to realtime
ws.send(json.dumps({"type": "render.set_mode", "mode": "realtime"}))
ws.close()

# 5. Use ffmpeg to assemble video
# ffmpeg -framerate 60 -i C:/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo.mp4
```

### 8.4 WS Commands

| Type | Payload | Description |
|------|---------|-------------|
| `render.set_mode` | `{mode: "realtime"\|"capture", width?, height?}` | Switch render mode |
| `render.get_mode` | `{}` | Query current mode |
| `render.capture_frame` | `{save_to?, format?: "png"}` | Render one frame (capture mode only) |
| `render.set_target_fps` | `{fps: 60}` | Set capture mode step size (1/fps seconds) |

### 8.5 REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/viewport/render_mode` | Current render mode and capture settings |
| `PUT` | `/viewport/render_mode` | Switch mode — `{"mode": "capture", "width": 1920, "height": 1080}` |

### 8.6 Adaptive Realtime + Recording-Driven Capture (current implementation)

The original §8.1-8.5 designed capture as a request/response loop where
the client polls `render.capture_frame`. The shipped implementation
inverts that: capture is driven by the **AnimationEngine itself** the
moment a recording is active, and the realtime mode now has its own
**closed-loop adaptive FPS controller** so the realtime preview gives
honest feedback even on huge scenes.

#### Realtime mode (closed-loop adaptive)

Drives the playback gate from the GPU's actual paint cost.

```
QTimer (500 Hz) ─┬─► tick all tracks at virtual_clock
                 │
                 ├─► _adapt_effective_fps()
                 │     paint_peak = viewport._paint_peak_ms
                 │     period_ms = paint_peak * 1.25 + cooldown_ms (2)
                 │     eff_fps   = clamp(min, max, 1000/period_ms)
                 │     snap to 5-FPS grid
                 │     hysteresis: drops instant, rises ≥ 1 step
                 │
                 ├─► gate (both must be satisfied)
                 │     since_last_render_request >= 1000/eff_fps
                 │     since_last_paint_end       >= cooldown_ms
                 │     and not _paint_in_progress
                 │
                 └─► viewport.update()       (cheap, schedules paint)
```

The paint cost signal is **`_paint_peak_ms`** — a slow-decaying peak of
the per-frame GPU duration measured via `GL_TIME_ELAPSED` queries
(non-blocking, with a 4-deep in-flight pool). The peak jumps up
instantly on spikes and decays at ~3% per frame, so the controller
remembers the worst case for ~30 frames and doesn't oscillate as the
camera orbits past dense regions.

Why GL queries instead of `glFinish()`: on macOS the Metal command
buffer drain has 5-15 ms of fixed overhead per `glFinish` call, which
would dwarf real GPU work on cheap scenes and force the controller to
report low FPS even when the GPU is idle. Timer queries report true GPU
nanoseconds with no CPU stall.

##### Virtual animation clock

When `eff_fps < nominal_fps` (default 60), the **virtual animation
clock** advances at `eff_fps / nominal_fps × wall_dt`. Tracks see this
slower clock, so each rendered frame represents one nominal-fps step
regardless of how long the GPU spent painting it. A 15-second nominal
flyover plays out over 180 wall seconds at 5 FPS, with smooth
inter-frame motion instead of huge jumps.

Disable via `animation.set_time_scale {auto: false, scale: 1.0}`.

##### Preview mode

Inverse trade-off: hold target FPS by adapting LOD. The engine sets
`_preview_mode=True` on the viewport, which switches the per-layer
stride logic to use a **dynamic global vertex budget**. The same
asymmetric controller tunes that budget instead of FPS:

- `paint > 1.15 × target_dt` → budget *= 0.7
- `paint < 0.55 × target_dt` → budget *= 1.10 + 50k
- snapped to 250k-vertex steps

Activated via `animation.set_preview_mode {enable: true, target_fps: 60}`.
Mutually exclusive with the full-res adaptive path; the controllers
won't fight.

#### Recording-driven capture mode

Recording is started via `POST /api/v1/recording/start` (see §5.8),
which:

1. Spawns ffmpeg with the selected encoder.
2. Switches the engine to `_render_mode = "capture"` and attaches the
   recorder to the engine.
3. Saves the viewer's current `show_grid` / `show_axes` / `bg_color`
   into `_rec_overrides` and applies any per-recording overrides.
4. Locks UI input on both the viewport AND the parent window
   (`window.setEnabled(False)`).

While recording, the engine's `_capture_tick` runs once per QTimer
tick (500 Hz) and:

1. Advances `virtual_clock` by *exactly* `1/fps` (wall-clock independent).
2. Ticks all tracks at the new virtual time.
3. If the recorder is **paused**, returns without rendering — the
   animation effectively freezes in place.
4. Otherwise: renders the scene to an offscreen FBO at the recording
   resolution (`viewport.render_to_buffer(W, H)`) and pushes the rgb24
   bytes to `recorder.feed_frame()`. The recorder's bounded queue is
   the natural backpressure mechanism — `feed_frame` blocks when the
   encoder is slower than the renderer, throttling the engine to
   encoder speed.
5. Calls `viewport.update()` so the editor widget also repaints,
   giving the operator a live preview while UI is locked.
6. When the last track finishes, calls `_stop_capture_session()`
   which stops the recorder, restores `_rec_overrides`, switches the
   engine back to realtime, and unlocks input on viewport AND window.

The same `_stop_capture_session()` runs from the explicit
`POST /recording/stop` path, so natural completion and explicit stop
are guaranteed to leave the editor in identical state.

Capture mode bypasses the realtime FPS gate, the adaptive FPS
controller, and the auto time-scale entirely — those are realtime
concerns. The recorder needs frame-exact determinism.

##### Offscreen FBO

`BaseGLViewport.render_to_buffer(width, height)` creates a
`QOpenGLFramebufferObject` with `samples=0` (multisample FBOs are not
directly readable via `glReadPixels` — undefined results, manifests as
random "wrong opacity / brightness" frames in the output). Width and
height come from the recorder config; the viewport's `_capture_w` /
`_capture_h` overrides are set so the projection-matrix code uses the
FBO's aspect ratio instead of the widget's.

Pixel format: `GL_RGB / GL_UNSIGNED_BYTE` (rgb24). The framebuffer is
bottom-up; ffmpeg's `vflip` filter flips it on the way to the encoder,
so we don't pay for a CPU row reversal at 4K.

##### Input lock

`viewport.set_input_locked(True)` short-circuits all mouse / keyboard
event handlers (`mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent`,
`mouseDoubleClickEvent`, `wheelEvent`, `keyPressEvent`). The dispatcher
also calls `window.setEnabled(False)` which greys out toolbars, panels,
menu bar — every widget the user could click. Re-enabled symmetrically
on stop / abort / natural completion via the same dispatcher helper, so
all paths converge on a fully responsive editor.

---

## 9. Thread Bridge: asyncio ↔ Qt

The critical design challenge is safely crossing from the aiohttp asyncio thread to the Qt main thread:

```python
# bridge.py

import asyncio
from PySide6.QtCore import QMetaObject, Qt, Q_ARG, QObject, Slot
from concurrent.futures import Future
from typing import Callable, Any


class QtBridge(QObject):
    """Bridge between asyncio (aiohttp thread) and Qt main thread.

    All viewer mutations go through invoke_on_qt(), which:
    1. Creates a concurrent.futures.Future for the result
    2. Posts a QMetaObject.invokeMethod to Qt's event loop 
    3. The Qt slot executes the callable and resolves the future
    4. The asyncio side awaits the future via run_in_executor
    """

    def __init__(self):
        super().__init__()

    @Slot(object, object)
    def _execute(self, fn: Callable, future: Future):
        """Runs on Qt main thread — executes fn and resolves future."""
        try:
            result = fn()
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    async def invoke_on_qt(self, fn: Callable[[], Any]) -> Any:
        """Call fn() on the Qt main thread, await result from asyncio."""
        future = Future()
        QMetaObject.invokeMethod(
            self,
            "_execute",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, fn),
            Q_ARG(object, future),
        )
        loop = asyncio.get_event_loop()
        return await asyncio.wrap_future(future, loop=loop)
```

### Safety Guarantees

1. **No concurrent GL access** — all viewport mutations execute on the Qt event loop
2. **No UI deadlocks** — `QueuedConnection` is fully async; the aiohttp thread never blocks the Qt thread
3. **Error propagation** — exceptions in Qt-side code propagate cleanly to the HTTP response
4. **Cancellation** — if the HTTP request is cancelled, the Qt-side callable still runs (fire-and-forget) but the result is discarded

---

## 10. Server Lifecycle

### Startup

The server is started automatically when the viewer/editor launches, or on-demand via a toolbar toggle:

```python
# In ViewerWindow.__init__ or via toolbar action:
from locul3d.remote import start_server

self._remote_server = start_server(
    window=self,
    port=8350,       # configurable via --api-port CLI arg
    host="127.0.0.1" # localhost only by default (security)
)
```

### Startup sequence

1. `start_server()` creates a `QtBridge` on the main thread
2. Spawns a daemon thread running `asyncio.run(serve(app, host, port))`
3. The aiohttp `Application` registers all REST routes and the WS endpoint
4. Server is ready — prints `Remote API listening on http://127.0.0.1:8350`

### Shutdown

- Server stops when the viewer window closes (`closeEvent`)
- Graceful shutdown via `runner.cleanup()` + thread join (1s timeout)
- All WS connections are closed with code 1001 (Going Away)

### CLI Integration

```bash
# Default: API enabled on port 8350
python start.py scan.ply

# Custom port
python start.py scan.ply --api-port 9000

# Disable API server
python start.py scan.ply --no-api

# Bind to all interfaces (for remote access)
python start.py scan.ply --api-host 0.0.0.0
```

---

## 11. OpenAPI Spec Generation

The `/openapi.json` endpoint serves a complete OpenAPI 3.1 specification, generated from the Pydantic models in `schemas.py`:

```python
# openapi.py — generates spec from registered handlers + schemas

def generate_openapi_spec(routes, schemas) -> dict:
    """Build OpenAPI 3.1 spec from aiohttp routes and Pydantic models."""
    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "Locul3D Remote Control API",
            "version": "1.0.0",
            "description": "Control the Locul3D 3D viewer remotely"
        },
        "servers": [{"url": "http://localhost:8350"}],
        "paths": {},
        "components": {"schemas": {}}
    }
    # ... route introspection and schema injection
    return spec
```

This enables:
- **Swagger UI** (just serve `swagger-ui` pointed at `/openapi.json`)
- **Client codegen** via `openapi-generator` (Python, TypeScript, etc.)
- **Postman import** for interactive testing

---

## 12. Pydantic Schemas

```python
# schemas.py

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Literal, Union
from enum import Enum


# --- Easing ---

class SpringEasing(BaseModel):
    """Spring physics easing — matches CASpringAnimation."""
    damping: float = Field(0.7, ge=0, le=2)
    stiffness: float = Field(100, ge=0)
    mass: float = Field(1.0, ge=0.01)
    initial_velocity: float = 0.0


class StepsEasing(BaseModel):
    """Step function easing — matches CSS steps()."""
    count: int = Field(..., ge=1)
    position: Literal["start", "end"] = "end"


# Easing can be a named preset string, a cubic-bezier 4-tuple, spring, or steps
EasingSpec = Union[
    str,                                        # Named preset: "ease_in_out"
    dict,                                       # {"cubic_bezier": [x1,y1,x2,y2]}
                                                # {"spring": {damping, stiffness, ...}}
                                                # {"steps": {count, position}}
]


# --- Camera ---

class CameraState(BaseModel):
    azimuth: float = Field(45.0, description="Horizontal rotation (degrees)")
    elevation: float = Field(30.0, description="Vertical rotation (degrees)")
    distance: float = Field(50.0, description="Distance from target")
    target: List[float] = Field([0, 0, 0], min_length=3, max_length=3)
    fov: float = Field(45.0, ge=1, le=170, description="Field of view (degrees)")


class CameraUpdate(BaseModel):
    """Partial camera update — set any subset of fields."""
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None
    target: Optional[List[float]] = None
    fov: Optional[float] = None


class ScalarValue(BaseModel):
    """Single numeric value for individual parameter setters."""
    value: float


class Vec3Value(BaseModel):
    """3D vector for individual parameter setters (target)."""
    value: List[float] = Field(..., min_length=3, max_length=3)


class Keyframe(BaseModel):
    t: float = Field(..., ge=0.0, le=1.0, description="Normalized time 0-1")
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None
    target: Optional[List[float]] = None
    fov: Optional[float] = None


class CameraAnimation(BaseModel):
    keyframes: List[Keyframe]
    duration_ms: int = Field(3000, ge=0)
    easing: EasingSpec = "ease_in_out"  # str | dict (cubic_bezier/spring/steps)
    loop: bool = False
    ping_pong: bool = False
    repeat_count: int = Field(0, ge=0, description="0 = infinite when loop=True")


# --- Layers ---

class LayerInfo(BaseModel):
    id: str
    name: str
    type: str  # "pointcloud", "mesh", "wireframe", "panorama", "dynamic_*"
    visible: bool
    opacity: float
    point_count: int
    tri_count: int
    dynamic: bool = False  # True if created via API


class LayerUpdate(BaseModel):
    visible: Optional[bool] = None
    opacity: Optional[float] = Field(None, ge=0.0, le=1.0)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)


# --- Dynamic Geometry Layers ---

class GeometryType(str, Enum):
    POINTCLOUD = "pointcloud"
    MESH = "mesh"
    BBOXES = "bboxes"
    SURFACES = "surfaces"
    FILE = "file"


class BBoxSpec(BaseModel):
    """A single bbox within a bboxes-type dynamic layer."""
    label: str = "custom"
    center: List[float] = Field(..., min_length=3, max_length=3)
    size: List[float] = Field(..., min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: float = 0.0
    fill_opacity: float = 0.0


class SurfaceSpec(BaseModel):
    """A single surface quad within a surfaces-type dynamic layer."""
    axis: str = "xy"  # "xy", "xz", "yz"
    center: List[float] = Field([0, 0, 0], min_length=3, max_length=3)
    size: List[float] = Field([10, 10], min_length=2, max_length=2)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: float = 0.3


class DynamicLayerCreate(BaseModel):
    """Create a dynamic geometry layer.

    The geometry_type determines which geometry fields are required:
    - pointcloud: points (required), colors (optional)
    - mesh:       vertices + triangles (required), normals (optional)
    - bboxes:     bboxes list (required)
    - surfaces:   surfaces list (required)
    - file:       path (required)

    Layer ID is generated as f"dyn_{name}". Names must be unique.
    """
    name: str
    geometry_type: GeometryType
    visible: bool = True
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)

    # Pointcloud geometry
    points: Optional[List[List[float]]] = None     # Nx3
    colors: Optional[List[List[float]]] = None     # Nx3 (0-255 int or 0-1 float)

    # Mesh geometry
    vertices: Optional[List[List[float]]] = None   # Nx3
    triangles: Optional[List[List[int]]] = None    # Mx3 vertex indices
    normals: Optional[List[List[float]]] = None    # Nx3

    # BBox collection
    bboxes: Optional[List[BBoxSpec]] = None

    # Surface collection
    surfaces: Optional[List[SurfaceSpec]] = None

    # File-based
    path: Optional[str] = None  # absolute path to STL/OBJ/PLY


class DynamicLayerPatch(BaseModel):
    """Property-only update (no geometry rebuild, 60Hz safe)."""
    visible: Optional[bool] = None
    opacity: Optional[float] = Field(None, ge=0.0, le=1.0)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)


class DynamicLayerInfo(BaseModel):
    """Response model for dynamic layer queries."""
    layer_id: str
    name: str
    geometry_type: str
    visible: bool
    opacity: float
    color: Optional[List[float]]
    point_count: int
    tri_count: int


# --- Animation ---

class ContinuousTransform(BaseModel):
    """Continuous property change at a fixed rate.

    For scalar properties (azimuth, opacity, etc.), `rate` is units/sec.
    For `property="color"`, use `target` instead of `rate` — the color
    interpolates linearly over `duration_ms`.
    """
    id: str
    property: str  # "azimuth", "rotation_z", "opacity", "color", etc.
    rate: Optional[float] = None          # units per second (for scalar props)
    target: Optional[List[float]] = None  # target value (for color: [r,g,b])
    duration_ms: int = Field(0, ge=0, description="0 = forever")
    # For dynamic layer transforms:
    layer_id: Optional[str] = None        # required for dynamic.transform_continuous


class DynamicTransformKeyframe(BaseModel):
    """A single keyframe for object animation."""
    t: float = Field(..., ge=0.0, le=1.0, description="Normalized time 0-1")
    position: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: Optional[float] = None
    scale: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: Optional[float] = None
    point_size: Optional[float] = None


class DynamicAnimation(BaseModel):
    """Keyframed animation for a dynamic layer."""
    layer_id: str
    keyframes: List[DynamicTransformKeyframe]
    duration_ms: int = Field(3000, ge=0)
    easing: EasingSpec = "ease_in_out"  # str | dict; can be per-property dict
    loop: bool = False
    ping_pong: bool = False
    repeat_count: int = Field(0, ge=0)


class InstantTransform(BaseModel):
    """One-shot transform applied immediately (no animation)."""
    layer_id: str
    position: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: Optional[float] = None
    scale: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: Optional[float] = None


# --- Render Mode ---

class RenderModeUpdate(BaseModel):
    """Switch between realtime and capture mode."""
    mode: Literal["realtime", "capture"]
    width: Optional[int] = Field(None, ge=1)
    height: Optional[int] = Field(None, ge=1)


class RenderModeState(BaseModel):
    """Current render mode and settings."""
    mode: str  # "realtime" or "capture"
    width: Optional[int] = None
    height: Optional[int] = None
    target_fps: int = 60


# --- Annotations (Editor overlays) ---

class BBoxCreate(BaseModel):
    label: str = "custom"
    center: List[float] = Field(..., min_length=3, max_length=3)
    size: List[float] = Field(..., min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: float = 0.0
    fill_opacity: float = 0.0


# --- Viewport ---

class ViewportSettings(BaseModel):
    point_size: Optional[float] = Field(None, ge=1, le=20)
    show_axes: Optional[bool] = None
    show_grid: Optional[bool] = None
    use_layer_colors: Optional[bool] = None
    fps_movement: Optional[bool] = None
    point_attenuation: Optional[bool] = None
    bg_color: Optional[List[float]] = None


class CorrectionState(BaseModel):
    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_z: float = 0.0


class ClipState(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


# --- System ---

class SceneLoadRequest(BaseModel):
    paths: List[str]


class SystemStatus(BaseModel):
    mode: str  # "viewer" or "editor"
    layers_count: int
    dynamic_layers_count: int
    total_points: int
    fps: float
    api_version: str = "1.0.0"
    server_port: int
```

---

## 13. REST Handler Example

```python
# handlers/camera.py

from aiohttp import web
from ..schemas import CameraState, CameraUpdate, ScalarValue, Vec3Value
from ..dispatcher import CommandDispatcher


def setup_routes(app: web.Application, dispatcher: CommandDispatcher):
    # Full state
    app.router.add_get("/api/v1/camera", get_camera)
    app.router.add_put("/api/v1/camera", set_camera)
    # Individual params
    app.router.add_put("/api/v1/camera/azimuth", set_azimuth)
    app.router.add_put("/api/v1/camera/elevation", set_elevation)
    app.router.add_put("/api/v1/camera/distance", set_distance)
    app.router.add_put("/api/v1/camera/fov", set_fov)
    app.router.add_put("/api/v1/camera/target", set_target)
    # Actions
    app.router.add_post("/api/v1/camera/fit", fit_camera)
    app.router.add_post("/api/v1/camera/preset", camera_preset)
    app.router.add_post("/api/v1/camera/look_at", look_at)


async def get_camera(request: web.Request) -> web.Response:
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.get_camera_state()
    return web.json_response(state.model_dump())


async def set_camera(request: web.Request) -> web.Response:
    """Set multiple camera params at once (partial update)."""
    data = await request.json()
    update = CameraUpdate(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(update)
    return web.json_response(state.model_dump())


async def set_azimuth(request: web.Request) -> web.Response:
    """Set azimuth only — convenience for single-param scripts."""
    data = await request.json()
    v = ScalarValue(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(CameraUpdate(azimuth=v.value))
    return web.json_response(state.model_dump())


async def set_elevation(request: web.Request) -> web.Response:
    data = await request.json()
    v = ScalarValue(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(CameraUpdate(elevation=v.value))
    return web.json_response(state.model_dump())


async def set_distance(request: web.Request) -> web.Response:
    data = await request.json()
    v = ScalarValue(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(CameraUpdate(distance=v.value))
    return web.json_response(state.model_dump())


async def set_fov(request: web.Request) -> web.Response:
    data = await request.json()
    v = ScalarValue(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(CameraUpdate(fov=v.value))
    return web.json_response(state.model_dump())


async def set_target(request: web.Request) -> web.Response:
    data = await request.json()
    v = Vec3Value(**data)
    dispatcher = request.app["dispatcher"]
    state = await dispatcher.set_camera(CameraUpdate(target=v.value))
    return web.json_response(state.model_dump())


async def fit_camera(request: web.Request) -> web.Response:
    dispatcher = request.app["dispatcher"]
    await dispatcher.fit_camera()
    return web.json_response({"status": "ok"})
```

---

## 14. WebSocket Handler Example

```python
# server.py (WS section)

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    dispatcher = request.app["dispatcher"]
    dispatcher.register_ws(ws)
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type", "")
                msg_id = data.get("id")
                
                try:
                    result = await dispatcher.handle_ws_command(msg_type, data)
                    await ws.send_json({
                        "type": "result",
                        "id": msg_id,
                        "status": "ok",
                        "data": result
                    })
                except ValueError as e:
                    await ws.send_json({
                        "type": "error",
                        "id": msg_id,
                        "code": "INVALID_PARAM",
                        "message": str(e)
                    })
                    
            elif msg.type == web.WSMsgType.BINARY:
                await dispatcher.handle_ws_binary(msg.data)
                
    finally:
        dispatcher.unregister_ws(ws)
    
    return ws
```

---

## 15. CommandDispatcher

The dispatcher is the single point of contact between the network layer and the Qt viewer:

```python
# dispatcher.py

class CommandDispatcher:
    """Validates, routes, and executes commands on the Qt main thread.
    
    The dispatcher holds a reference to the active window (ViewerWindow 
    or EditorWindow), the QtBridge, and optionally the AnimationEngine.
    All mutating operations are dispatched via bridge.invoke_on_qt() 
    to ensure thread safety.
    """

    def __init__(self, window, bridge: QtBridge):
        self._window = window
        self._bridge = bridge
        self._ws_clients: set = set()
        self._animation_engine = None  # set by server startup

    def set_animation_engine(self, engine):
        """Called during startup if animation package is available."""
        self._animation_engine = engine

    # --- Camera ---

    async def get_camera_state(self) -> CameraState:
        def _get():
            vp = self._viewport
            return CameraState(
                azimuth=vp.cam_azimuth,
                elevation=vp.cam_elevation,
                distance=vp.cam_distance,
                target=vp.cam_target.tolist(),
                fov=vp.cam_fov,
            )
        return await self._bridge.invoke_on_qt(_get)

    async def set_camera(self, update: CameraUpdate) -> CameraState:
        def _set():
            vp = self._viewport
            if update.azimuth is not None:
                vp.cam_azimuth = update.azimuth
            if update.elevation is not None:
                vp.cam_elevation = update.elevation
            if update.distance is not None:
                vp.cam_distance = update.distance
            if update.target is not None:
                vp.cam_target = np.array(update.target)
            if update.fov is not None:
                vp.cam_fov = update.fov
            vp.update()
            return self._read_camera_state(vp)
        return await self._bridge.invoke_on_qt(_set)

    # --- Layers ---

    async def get_layers(self) -> list[LayerInfo]:
        def _get():
            return [
                LayerInfo(
                    id=l.id, name=l.name, type=l.layer_type,
                    visible=l.visible, opacity=l.opacity,
                    point_count=l.point_count, tri_count=l.tri_count,
                )
                for l in self._layer_manager.layers
            ]
        return await self._bridge.invoke_on_qt(_get)

    # --- Animation Routing ---

    async def handle_ws_command(self, msg_type: str, data: dict) -> dict:
        """Route WS commands to the appropriate handler."""
        # Animation commands → AnimationEngine (runs on Qt thread)
        if msg_type in ("camera.animate", "camera.transform_continuous",
                        "dynamic.animate", "dynamic.transform_continuous",
                        "dynamic.transform", "animation.stop",
                        "transform.stop", "transform.stop_all"):
            if self._animation_engine is None:
                raise ValueError("Animation engine not available")
            return await self._bridge.invoke_on_qt(
                lambda: self._animation_engine.handle_command(msg_type, data)
            )

        # Render mode commands
        if msg_type.startswith("render."):
            return await self._bridge.invoke_on_qt(
                lambda: self._handle_render_command(msg_type, data)
            )

        # Camera, layer, dynamic, viewport, scene commands...
        handler = self._command_handlers.get(msg_type)
        if handler is None:
            raise ValueError(f"Unknown command type: {msg_type}")
        return await handler(data)

    # --- Event Broadcasting ---

    async def broadcast_event(self, event_type: str, data: dict):
        """Push an event to all connected WebSocket clients."""
        msg = {"type": event_type, **data}
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    @property
    def _viewport(self):
        """Get the active viewport (works for both Viewer and Editor)."""
        if hasattr(self._window, 'viewport'):
            return self._window.viewport      # ViewerWindow
        return self._window.gl_viewport        # EditorWindow

    @property
    def _layer_manager(self):
        return self._window.layer_manager
```

---

## 16. Security Considerations

| Concern | Mitigation |
|---------|------------|
| **Localhost only** | Default bind `127.0.0.1` — external access requires explicit `--api-host 0.0.0.0` |
| **No auth by default** | Acceptable for localhost dev use. Optional API key via `--api-key` flag for network access |
| **File path access** | `scene.load` validates paths exist and are readable before loading |
| **Resource limits** | WS message size capped at 100MB (for point cloud binary). REST body limit 50MB |
| **CORS** | Disabled by default. Enable via `--api-cors` for browser-based clients |
| **Rate limiting** | Not implemented in v1 — local-only access makes DoS unlikely |

---

## 17. Official Demo Scripts (`scripts/`)

To validate the API and provide reference implementations for users, a suite of official demo scripts will be implemented in `c:/Projects/EPAM/locul3d/scripts/`.

### 17.1 `demo_01_warehouse_walkthrough.py`
**Goal:** Demonstrate file loading, basic scene correction, and scripted camera movement.
**Workflow:**
1. Loads `C:/scans/warehouse.e57` (or a bundled test scan).
2. Applies `[rotate_x=-90, shift_z=-1.2]` correction to level it.
3. Uses `camera.animate` to execute a 15-second guided fly-through using a smooth `ease_in_out` curve across 4 keyframes.
4. Adds static bounding boxes to highlight specific warehouse zones.

### 17.2 `demo_02_dynamic_geometry.py`
**Goal:** Showcase 120Hz real-time object animation overlaying a static scene.
**Workflow:**
1. Loads a base scene.
2. Creates an STL-like triangle mesh cube (`dynamic.create`).
3. Starts a 10°/sec continuous camera orbit (`camera.transform_continuous`).
4. Sends a 60Hz stream of `dynamic.update` commands from a Python `time.sleep()` loop to stretch the cube dynamically on the Z-axis, simulating live data feedback.

### 17.3 `demo_03_color_pulse_and_easing.py`
**Goal:** Demonstrate the advanced timing and color animation capabilities.
**Workflow:**
1. Creates several dynamic BBox and Surface layers.
2. Applies different easing curves (e.g., `"spring"`, `"ease_out_bounce"`) to their positions so they pop into existence.
3. Uses `dynamic.animate` to smoothly pulse their `opacity` and `color` properties.

### 17.4 `demo_04_high_res_capture.py`
**Goal:** Show how to generate a deterministic, high-quality MP4 sequence.
**Workflow:**
1. Loads a heavy point cloud.
2. Sets render mode to `capture` with `1920x1080` resolution.
3. Sends 300 sequential `render.capture_frame` step commands mapped to a slow camera orbit.
4. Automatically saves frames sequentially to `C:/frames/`.
5. *(Commented out step)*: Provides the `ffmpeg` command to stitch them into a 60fps video.

### 17.5 `demo_flyover_search_area.py`
**Goal:** End-to-end exercise of the cone-shadow shader fade,
adaptive realtime FPS controller, virtual animation clock, and the
recording API.

**Workflow:**
1. Reads the editor's `search_region` bbox annotation via
   `GET /api/v1/shapes/bboxes`.
2. Frames the camera on it, optionally applies an AABB scene clip
   above/below the bbox to remove ceiling and floor.
3. Optionally enables the cone-shadow shader fade with the bbox as
   the AoI (`PUT /api/v1/viewport/fade`).
4. Configures the realtime FPS controller (`animation.set_realtime_fps`)
   or preview mode (`animation.set_preview_mode`).
5. Sets the virtual-clock auto slowdown (`animation.set_time_scale`).
6. Optionally starts a video recording (`POST /api/v1/recording/start`)
   with HW or SW encoder selection, codec, resolution, fps, and
   per-recording grid/axes/bg overrides.
7. Runs a continuous azimuth rotation around the bbox via
   `camera.transform_continuous` (with `track_id` and a finite
   `duration_ms` when recording).
8. When recording, polls `GET /api/v1/recording/status` once per
   second and reports frames/bytes; exits on natural completion.

**Key flags:**

| Flag | Description |
|------|-------------|
| `--fade` | Enable cone-shadow shader fade through the search bbox |
| `--clip {none,z,box}` | Scene clip — `z` removes ceiling/floor outside the bbox |
| `--max-fps`, `--min-fps`, `--no-adaptive` | Realtime FPS controller |
| `--preview`, `--preview-fps` | Preview mode (LOD adapts, FPS held) |
| `--no-time-scale`, `--nominal-fps` | Virtual-clock slowdown control |
| `--record PATH` | Record to mp4. Relative paths placed under `<repo>/video/` |
| `--rec-resolution`, `--rec-fps`, `--rec-codec`, `--rec-hw`, `--rec-bitrate` | Recording params |
| `--rec-grid`, `--rec-axes`, `--rec-bg` | Per-recording viewport overrides |
| `--stop` | Stop a running flyover and exit |

This is the demo to look at first when wiring a new client — every
shipped feature has at least one CLI flag that exercises its REST or
WS endpoint.

---

## 18. Implementation Plan

### Phase 1: Core Infrastructure (Foundation)
1. Add `aiohttp` + `pydantic` to `pyproject.toml` optional dependencies
2. Create `remote/` package with server, bridge, and dispatcher
3. Implement `QtBridge` with `invoke_on_qt`
4. Basic server lifecycle (start/stop/CLI args: `--api-port`, `--no-api`, `--api-host`)
5. `GET /system/ping` and `GET /system/status`

### Phase 2: REST API
6. Camera handlers (GET/PUT full-state + individual param setters + preset/fit/look_at)
7. Scene handlers (layers list, load, load_folder, clear, update)
8. Viewport handlers (settings, correction, clip)
9. Dynamic geometry layer CRUD (pointcloud, mesh, bboxes, surfaces, file) with `layer_id = f"dyn_{name}"` generation
10. Annotation shape handlers (editor bbox/plane CRUD)
11. Screenshot endpoint

### Phase 3: WebSocket
12. WS connection handler with client registry
13. Command dispatch — camera (full + granular), dynamic layers, annotations
14. Event broadcasting (camera, layer, dynamic, scene, animation, transform, render events)
15. Binary point streaming mode for dynamic layers

### Phase 4: Animation Engine (`animation/` package)
16. Create `animation/` package: engine, tracks, easing, interpolation modules
17. `AnimationEngine` with `QTimer(PreciseTimer)` at 125Hz
18. Camera keyframe tracks (with loop/ping_pong/repeat_count)
19. Easing system: named presets, cubic-bezier, spring physics, steps, per-property
20. Continuous transform tracks (camera + dynamic layer properties, including color target mode)
21. Dynamic layer keyframed animation tracks (position, rotation, scale, color, opacity)
22. Animation ↔ Dispatcher integration (`handle_command` routing)
23. Animation start/stop/done/expired events

### Phase 5: Render Modes
24. Capture mode: disable LOD, set resolution, stepped animation
25. `render.capture_frame` WS command with disk save support
26. `render.set_mode` / `render.get_mode` REST + WS
27. `render.set_target_fps` for capture step size

### Phase 6: OpenAPI & Polish
28. OpenAPI 3.1 spec generation from Pydantic schemas
29. Swagger UI static hosting (optional)
30. Integration tests with `aiohttp.test_utils`
31. CLI documentation update
32. README update with API examples

### Phase 7: Demo Scripts (`scripts/`)
33. `demo_01_warehouse_walkthrough.py`
34. `demo_02_dynamic_geometry.py`
35. `demo_03_color_pulse_and_easing.py`
36. `demo_04_high_res_capture.py`

### Dependencies to Add

```toml
# pyproject.toml
[project.optional-dependencies]
remote = ["aiohttp>=3.9.0", "pydantic>=2.0.0"]
```

---

## 19. Testing Strategy

### Unit Tests
- **Schema validation**: Pydantic model edge cases (missing fields, invalid ranges, EasingSpec union)
- **Bridge**: Mock Qt thread, verify invoke_on_qt behavior
- **Dispatcher**: Mock window, verify correct property reads/writes
- **Easing functions**: Pure `t→t` math — verify cubic-bezier, spring, steps produce correct curves at known t values
- **Interpolation**: Verify lerp, shortest-arc angle interpolation, color space conversion
- **Track logic**: Mock viewport, verify `CameraTrack.tick()` produces correct intermediate values
- **Continuous transforms**: Verify expiry at `duration_ms`, verify `rate * dt` accumulation
- **Loop/ping_pong**: Verify correct cycle behavior, repeat_count termination

### Integration Tests
- **aiohttp test client**: Full request/response cycle without a real viewer
- **WS protocol**: Connect, send commands, verify responses and events
- **Animation commands**: Send `camera.animate`, verify `event.animation_started` + `event.animation_done`
- **Continuous transforms**: Start/stop via WS, verify `event.transform_started` / `event.transform_stopped`
- **Render mode**: Switch to capture, send `capture_frame`, verify frame data returned
- **Layer ID generation**: Create dynamic layer, verify `layer_id = f"dyn_{name}"`; duplicate returns 409

### End-to-End Tests
- Launch viewer with `--api-port` in a subprocess
- Run a script that exercises all API endpoints
- Verify scene state via GET requests after mutations
- Screenshot comparison for visual regression
- Run each demo script (`scripts/demo_*.py`) against a test scene

### Example Integration Test

```python
from aiohttp.test_utils import AioHTTPTestCase
from locul3d.remote.server import create_app

class TestCameraAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(mock_window(), mock_bridge())

    async def test_get_camera(self):
        resp = await self.client.get("/api/v1/camera")
        assert resp.status == 200
        data = await resp.json()
        assert "azimuth" in data
        assert "elevation" in data

    async def test_set_camera(self):
        resp = await self.client.put("/api/v1/camera", json={
            "azimuth": 90, "elevation": 0
        })
        assert resp.status == 200
        data = await resp.json()
        assert data["azimuth"] == 90

    async def test_dynamic_layer_duplicate_name(self):
        resp = await self.client.post("/api/v1/scene/dynamic", json={
            "name": "test", "geometry_type": "pointcloud", "points": [[0,0,0]]
        })
        assert resp.status == 200
        assert (await resp.json())["layer_id"] == "dyn_test"
        # Duplicate
        resp2 = await self.client.post("/api/v1/scene/dynamic", json={
            "name": "test", "geometry_type": "pointcloud", "points": [[1,1,1]]
        })
        assert resp2.status == 409
```

### Example Easing Unit Test

```python
from locul3d.animation.easing import resolve_easing, cubic_bezier

def test_linear_easing():
    fn = resolve_easing("linear")
    assert fn(0.0) == 0.0
    assert fn(0.5) == 0.5
    assert fn(1.0) == 1.0

def test_cubic_bezier_ease_in_out():
    fn = resolve_easing("ease_in_out")
    assert fn(0.0) == 0.0
    assert fn(1.0) == 1.0
    assert fn(0.5) > 0.4  # middle of S-curve
    assert fn(0.5) < 0.6

def test_spring_easing_overshoots():
    fn = resolve_easing({"spring": {"damping": 0.3, "stiffness": 200}})
    # Low damping should overshoot past 1.0 before settling
    values = [fn(t/100) for t in range(101)]
    assert max(values) > 1.0
    assert abs(values[-1] - 1.0) < 0.01  # settles near 1.0
```

---

## Appendix A: Port Selection

Default port **8350** chosen because:
- Not commonly used by other dev tools
- Easy to remember (8 + "3D" + "50" → 8350)
- Above 1024 (no root/admin required)
- Avoids conflicts with: Jupyter (8888), Django (8000), Flask (5000), React (3000)

## Appendix B: Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **REST + WS (this design)** | Universal clients, OpenAPI docs, real-time streaming | Two protocols to maintain |
| **gRPC** | Efficient binary, strong typing | Requires protobuf toolchain, poor browser support |
| **ZMQ** | Ultra-low latency | No built-in HTTP, custom protocol needed |
| **OSC** | Standard in creative tools | Limited data types, no request/response pattern |
| **Named Pipes / Shared Memory** | Fastest IPC | Platform-specific, complex, no network access |

The REST + WS approach offers the best balance of universality, discoverability (OpenAPI), real-time performance, and ease of client implementation.
