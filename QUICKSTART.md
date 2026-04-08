# ⚡ Locul3D — Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch

```bash
python start.py                # 3D Viewer (default)
python start.py editor         # Annotation Editor
python start.py scan.ply       # Viewer with a file
python start.py editor -a annotations.yaml output/    # Editor with annotations
```

### 3. Navigate

- **Orbit**: Left drag
- **Pan**: Middle drag
- **Zoom**: Scroll
- **Fit view**: `F`

### Need E57 panoramas?

E57 support (pye57, Pillow) is included in requirements.txt — no extra install needed.

### 4. Panorama Mode

- Open an E57 file containing panoramas (Leica BLK, FARO, etc.)
- Click the **360°** button on any panorama layer
- Adjust opacity to see through the panorama to the point cloud
- Fine-tune alignment: Arrow keys (yaw/pitch), Q/E (roll), Shift for 1° steps
- **Esc** to exit panorama mode

### 5. Scene Clipping

- Click **Scene** in the toolbar to see scene bounds
- Click **Hide Ceiling** for automatic ceiling removal
- Adjust any axis min/max to clip the view — changes apply in real-time
- Click **Reset** to restore the full scene

### 6. Remote Control

Locul3D ships with a REST + WebSocket API on `http://localhost:8350` for
scripting and automation. It's enabled by default in **both** the viewer
and the editor.

```bash
# Bind on a different port:
python start.py --api-port 9000 scan.e57
# Disable the API entirely:
python start.py --no-api scan.e57

# Run a scripted camera flyover around the scene's `search_region` bbox
# (requires the editor — annotations live there):
python start.py editor scan.e57
python scripts/demo_flyover_search_area.py
```

Open `http://localhost:8350/api/v1/system/info` for a sanity check, or
`http://localhost:8350/openapi.json` for the full schema.

### 7. Video Recording

Record the GL viewport to an H.264 / HEVC mp4 with hardware-accelerated
encoding (videotoolbox on macOS, NVENC/QSV/AMF on Windows, NVENC/VAAPI on
Linux). Files default to `<repo>/video/`.

```bash
# Default flyover recording (HEVC HW, viewport resolution, 60 fps):
python scripts/demo_flyover_search_area.py --record fly.mp4

# 4K HEVC, force HW encoder, custom bitrate:
python scripts/demo_flyover_search_area.py --record fly_4k.mp4 \
       --rec-resolution 4k --rec-fps 60 --rec-codec hevc \
       --rec-hw hw --rec-bitrate 80000

# Force software fallback (libx265):
python scripts/demo_flyover_search_area.py --record fly_sw.mp4 --rec-hw sw

# Clean look — no grid/axes, white background:
python scripts/demo_flyover_search_area.py --record fly_clean.mp4 \
       --rec-grid off --rec-axes off --rec-bg 1,1,1
```

UI input is locked while recording so the captured animation stays
deterministic; the editor still refreshes the viewport so you can watch
the recording happen.

### Trouble?

```bash
pip install --upgrade PyOpenGL PySide6
# Recording requires ffmpeg on PATH:
brew install ffmpeg     # macOS
apt install ffmpeg      # Debian/Ubuntu
choco install ffmpeg    # Windows (Chocolatey)
```
