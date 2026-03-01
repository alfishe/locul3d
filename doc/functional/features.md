# Locul3D — Functional Overview

> Desktop 3D scanner data editor and viewer for LiDAR point clouds, meshes, and E57 scan files.

## Product Summary

Locul3D is a native desktop application for inspecting, annotating, and measuring 3D scan data. It handles datasets from a few thousand points to 100M+ points captured by terrestrial laser scanners.

**Two applications in one package:**

| Application | Purpose | Launch |
|---|---|---|
| **Editor** | Full annotation workflow — bounding boxes, planes, measurements | `locul3d editor` |
| **Viewer** | Lightweight read-only inspection | `locul3d viewer` |

Both share the same rendering engine and can open the same file formats.

---

## Features

### Data Import

| Format | Description |
|---|---|
| **E57** | Full processing pipeline: ingest → filter → align → decimate → build layers |
| **PLY** | Point clouds and meshes (binary + ASCII) |
| **OBJ** | Triangle meshes with optional materials |
| **Folder** | Batch-load all supported files in a directory as separate layers |

#### E57 Processing Pipeline

When importing an E57 file, Locul3D runs a multi-stage processing pipeline on a background thread:

1. **Ingest** — Read raw scanner data (points + colors + intensity)
2. **Filter** — Voxel downsample + statistical outlier removal
3. **Align** — Detect ground plane via RANSAC, rotate to Z-up, zero floor
4. **Decimate** — Final voxel downsample for interactive display
5. **Build Layers** — Create separate layers for raw scan, aligned cloud, decimated preview

A progress dialog shows each stage with real-time status updates.

---

### 3D Viewport

Interactive 3D rendering with orbit camera control.

#### Navigation

| Input | Action |
|---|---|
| **Left drag** | Orbit camera |
| **Shift+Left drag** | Pan camera |
| **Middle drag** | Pan camera |
| **Right drag** | Dolly (move through scene) |
| **Scroll wheel** | Zoom in/out |
| **F** | Fit all layers in view |
| **Home** | Reset view (fit to selected box or scene) |

#### Camera Presets

Switch between orthographic-style views:
- **Perspective** — Default 3D perspective
- **Top** — Bird's eye (XY plane)
- **Front** — Front elevation (XZ plane)
- **Right** — Side elevation (YZ plane)
- **Isometric** — 45° isometric view

#### Display Controls

- **Point Size** — Adjustable slider (1–20 pixels)
- **Layer Colors** — Toggle between per-layer uniform colors and per-vertex RGB
- **Axes** — Show/hide XYZ coordinate axes
- **Grid** — Show/hide ground plane grid (auto-scales to scene)
- **Dark / Light Theme** — Full UI theme switching

#### Screenshot

Save the current viewport as an image file (PNG).

---

### Layer System

Every loaded file or processing result becomes a **layer** that can be independently controlled.

#### Per-Layer Controls

| Control | Description |
|---|---|
| **Visibility** | Show/hide toggle (frees GPU memory when hidden) |
| **Opacity** | 0–100% slider (GPU-side, instant even for 100M+ points) |
| **Color swatch** | Layer color indicator |
| **Point/triangle count** | Displayed next to each layer name |

#### Layer Groups

Layers are automatically categorized:
- **Scan** — aligned, decimated, raw_scan, unclassified
- **Panoramas** — embedded panorama images from E57 scans
- **Surface** — detected planar surfaces
- **Feature** — extracted feature points
- **Other** — uncategorized layers

#### Bulk Actions

- **Show All** — Make all layers visible
- **Hide All** — Hide all layers

---

### Panorama Support

E57 files may embed panoramic images alongside point cloud data. Locul3D extracts and displays these as separate panorama layers.

#### Supported Types

| Type | Description |
|---|---|
| **Spherical** | Equirectangular (360° × 180°) images |
| **Cubemap** | Six-face cube maps (automatically assembled to equirectangular) |
| **Pinhole / Cylindrical** | Standard camera projections embedded in E57 |

#### Scene Integration

- Each panorama appears as a **layer** in the layer panel with a configurable color swatch
- When the layer is **visible**, a diamond-shaped **station marker** appears in the 3D scene at the camera position
- Marker color follows the layer color; hiding the layer hides the marker
- Click the **360°** button to enter immersive panorama view

#### Immersive 360° View

| Input | Action |
|---|---|
| **Mouse drag** | Look around (yaw/pitch) |
| **Scroll wheel** | Adjust field of view (20°–120°) |
| **Esc** | Exit panorama and return to 3D scene |

#### Architecture

Panorama support is implemented as a self-contained subpackage (`rendering/panorama/`) with four independent modules. The entire feature can be disabled by removing the subpackage.

---

### Bounding Box Annotation (Editor)

Create and edit 3D oriented bounding boxes (OBBs) for object annotation.

#### Workflow

1. Click on a point in the cloud to place a new bounding box
2. Select a box to edit its properties
3. Use tools to move, rotate, or resize
4. Save annotations to YAML

#### Tools

| Tool | Shortcut | Description |
|---|---|---|
| **Select** | `1` | Click to select bounding boxes |
| **Move** | `2` | Drag to reposition boxes |
| **Rotate** | `3` | Drag to rotate boxes around an axis |

#### Axis Constraint

Constrain movement/rotation to a single axis:

| Key | Axis |
|---|---|
| **X** | X-axis only (red) |
| **Y** | Y-axis only (green) |
| **Z** | Z-axis only (blue) |

#### BBox Panel

- Edit position (X, Y, Z) numerically
- Edit dimensions (W, D, H) numerically
- Set label / category
- Color-coded per category
- Undo support (`Ctrl+Z`)

---

### Surface Plane Detection (Editor)

Detect and annotate planar surfaces in the point cloud.

- Create plane from selected region
- View plane normal and area
- Edit plane properties in sidebar panel

---

### Reference Point System (Editor)

Set a custom coordinate origin for relative measurements.

#### Features

- **Set Reference** — Click on any point to set as coordinate origin
- **Clear Reference** — Reset to scene origin
- **Coordinate Modes:**
  - **Scene (absolute)** — Standard world coordinates
  - **Relative (from ref)** — Distances measured from reference point

Coordinates display updates live as you move the cursor.

---

### Annotation Save/Load (Editor)

Annotations persist as YAML files alongside the scan data.

| Action | Shortcut |
|---|---|
| **Save** | `Ctrl+S` |
| **Save As** | `Ctrl+Shift+S` |
| **Load** | `Ctrl+L` |

YAML files store:
- Bounding box positions, dimensions, rotations, labels
- Surface plane definitions
- Reference point coordinates

---

### Hot Reload

Files are monitored for changes on disk (every 2 seconds). When a source file is modified externally, the layer automatically reloads without restarting the application.

---

### Performance Optimizations

Locul3D is designed for datasets of 100M+ points on standard hardware.

| Optimization | Effect |
|---|---|
| **VBO caching** | Data uploaded to GPU once, drawn every frame at zero cost |
| **Stride-based LOD** | During orbit/pan, shows 15M of 100M points for smooth interaction |
| **Opacity stride preview** | During slider drag, shows 2M points for instant opacity feedback |
| **GPU-side opacity** | Opacity changes via `GL_CONSTANT_ALPHA` — no data regeneration |
| **Memory eviction** | Hidden layers free GPU memory and CPU caches |
| **dtype enforcement** | Automatic float64→float32 conversion prevents GPU crashes |

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open file |
| `Ctrl+Shift+O` | Open folder |
| `Ctrl+S` | Save annotations |
| `Ctrl+Shift+S` | Save annotations as... |
| `Ctrl+L` | Load annotations |
| `Ctrl+Z` | Undo |
| `F` | Fit scene in view |
| `Home` | Reset view |
| `1` | Select tool |
| `2` | Move tool |
| `3` | Rotate tool |
| `X` / `Y` / `Z` | Constrain to axis |
| `Esc` | Exit panorama view |
