# 🌐 Locul3D — *The Place in 3D*

> A fast, modern 3D point cloud viewer and bounding box annotation editor.
>

<p align="center">
  <strong>View</strong> · <strong>Annotate</strong> · <strong>Explore</strong>
</p>

---

## ✨ Features

| | |
|---|---|
| 🔭 **Real-time 3D Viewer** | Point clouds, meshes, wireframes — rendered with OpenGL |
| 🌐 **360° Panorama Viewer** | Jump into E57 scan panoramas — Leica BLK, NavVis VLX, FARO supported |
| 📦 **3D Annotation Layouts** | Place, move, and resize reference boxes with center+size or min/max corners — toggle between modes with one click |
| 🗂️ **Multi-Layer Scene** | Load point clouds, meshes, and annotations from separate files (PLY, OBJ, E57) into a single scene — control visibility and opacity per layer |
| ✂️ **Scene Clipping** | Inspect scene bounds, hide ceiling with one click, clip to any axis-aligned region — all via GL clip planes (no data copies) |
| 🌗 **Auto Dark/Light Theme** | Follows your OS appearance automatically |
| ⌨️ **Blender-style Shortcuts** | Q/G/R/S for tools, X/Y/Z for axis constraints |
| ↩️ **Undo/Redo** | Full undo stack for annotation work |
| 💾 **JSON/YAML Export** | Save and reload annotations |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch viewer
python start.py

# Launch annotation editor
python start.py editor

# Open a file or folder directly
python start.py scan.ply
python start.py folder_with_files_as_layers/
```

---

## 🎮 Controls

### Viewer

| Action | Input |
|--------|-------|
| Orbit camera | Left drag |
| Pan camera | Middle drag / Shift+Left drag |
| Zoom | Scroll / Right drag |
| Fit to scene | `F` |
| Toggle grid | `G` |
| Toggle axes | `A` |
| Open file | `Ctrl+O` |
| Open folder | `Ctrl+Shift+O` |
| Enter 360° panorama | Click **360°** button on panorama layer |
| Exit panorama | `Esc` |
| Panorama opacity | Drag opacity slider while inside |

### Editor (all viewer controls plus)

| Action | Input |
|--------|-------|
| Place new bbox | `Ctrl+Click` |
| Select tool | `Q` |
| Move tool | `G` |
| Rotate tool | `R` |
| Scale tool | `S` |
| Axis constraint | `X` / `Y` / `Z` |
| Delete bbox | `Delete` |
| Undo | `Ctrl+Z` |
| Duplicate | `Ctrl+D` |
| Center/Corners toggle | **Center** button in BBox panel |
| Scene dialog | **Scene** toolbar button |
| Hide ceiling | **Scene** → **Hide Ceiling** |

---

## � Scene Correction

Locul3D supports rotation and shift corrections for aligning scan data to world axes. Corrections can be applied via YAML sidecar file, CLI flags, the in-app dialog, or keyboard fine-tuning.

### Correction YAML File

Place a YAML file next to your scan file. Auto-detected names (in order):

```
<filename>.correction.yaml
<filename>.correction.yml
correction.yaml
correction.yml
```

Format:

```yaml
# Scene correction (degrees / scene units)
correction:
  rotate_x: -90.0
  rotate_y: 0.0
  rotate_z: 3.5
  shift_x: 0.0
  shift_y: 0.0
  shift_z: -1.2
```

### CLI Options

```bash
python start.py scan.e57 --rotate-x -90 --shift-z -1.2
```

All six axes are supported: `--rotate-x`, `--rotate-y`, `--rotate-z`, `--shift-x`, `--shift-y`, `--shift-z`. CLI values override sidecar values.

### In-App

- **Scene Correction** toolbar button opens a dialog for live editing
- **Keyboard fine-tuning** (WASD+QE/and arrows for rotation and shift, Shift for larger steps)

---

## ✂️ Scene Clipping

The **Scene** toolbar button (available in both Viewer and Editor) opens a non-modal dialog for inspecting and clipping the scene.

### Scene Dialog

- **Scene Bounds** — Shows X, Y, Z min/max and span in metres, pre-populated from the cached axis-aligned bounding box (AABB). Values update the viewport clip planes in real-time as you type.
- **Hide Ceiling** — One-click ceiling removal. The ceiling height is auto-detected in the background after scene load using Z-histogram peak analysis. Clips the scene 0.3m below the detected ceiling.
- **Reset** — Removes all clipping and restores the full scene.

### Performance

- Scene AABB is cached in `LayerManager.scene_aabb` — computed once after geometry loads, excludes panorama layers.
- Ceiling height is pre-computed silently in the background via `QTimer.singleShot(0)` after every load path.
- Clipping uses OpenGL clip planes (`GL_CLIP_PLANE0..5`) — **no point data is copied or modified**.

---

## �📦 Installation

### Requirements

- Python 3.9+
- PySide6, PyOpenGL, NumPy, Open3D, SciPy, pye57, Pillow

```bash
pip install -r requirements.txt
```

### Package Install (editable)

```bash
pip install -e .
```

Then use anywhere:

```bash
python -m locul3d               # viewer (default)
python -m locul3d editor        # annotation editor
locul3d-viewer                  # viewer via entry point
locul3d-editor                  # editor via entry point
```

---

## 🏗️ Architecture

```
locul3d/
├── start.py          ← Launch here
├── src/locul3d/      ← Python package
│   ├── viewer/       ← 3D viewer application
│   ├── editor/       ← BBox annotation editor
│   ├── analysis/     ← Scene analysis (ceiling detection)
│   ├── rendering/
│   │   ├── gl/       ← OpenGL viewport
│   │   └── panorama/ ← 360° panorama (extractor, sphere, camera)
│   └── ...           ← Core, UI, plugins
└── doc/architecture/ ← Architecture documentation
```

---

## 📄 License

MIT

---

<p align="center">
  <em>Locul3D — from Romanian "locul" (the place) + 3D</em>
</p>
