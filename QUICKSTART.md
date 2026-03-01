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

### Trouble?

```bash
pip install --upgrade PyOpenGL PySide6
```
