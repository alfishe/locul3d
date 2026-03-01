# Data Model

> Layer types, data accessors, and memory management in Locul3D.

## Layer Data Model

```mermaid
classDiagram
    class LayerManager {
        +layers: LayerData[]
        +base_dir: str
        +load_single_file(path)
        +get_scene_bounds() → center, radius
        +visible_layers() → LayerData[]
        +total_stats() → layers, points, tris
    }

    class LayerData {
        +id: str
        +name: str
        +layer_type: str
        +visible: bool
        +opacity: float
        +color: float[3]
        +points: ndarray Nx3
        +colors: ndarray Nx3
        +normals: ndarray Nx3
        +triangles: ndarray Mx3
        +line_points: ndarray Lx3
        +point_count: int
        +tri_count: int
        +loaded: bool
        +get_pts_array() → float32 Nx3
        +get_colors_array() → float32 Nx3
        +get_normals_array() → float32 Nx3
        +get_tris_array() → uint32 Mx3
        +get_lines_array() → float32 Lx3
        +get_bounds() → center, radius
        +evict_byte_caches()
        +release_source_data()
    }

    LayerManager "1" --> "*" LayerData
```

## Layer Types

| Type | Geometry | Rendering Method | Example |
|---|---|---|---|
| `pointcloud` | `points` Nx3 | `glDrawArrays(GL_POINTS)` | Scanned points |
| `mesh` | `points` + `triangles` | `glDrawElements(GL_TRIANGLES)` | Reconstructed surfaces |
| `wireframe` | `line_points` Lx3 | `glDrawArrays(GL_LINES)` | Oriented bounding boxes |
| `panorama` | Position + image data | Special marker rendering | 360° scan images |

## Data Safety: dtype Enforcement

All `get_*_array()` methods enforce OpenGL-compatible formats:

```mermaid
flowchart LR
    RAW["Raw data<br/>(any dtype)"] --> CHECK{"dtype ==<br/>float32?"}
    CHECK -->|Yes| CONTIG{"C-contiguous?"}
    CHECK -->|No| CONVERT["np.ascontiguousarray<br/>(dtype=float32)"]
    CONTIG -->|Yes| RETURN["Return<br/>(zero-copy)"]
    CONTIG -->|No| CONVERT
    CONVERT --> CACHE["Cache on instance"]
    CACHE --> RETURN
```

This is critical because:
- **E57 files** load as `float64` by default
- **OpenGL** expects `GL_FLOAT` (32-bit) in `glVertexPointer`
- Sending `float64` to a 32-bit pointer causes the GPU to read **2× past the buffer** → SIGSEGV

## Data Flow: Load → Render

```mermaid
sequenceDiagram
    participant FS as Filesystem
    participant LD as LayerData
    participant VP as BaseGLViewport
    participant GPU as GPU (VBO)

    FS->>LD: load() reads PLY/OBJ
    Note over LD: points = Nx3 (may be float64)
    
    VP->>LD: get_pts_array()
    Note over LD: Enforce float32 + C-contiguous
    LD-->>VP: float32 Nx3

    VP->>GPU: _get_or_create_vbo('pts', data)
    Note over GPU: glBufferData(GL_STATIC_DRAW)
    
    loop Every frame
        VP->>GPU: glBindBuffer + glDrawArrays
        Note over GPU: Draw from cached VBO
    end
```

## Memory Management

### Cache Hierarchy

```
LayerData
  ├── Source arrays (points, colors, normals, triangles)
  │     Always in memory while layer is loaded
  │
  ├── Byte caches (_pts_bytes, _rgba_bytes, etc.)
  │     Evicted via evict_byte_caches() when layer hidden
  │
  └── GPU VBOs (managed by BaseGLViewport._vbos dict)
        Freed via delete_vbos_for_layer() when layer hidden
        Freed via delete_all_vbos() on file load
```

### Eviction Strategy

| Event | Action |
|---|---|
| Layer hidden | `evict_byte_caches()` + `delete_vbos_for_layer()` |
| New file loaded | `delete_all_vbos()` |
| Layer deleted | `release_source_data()` + `delete_vbos_for_layer()` |

## File Reference

| File | Contents |
|---|---|
| [`core/layer.py`](../../src/locul3d/core/layer.py) | `LayerData`, `LayerManager` |
| [`core/constants.py`](../../src/locul3d/core/constants.py) | `COLORS`, `LAYER_GROUPS`, `DEFAULT_SIZES` |
