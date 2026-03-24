# Locul3D Architecture

Technical documentation for the Locul3D 3D scanner data editor and viewer.

## Documents

| Document | Contents |
|---|---|
| [Rendering Pipeline](rendering-pipeline.md) | OpenGL rendering, VBO management, stride-based LOD, GPU-side opacity, camera model, memory budget |
| [Panorama Rendering](panorama-rendering.md) | 360° panorama extraction, cubemap assembly, scene-space & immersive modes, scanner support |
| [E57 Import Pipeline](e57-import-pipeline.md) | E57 ingestion, filtering, alignment, decimation, layer building, panorama extraction |
| [Scene Correction](scene-correction.md) | Multi-step floor & wall detection, voxelized normal estimation, surface classification, rotation optimization |
| [Data Model](data-model.md) | `LayerData`, `LayerManager`, dtype safety, layer types, and data flow |
| [Loader Performance](loader-performance.md) | Baseline vs optimized benchmarks, E57/PLY fast parsers, chunked CRC strip, F-order extraction, memory management, GPU upload, glossary of technical terms |

## Package Structure

```
locul3d/
├── analysis/           Scene analysis algorithms
│   ├── ceiling.py       Ceiling height detection (Z-histogram)
│   └── scene_correction.py  Auto-detect floor/wall correction
├── core/               Core data structures
│   ├── constants.py      Colors, sizes, layer groups
│   ├── correction.py     Scene correction state
│   ├── geometry.py       Geometric utilities
│   ├── layer.py          LayerData + LayerManager
│   └── scene.py          Scene-level state
├── editor/             Editor application
│   ├── main.py           Entry point
│   ├── viewport.py       Editor GL viewport (subclass)
│   └── window.py         Main window + E57 orchestration
├── viewer/             Minimal viewer application
│   ├── main.py           Entry point
│   └── window.py         Viewer window
├── rendering/          Rendering subsystem
│   ├── camera.py         Camera utilities
│   ├── gizmos.py         3D gizmo rendering
│   ├── gl/viewport.py    BaseGLViewport (shared renderer)
│   └── panorama/         360° panorama subsystem
│       ├── __init__.py     PanoramaManager + equirect assembly
│       ├── camera.py       Pinhole camera model + projection
│       ├── extractor.py    E57 image extraction + face sorting
│       ├── geometry.py     UV sphere mesh generation
│       ├── immersive.py    GL texture + sphere rendering
│       └── station_marker.py  Diamond marker rendering
├── ui/                 UI components
│   ├── themes.py         Theme management (dark/light)
│   ├── dialogs/          Modal dialogs
│   │   ├── correction_dialog.py  Scene correction dialog
│   │   └── scene_dialog.py       Scene settings dialog
│   ├── widgets/          Reusable widgets
│   │   ├── color_swatch.py  Color picker swatch
│   │   ├── info.py          Info panel widget
│   │   └── layers.py        Layer list widget
│   └── panels/           Sidebar panels
│       ├── bbox.py          Bounding box panel
│       ├── plane.py         Plane fitting panel
│       └── reference.py     Reference plane panel
├── plugins/            Import/export plugins
│   ├── base.py           Plugin base class
│   ├── importers/        File format importers
│   │   ├── e57.py          E57 import pipeline
│   │   ├── loaders.py      Unified loader dispatch
│   │   ├── obj.py          OBJ import
│   │   └── ply.py          PLY fast binary parser
│   └── tools/            Interactive tools
│       ├── move.py         Move tool
│       ├── rotate.py       Rotate tool
│       └── select.py       Selection tool
└── utils/              Shared utilities
    ├── io.py             File I/O helpers (fast PLY reader)
    ├── math.py           Math utilities
    └── signals.py        Qt signal helpers
```
