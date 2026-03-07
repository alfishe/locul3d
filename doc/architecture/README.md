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

## Package Structure

```
locul3d/
├── core/               Core data structures
│   ├── constants.py      Colors, sizes, layer groups
│   └── layer.py          LayerData + LayerManager
├── editor/             Editor application
│   ├── main.py           Entry point
│   ├── viewport.py       Editor GL viewport (subclass)
│   └── window.py         Main window + E57 orchestration
├── viewer/             Minimal viewer application
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
│   ├── widgets/          Reusable widgets (layers, color swatch)
│   └── panels/           Sidebar panels (info, bbox, surfaces)
├── plugins/            Import/export plugins
│   └── importers/e57.py  E57 import pipeline
└── utils/              Shared utilities
```
