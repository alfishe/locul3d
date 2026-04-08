"""Pydantic schemas for the Remote Control API.

Every request/response payload is validated through these models.
They also drive OpenAPI 3.1 spec generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Easing ────────────────────────────────────────────────────────────


class SpringEasing(BaseModel):
    """Spring physics easing — matches ``CASpringAnimation``."""

    damping: float = Field(0.7, ge=0, le=2)
    stiffness: float = Field(100.0, ge=0)
    mass: float = Field(1.0, ge=0.01)
    initial_velocity: float = 0.0


class StepsEasing(BaseModel):
    """Step function easing — matches CSS ``steps()``."""

    count: int = Field(..., ge=1)
    position: Literal["start", "end"] = "end"


# Easing can be a named preset string, or a dict
# (e.g. {"cubic_bezier": [x1,y1,x2,y2]}, {"spring": {...}}, {"steps": {...}})
EasingSpec = Union[str, Dict[str, Any]]


# ── Camera ────────────────────────────────────────────────────────────


class CameraState(BaseModel):
    """Full camera state snapshot."""

    azimuth: float = Field(45.0, description="Horizontal rotation (degrees)")
    elevation: float = Field(30.0, description="Vertical rotation (degrees)")
    distance: float = Field(50.0, description="Distance from target")
    target: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    fov: float = Field(45.0, ge=1, le=170, description="Field of view (degrees)")


class CameraUpdate(BaseModel):
    """Partial camera update — set any subset of fields."""

    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None
    target: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    fov: Optional[float] = None


class ScalarValue(BaseModel):
    """Single numeric value for individual parameter setters."""

    value: float


class Vec3Value(BaseModel):
    """3D vector for individual parameter setters (target)."""

    value: List[float] = Field(..., min_length=3, max_length=3)


class CameraPreset(BaseModel):
    """Named camera preset."""

    preset: str  # "Top", "Front", "Right", "Isometric"


class LookAtRequest(BaseModel):
    """Look-at request."""

    target: List[float] = Field(..., min_length=3, max_length=3)
    distance: Optional[float] = None


# ── Camera Animation ──────────────────────────────────────────────────


class CameraKeyframe(BaseModel):
    """A single camera keyframe."""

    t: float = Field(..., ge=0.0, le=1.0, description="Normalised time 0–1")
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None
    target: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    fov: Optional[float] = None


class CameraAnimation(BaseModel):
    """Keyframed camera animation."""

    id: str = "camera-anim"
    keyframes: List[CameraKeyframe]
    duration_ms: int = Field(3000, ge=0)
    easing: EasingSpec = "ease_in_out"
    loop: bool = False
    ping_pong: bool = False
    repeat_count: int = Field(0, ge=0, description="0 = infinite when loop=True")


# ── Layers ────────────────────────────────────────────────────────────


class LayerInfo(BaseModel):
    """Read-only layer metadata."""

    id: str
    name: str
    type: str  # "pointcloud", "mesh", "wireframe", "panorama", "dynamic_*"
    visible: bool
    opacity: float
    point_count: int = 0
    tri_count: int = 0
    dynamic: bool = False  # True if created via API


class LayerUpdate(BaseModel):
    """Mutable layer properties (works for any layer)."""

    visible: Optional[bool] = None
    opacity: Optional[float] = Field(None, ge=0.0, le=1.0)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)


# ── Dynamic Geometry Layers ───────────────────────────────────────────


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
    center: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    size: List[float] = Field(
        default_factory=lambda: [10.0, 10.0], min_length=2, max_length=2
    )
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: float = 0.3


class DynamicLayerCreate(BaseModel):
    """Create a dynamic geometry layer.

    ``geometry_type`` determines which geometry fields are required:

    * ``pointcloud``: *points* (required), *colors* (optional)
    * ``mesh``:       *vertices* + *triangles* (required), *normals* (optional)
    * ``bboxes``:     *bboxes* list (required)
    * ``surfaces``:   *surfaces* list (required)
    * ``file``:       *path* (required)

    ``layer_id`` is generated as ``f"dyn_{name}"``.  Names must be unique.
    """

    name: str
    geometry_type: GeometryType
    visible: bool = True
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)

    # Pointcloud geometry
    points: Optional[List[List[float]]] = None  # Nx3
    colors: Optional[List[List[float]]] = None  # Nx3 (0-255 int or 0-1 float)

    # Mesh geometry
    vertices: Optional[List[List[float]]] = None  # Nx3
    triangles: Optional[List[List[int]]] = None  # Mx3 vertex indices
    normals: Optional[List[List[float]]] = None  # Nx3

    # BBox collection
    bboxes: Optional[List[BBoxSpec]] = None

    # Surface collection
    surfaces: Optional[List[SurfaceSpec]] = None

    # File-based
    path: Optional[str] = None  # absolute path to STL/OBJ/PLY


class DynamicLayerPatch(BaseModel):
    """Property-only update (no geometry rebuild, 60 Hz safe)."""

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
    color: Optional[List[float]] = None
    point_count: int = 0
    tri_count: int = 0


# ── Animation (transforms) ───────────────────────────────────────────


class ContinuousTransform(BaseModel):
    """Continuous property change at a fixed rate.

    For scalar properties (azimuth, opacity…), ``rate`` is units/sec.
    For ``property="color"``, use ``target`` instead of ``rate`` — the
    colour interpolates linearly over ``duration_ms``.
    """

    id: str
    property: str  # "azimuth", "rotation_z", "opacity", "color", …
    rate: Optional[float] = None  # units per second (scalar props)
    target: Optional[List[float]] = None  # target value (for color: [r,g,b])
    duration_ms: int = Field(0, ge=0, description="0 = forever")
    layer_id: Optional[str] = None  # required for dynamic.transform_continuous


class DynamicTransformKeyframe(BaseModel):
    """A single keyframe for object animation."""

    t: float = Field(..., ge=0.0, le=1.0, description="Normalised time 0–1")
    position: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: Optional[float] = None
    scale: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: Optional[float] = None
    point_size: Optional[float] = None


class DynamicAnimation(BaseModel):
    """Keyframed animation for a dynamic layer."""

    id: str = "layer-anim"
    layer_id: str
    keyframes: List[DynamicTransformKeyframe]
    duration_ms: int = Field(3000, ge=0)
    easing: EasingSpec = "ease_in_out"
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


# ── Render Mode ───────────────────────────────────────────────────────


class RenderModeUpdate(BaseModel):
    """Switch between realtime and capture mode."""

    mode: Literal["realtime", "capture"]
    width: Optional[int] = Field(None, ge=1)
    height: Optional[int] = Field(None, ge=1)


class RenderModeState(BaseModel):
    """Current render mode and settings."""

    mode: str = "realtime"
    width: Optional[int] = None
    height: Optional[int] = None
    target_fps: int = 60


# ── Annotations (editor overlays) ────────────────────────────────────


class BBoxCreate(BaseModel):
    """Create an editor annotation bbox."""

    label: str = "custom"
    center: List[float] = Field(..., min_length=3, max_length=3)
    size: List[float] = Field(..., min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: float = 0.0
    fill_opacity: float = 0.0


class BBoxUpdate(BaseModel):
    """Update an editor annotation bbox (all fields optional)."""

    label: Optional[str] = None
    center: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    size: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation_z: Optional[float] = None
    fill_opacity: Optional[float] = None


class PlaneCreate(BaseModel):
    """Create an editor annotation plane."""

    axis: str = "xy"  # "xy", "xz", "yz"
    center: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    size: List[float] = Field(
        default_factory=lambda: [10.0, 10.0], min_length=2, max_length=2
    )
    color: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    opacity: float = 0.3


# ── Viewport ──────────────────────────────────────────────────────────


class ViewportSettings(BaseModel):
    """Mutable viewport render settings."""

    point_size: Optional[float] = Field(None, ge=1, le=20)
    show_axes: Optional[bool] = None
    show_grid: Optional[bool] = None
    use_layer_colors: Optional[bool] = None
    fps_movement: Optional[bool] = None
    point_attenuation: Optional[bool] = None
    bg_color: Optional[List[float]] = None
    # Vsync is read-only at runtime (locked into the GL context).
    # PUT will accept the field but mark `vsync_restart_required` in
    # the response if the requested value differs from the current.
    vsync: Optional[bool] = None


class CorrectionState(BaseModel):
    """Scene correction (rotation + shift for axis alignment)."""

    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_z: float = 0.0


class ClipState(BaseModel):
    """AABB clipping planes."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


# ── System ────────────────────────────────────────────────────────────


class SceneLoadRequest(BaseModel):
    """Load files by path."""

    paths: List[str]


class FolderLoadRequest(BaseModel):
    """Load all supported files from a folder."""

    path: str


class SystemStatus(BaseModel):
    """Server health / viewer summary."""

    mode: str  # "viewer" or "editor"
    layers_count: int
    dynamic_layers_count: int = 0
    total_points: int
    fps: float
    api_version: str = "1.0.0"
    server_port: int
