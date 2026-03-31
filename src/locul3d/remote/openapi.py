"""OpenAPI 3.1 spec generation from Pydantic schemas and route metadata.

Serves at ``GET /openapi.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from . import schemas


def _schema_ref(model: Type[BaseModel]) -> dict:
    """Return a JSON Schema $ref for a Pydantic model."""
    return {"$ref": f"#/components/schemas/{model.__name__}"}


def _json_body(model: Type[BaseModel], required: bool = True) -> dict:
    return {
        "required": required,
        "content": {
            "application/json": {
                "schema": _schema_ref(model),
            }
        },
    }


def _json_response(model: Optional[Type[BaseModel]] = None, description: str = "OK") -> dict:
    if model:
        return {
            "200": {
                "description": description,
                "content": {"application/json": {"schema": _schema_ref(model)}},
            }
        }
    return {"200": {"description": description, "content": {"application/json": {"schema": {"type": "object"}}}}}


def _ok_response(description: str = "OK") -> dict:
    return {"200": {"description": description, "content": {"application/json": {"schema": {
        "type": "object",
        "properties": {"status": {"type": "string", "example": "ok"}},
    }}}}}


def generate_openapi_spec() -> dict:
    """Build the complete OpenAPI 3.1 specification."""

    # Collect all Pydantic model schemas
    models: List[Type[BaseModel]] = [
        schemas.CameraState, schemas.CameraUpdate, schemas.ScalarValue,
        schemas.Vec3Value, schemas.CameraPreset, schemas.LookAtRequest,
        schemas.CameraKeyframe, schemas.CameraAnimation,
        schemas.LayerInfo, schemas.LayerUpdate,
        schemas.GeometryType, schemas.BBoxSpec, schemas.SurfaceSpec,
        schemas.DynamicLayerCreate, schemas.DynamicLayerPatch, schemas.DynamicLayerInfo,
        schemas.ContinuousTransform, schemas.DynamicTransformKeyframe,
        schemas.DynamicAnimation, schemas.InstantTransform,
        schemas.RenderModeUpdate, schemas.RenderModeState,
        schemas.BBoxCreate, schemas.BBoxUpdate, schemas.PlaneCreate,
        schemas.ViewportSettings, schemas.CorrectionState, schemas.ClipState,
        schemas.SceneLoadRequest, schemas.FolderLoadRequest, schemas.SystemStatus,
    ]

    component_schemas = {}
    for model in models:
        try:
            model_schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
            # Extract $defs and merge them
            defs = model_schema.pop("$defs", {})
            for def_name, def_schema in defs.items():
                component_schemas[def_name] = def_schema
            component_schemas[model.__name__] = model_schema
        except Exception:
            component_schemas[model.__name__] = {"type": "object"}

    spec: Dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {
            "title": "Locul3D Remote Control API",
            "version": "1.0.0",
            "description": "Control the Locul3D 3D viewer and editor remotely via HTTP REST and WebSocket.",
        },
        "servers": [{"url": "http://localhost:8350", "description": "Local viewer"}],
        "paths": {
            # System
            "/api/v1/system/ping": {
                "get": {
                    "tags": ["System"],
                    "summary": "Heartbeat",
                    "responses": {"200": {"description": "Pong", "content": {"application/json": {"schema": {
                        "type": "object", "properties": {"pong": {"type": "boolean"}},
                    }}}}},
                },
            },
            "/api/v1/system/status": {
                "get": {
                    "tags": ["System"],
                    "summary": "Server health and viewer summary",
                    "responses": _json_response(schemas.SystemStatus),
                },
            },
            "/api/v1/system/screenshot": {
                "get": {
                    "tags": ["System"],
                    "summary": "Capture viewport as PNG",
                    "responses": {"200": {"description": "PNG image", "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
                },
            },
            # Camera
            "/api/v1/camera": {
                "get": {
                    "tags": ["Camera"],
                    "summary": "Get full camera state",
                    "responses": _json_response(schemas.CameraState),
                },
                "put": {
                    "tags": ["Camera"],
                    "summary": "Set camera (partial update)",
                    "requestBody": _json_body(schemas.CameraUpdate),
                    "responses": _json_response(schemas.CameraState),
                },
            },
            "/api/v1/camera/azimuth": {"put": {"tags": ["Camera"], "summary": "Set azimuth", "requestBody": _json_body(schemas.ScalarValue), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/elevation": {"put": {"tags": ["Camera"], "summary": "Set elevation", "requestBody": _json_body(schemas.ScalarValue), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/distance": {"put": {"tags": ["Camera"], "summary": "Set distance", "requestBody": _json_body(schemas.ScalarValue), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/fov": {"put": {"tags": ["Camera"], "summary": "Set FOV", "requestBody": _json_body(schemas.ScalarValue), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/target": {"put": {"tags": ["Camera"], "summary": "Set target", "requestBody": _json_body(schemas.Vec3Value), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/fit": {"post": {"tags": ["Camera"], "summary": "Fit camera to scene bounds", "responses": _ok_response()}},
            "/api/v1/camera/preset": {"post": {"tags": ["Camera"], "summary": "Apply named preset", "requestBody": _json_body(schemas.CameraPreset), "responses": _json_response(schemas.CameraState)}},
            "/api/v1/camera/look_at": {"post": {"tags": ["Camera"], "summary": "Look at point", "requestBody": _json_body(schemas.LookAtRequest), "responses": _json_response(schemas.CameraState)}},
            # Scene
            "/api/v1/scene/layers": {
                "get": {"tags": ["Scene"], "summary": "List all layers", "responses": {"200": {"description": "Layer list", "content": {"application/json": {"schema": {"type": "array", "items": _schema_ref(schemas.LayerInfo)}}}}}},
            },
            "/api/v1/scene/layers/{layer_id}": {
                "put": {"tags": ["Scene"], "summary": "Update layer properties", "parameters": [{"name": "layer_id", "in": "path", "required": True, "schema": {"type": "string"}}], "requestBody": _json_body(schemas.LayerUpdate), "responses": _ok_response()},
            },
            "/api/v1/scene/load": {"post": {"tags": ["Scene"], "summary": "Load files", "requestBody": _json_body(schemas.SceneLoadRequest), "responses": _ok_response()}},
            "/api/v1/scene/load_folder": {"post": {"tags": ["Scene"], "summary": "Load folder", "requestBody": _json_body(schemas.FolderLoadRequest), "responses": _ok_response()}},
            "/api/v1/scene/clear": {"delete": {"tags": ["Scene"], "summary": "Clear all layers", "responses": _ok_response()}},
            "/api/v1/scene/bounds": {"get": {"tags": ["Scene"], "summary": "Scene AABB", "responses": {"200": {"description": "Bounds", "content": {"application/json": {"schema": {"type": "object"}}}}}}},
            # Dynamic layers
            "/api/v1/scene/dynamic": {
                "get": {"tags": ["Dynamic Layers"], "summary": "List dynamic layers", "responses": {"200": {"description": "Dynamic layers", "content": {"application/json": {"schema": {"type": "array", "items": _schema_ref(schemas.DynamicLayerInfo)}}}}}},
                "post": {"tags": ["Dynamic Layers"], "summary": "Create dynamic layer", "requestBody": _json_body(schemas.DynamicLayerCreate), "responses": _json_response(schemas.DynamicLayerInfo, "Created")},
                "delete": {"tags": ["Dynamic Layers"], "summary": "Clear all dynamic layers", "responses": _ok_response()},
            },
            "/api/v1/scene/dynamic/{layer_id}": {
                "get": {"tags": ["Dynamic Layers"], "summary": "Get dynamic layer", "parameters": [{"name": "layer_id", "in": "path", "required": True, "schema": {"type": "string"}}], "responses": _json_response(schemas.DynamicLayerInfo)},
                "put": {"tags": ["Dynamic Layers"], "summary": "Update dynamic layer geometry", "parameters": [{"name": "layer_id", "in": "path", "required": True, "schema": {"type": "string"}}], "requestBody": _json_body(schemas.DynamicLayerCreate), "responses": _json_response(schemas.DynamicLayerInfo)},
                "patch": {"tags": ["Dynamic Layers"], "summary": "Patch dynamic layer properties", "parameters": [{"name": "layer_id", "in": "path", "required": True, "schema": {"type": "string"}}], "requestBody": _json_body(schemas.DynamicLayerPatch), "responses": _json_response(schemas.DynamicLayerInfo)},
                "delete": {"tags": ["Dynamic Layers"], "summary": "Delete dynamic layer", "parameters": [{"name": "layer_id", "in": "path", "required": True, "schema": {"type": "string"}}], "responses": _ok_response()},
            },
            # Shapes (annotations)
            "/api/v1/shapes/bboxes": {
                "get": {"tags": ["Shapes"], "summary": "List annotation bboxes", "responses": {"200": {"description": "BBox list", "content": {"application/json": {"schema": {"type": "array", "items": {"type": "object"}}}}}}},
                "post": {"tags": ["Shapes"], "summary": "Create annotation bbox", "requestBody": _json_body(schemas.BBoxCreate), "responses": _ok_response("Created")},
            },
            "/api/v1/shapes/bboxes/{index}": {
                "put": {"tags": ["Shapes"], "summary": "Update annotation bbox", "parameters": [{"name": "index", "in": "path", "required": True, "schema": {"type": "integer"}}], "requestBody": _json_body(schemas.BBoxUpdate), "responses": _ok_response()},
                "delete": {"tags": ["Shapes"], "summary": "Delete annotation bbox", "parameters": [{"name": "index", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": _ok_response()},
            },
            "/api/v1/shapes/planes": {
                "get": {"tags": ["Shapes"], "summary": "List annotation planes", "responses": {"200": {"description": "Plane list", "content": {"application/json": {"schema": {"type": "array", "items": {"type": "object"}}}}}}},
                "post": {"tags": ["Shapes"], "summary": "Create annotation plane", "requestBody": _json_body(schemas.PlaneCreate), "responses": _ok_response("Created")},
            },
            "/api/v1/shapes/planes/{index}": {
                "delete": {"tags": ["Shapes"], "summary": "Delete annotation plane", "parameters": [{"name": "index", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": _ok_response()},
            },
            # Viewport
            "/api/v1/viewport": {
                "get": {"tags": ["Viewport"], "summary": "Get viewport settings", "responses": _json_response(schemas.ViewportSettings)},
                "put": {"tags": ["Viewport"], "summary": "Update viewport settings", "requestBody": _json_body(schemas.ViewportSettings), "responses": _ok_response()},
            },
            "/api/v1/viewport/correction": {
                "get": {"tags": ["Viewport"], "summary": "Get scene correction", "responses": _json_response(schemas.CorrectionState)},
                "put": {"tags": ["Viewport"], "summary": "Set scene correction", "requestBody": _json_body(schemas.CorrectionState), "responses": _ok_response()},
            },
            "/api/v1/viewport/clip": {
                "get": {"tags": ["Viewport"], "summary": "Get clip planes", "responses": _json_response(schemas.ClipState)},
                "put": {"tags": ["Viewport"], "summary": "Set clip planes", "requestBody": _json_body(schemas.ClipState), "responses": _ok_response()},
                "delete": {"tags": ["Viewport"], "summary": "Remove clipping", "responses": _ok_response()},
            },
            "/api/v1/viewport/render_mode": {
                "get": {"tags": ["Viewport"], "summary": "Get render mode", "responses": _json_response(schemas.RenderModeState)},
                "put": {"tags": ["Viewport"], "summary": "Set render mode", "requestBody": _json_body(schemas.RenderModeUpdate), "responses": _ok_response()},
            },
        },
        "components": {
            "schemas": component_schemas,
        },
        "tags": [
            {"name": "System", "description": "Server health and screenshots"},
            {"name": "Camera", "description": "Camera position, orientation, and presets"},
            {"name": "Scene", "description": "Layer management, file loading, bounds"},
            {"name": "Dynamic Layers", "description": "API-created geometry layers (pointcloud, mesh, bboxes, surfaces)"},
            {"name": "Shapes", "description": "Editor annotation overlays (bboxes, planes)"},
            {"name": "Viewport", "description": "Render settings, scene correction, clipping, render mode"},
        ],
    }

    return spec
