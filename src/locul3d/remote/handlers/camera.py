"""Camera REST endpoints — full state + individual params + actions."""

from __future__ import annotations

from aiohttp import web

from ..dispatcher import CommandDispatcher
from ..schemas import CameraUpdate, LookAtRequest, ScalarValue, Vec3Value


def setup_routes(app: web.Application, dispatcher: CommandDispatcher) -> None:
    # Full state
    app.router.add_get("/api/v1/camera", _get_camera)
    app.router.add_put("/api/v1/camera", _set_camera)
    # Individual params
    app.router.add_put("/api/v1/camera/azimuth", _set_azimuth)
    app.router.add_put("/api/v1/camera/elevation", _set_elevation)
    app.router.add_put("/api/v1/camera/distance", _set_distance)
    app.router.add_put("/api/v1/camera/fov", _set_fov)
    app.router.add_put("/api/v1/camera/target", _set_target)
    # Actions
    app.router.add_post("/api/v1/camera/fit", _fit)
    app.router.add_post("/api/v1/camera/preset", _preset)
    app.router.add_post("/api/v1/camera/look_at", _look_at)


async def _get_camera(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    state = await dispatcher.get_camera_state()
    return web.json_response(state.model_dump())


async def _set_camera(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    update = CameraUpdate(**data)
    state = await dispatcher.set_camera(update)
    return web.json_response(state.model_dump())


async def _set_azimuth(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    v = ScalarValue(**(await request.json()))
    state = await dispatcher.set_camera(CameraUpdate(azimuth=v.value))
    return web.json_response(state.model_dump())


async def _set_elevation(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    v = ScalarValue(**(await request.json()))
    state = await dispatcher.set_camera(CameraUpdate(elevation=v.value))
    return web.json_response(state.model_dump())


async def _set_distance(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    v = ScalarValue(**(await request.json()))
    state = await dispatcher.set_camera(CameraUpdate(distance=v.value))
    return web.json_response(state.model_dump())


async def _set_fov(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    v = ScalarValue(**(await request.json()))
    state = await dispatcher.set_camera(CameraUpdate(fov=v.value))
    return web.json_response(state.model_dump())


async def _set_target(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    v = Vec3Value(**(await request.json()))
    state = await dispatcher.set_camera(CameraUpdate(target=v.value))
    return web.json_response(state.model_dump())


async def _fit(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    result = await dispatcher.fit_camera()
    return web.json_response(result)


async def _preset(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    preset = data.get("preset", "Isometric")
    state = await dispatcher.camera_preset(preset)
    return web.json_response(state.model_dump())


async def _look_at(request: web.Request) -> web.Response:
    dispatcher: CommandDispatcher = request.app["dispatcher"]
    data = await request.json()
    req = LookAtRequest(**data)
    state = await dispatcher.camera_look_at(req)
    return web.json_response(state.model_dump())
