"""Interpolation utilities for the animation engine.

Provides lerp, shortest-arc angle interpolation, and colour space helpers.
"""

from __future__ import annotations

from typing import List, Optional


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two scalars."""
    return a + (b - a) * t


def lerp_vec3(a: List[float], b: List[float], t: float) -> List[float]:
    """Linear interpolation between two 3D vectors."""
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate between two angles (degrees) using shortest arc.

    Always takes the shortest path around the circle, so e.g.
    lerp_angle(350, 10, 0.5) == 0.0, not 180.0.
    """
    diff = ((b - a) + 180.0) % 360.0 - 180.0
    return a + diff * t


def interpolate_keyframes(
    keyframes: list,
    t: float,
    properties: list[str],
    angle_properties: Optional[set] = None,
    vec3_properties: Optional[set] = None,
) -> dict:
    """Interpolate between keyframes at normalised time t.

    Args:
        keyframes: List of keyframe dicts, each with a ``t`` field (0-1)
                   and optional property fields.
        t: Normalised time in [0, 1].
        properties: List of property names to interpolate.
        angle_properties: Set of property names that are angles (use shortest-arc).
        vec3_properties: Set of property names that are 3D vectors.

    Returns:
        Dict of interpolated property values (only includes properties
        that appear in at least one keyframe).
    """
    if not keyframes:
        return {}

    angle_properties = angle_properties or set()
    vec3_properties = vec3_properties or set()

    # Clamp t
    t = max(0.0, min(1.0, t))

    # Find the bracketing keyframes
    if t <= keyframes[0].get("t", 0.0):
        return _extract_props(keyframes[0], properties)
    if t >= keyframes[-1].get("t", 1.0):
        return _extract_props(keyframes[-1], properties)

    # Find the two keyframes that bracket t
    kf_a = keyframes[0]
    kf_b = keyframes[-1]
    for i in range(len(keyframes) - 1):
        t_a = keyframes[i].get("t", 0.0)
        t_b = keyframes[i + 1].get("t", 1.0)
        if t_a <= t <= t_b:
            kf_a = keyframes[i]
            kf_b = keyframes[i + 1]
            break

    # Compute local t within the segment
    t_a = kf_a.get("t", 0.0)
    t_b = kf_b.get("t", 1.0)
    seg_len = t_b - t_a
    local_t = (t - t_a) / seg_len if seg_len > 0 else 1.0

    result = {}
    for prop in properties:
        val_a = _get_prop(kf_a, prop)
        val_b = _get_prop(kf_b, prop)
        if val_a is None and val_b is None:
            continue
        if val_a is None:
            val_a = val_b
        if val_b is None:
            val_b = val_a

        if prop in vec3_properties:
            result[prop] = lerp_vec3(val_a, val_b, local_t)
        elif prop in angle_properties:
            result[prop] = lerp_angle(val_a, val_b, local_t)
        else:
            result[prop] = lerp(val_a, val_b, local_t)

    return result


def _get_prop(kf: dict, prop: str):
    """Extract a property from a keyframe (dict or Pydantic model)."""
    if isinstance(kf, dict):
        return kf.get(prop)
    return getattr(kf, prop, None)


def _extract_props(kf: dict, properties: list[str]) -> dict:
    """Extract all non-None properties from a keyframe."""
    result = {}
    for prop in properties:
        val = _get_prop(kf, prop)
        if val is not None:
            result[prop] = val
    return result
