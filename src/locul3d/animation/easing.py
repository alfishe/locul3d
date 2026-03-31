"""Timing/easing functions for the animation engine.

Modeled after Core Animation (iOS/macOS) and the Web Animations API.
Every function maps normalised time ``t`` in [0, 1] to a progress value.
"""

from __future__ import annotations

import math
from typing import Callable

# Type alias — an easing function maps [0,1] → float (may overshoot)
EasingFn = Callable[[float], float]


# ── Named Presets ────────────────────────────────────────────────────


def linear(t: float) -> float:
    return t


def _make_cubic(x1: float, y1: float, x2: float, y2: float) -> EasingFn:
    """Build a cubic-bezier easing function from 4 control points."""
    return cubic_bezier(x1, y1, x2, y2)


# Standard CSS / Core Animation presets
NAMED_PRESETS: dict[str, EasingFn] = {}


def _register_presets():
    """Register all named easing presets."""
    global NAMED_PRESETS
    NAMED_PRESETS = {
        "linear": linear,
        "ease": cubic_bezier(0.25, 0.1, 0.25, 1.0),
        "ease_in": cubic_bezier(0.42, 0.0, 1.0, 1.0),
        "ease_out": cubic_bezier(0.0, 0.0, 0.58, 1.0),
        "ease_in_out": cubic_bezier(0.42, 0.0, 0.58, 1.0),
        "ease_in_cubic": cubic_bezier(0.32, 0.0, 0.67, 0.0),
        "ease_out_cubic": cubic_bezier(0.33, 1.0, 0.68, 1.0),
        "ease_in_out_cubic": cubic_bezier(0.65, 0.0, 0.35, 1.0),
        "ease_in_back": cubic_bezier(0.36, 0.0, 0.66, -0.56),
        "ease_out_back": cubic_bezier(0.34, 1.56, 0.64, 1.0),
        "ease_out_bounce": _ease_out_bounce,
    }


# ── Cubic Bezier ─────────────────────────────────────────────────────


def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> EasingFn:
    """Standard cubic Bezier curve — same algorithm as CSS/Core Animation.

    Control points: P0=(0,0), P1=(x1,y1), P2=(x2,y2), P3=(1,1).
    Uses Newton-Raphson to invert the x(t) curve, then evaluates y(t).
    """

    def _sample_x(t: float) -> float:
        return 3.0 * (1 - t) ** 2 * t * x1 + 3.0 * (1 - t) * t ** 2 * x2 + t ** 3

    def _sample_y(t: float) -> float:
        return 3.0 * (1 - t) ** 2 * t * y1 + 3.0 * (1 - t) * t ** 2 * y2 + t ** 3

    def _dx(t: float) -> float:
        return (3.0 * (1 - t) ** 2 * x1
                + 6.0 * (1 - t) * t * (x2 - x1)
                + 3.0 * t ** 2 * (1 - x2))

    def _solve_t(x: float) -> float:
        """Newton-Raphson to find t for given x."""
        t = x  # initial guess
        for _ in range(8):
            residual = _sample_x(t) - x
            d = _dx(t)
            if abs(d) < 1e-12:
                break
            t -= residual / d
            t = max(0.0, min(1.0, t))
        return t

    def easing(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return _sample_y(_solve_t(t))

    return easing


# ── Spring Physics ───────────────────────────────────────────────────


def spring_timing(
    damping: float = 0.7,
    stiffness: float = 100.0,
    mass: float = 1.0,
    initial_velocity: float = 0.0,
) -> EasingFn:
    """Critically-damped spring solver — matches CASpringAnimation.

    Returns a function that maps t in [0,1] to a progress value.
    Low damping (<1) produces overshoot/bounce.
    """
    omega0 = math.sqrt(stiffness / mass)
    zeta = damping  # damping ratio

    if zeta < 1.0:
        # Underdamped
        omega_d = omega0 * math.sqrt(1.0 - zeta ** 2)

        def easing(t: float) -> float:
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            # Scale t to ~5 time constants for the spring to settle
            T = t * 5.0 / (zeta * omega0) if zeta * omega0 > 0 else t * 5.0
            exp_term = math.exp(-zeta * omega0 * T)
            A = 1.0
            B = (zeta * omega0 + initial_velocity) / omega_d if omega_d > 0 else 0
            return 1.0 - exp_term * (A * math.cos(omega_d * T) + B * math.sin(omega_d * T))

    elif zeta == 1.0:
        # Critically damped
        def easing(t: float) -> float:
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            T = t * 5.0 / omega0 if omega0 > 0 else t * 5.0
            return 1.0 - math.exp(-omega0 * T) * (1.0 + omega0 * T)

    else:
        # Overdamped
        r1 = -omega0 * (zeta + math.sqrt(zeta ** 2 - 1))
        r2 = -omega0 * (zeta - math.sqrt(zeta ** 2 - 1))

        def easing(t: float) -> float:
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            T = t * 5.0 / abs(r1) if r1 != 0 else t * 5.0
            c2 = (r1 + initial_velocity) / (r1 - r2) if r1 != r2 else 0
            c1 = 1.0 - c2
            return 1.0 - (c1 * math.exp(r1 * T) + c2 * math.exp(r2 * T))

    return easing


# ── Step Functions ───────────────────────────────────────────────────


def step_timing(count: int = 1, position: str = "end") -> EasingFn:
    """Step function easing — matches CSS ``steps()``."""

    def easing(t: float) -> float:
        if t <= 0.0:
            return 0.0 if position == "end" else 1.0 / count
        if t >= 1.0:
            return 1.0
        step = int(t * count)
        if position == "start":
            step += 1
        return min(step / count, 1.0)

    return easing


# ── Bounce ───────────────────────────────────────────────────────────


def _ease_out_bounce(t: float) -> float:
    """Bouncing settle effect."""
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


# ── Resolver ─────────────────────────────────────────────────────────


def resolve_easing(spec) -> EasingFn:
    """Convert an easing spec (string, dict, or callable) to a t->t function.

    Accepted forms:
    - ``"ease_in_out"`` — named preset string
    - ``{"cubic_bezier": [x1, y1, x2, y2]}`` — custom cubic bezier
    - ``{"spring": {"damping": 0.6, ...}}`` — spring physics
    - ``{"steps": {"count": 10, "position": "end"}}`` — step function
    - A callable — returned as-is
    """
    if callable(spec):
        return spec

    if isinstance(spec, str):
        if not NAMED_PRESETS:
            _register_presets()
        fn = NAMED_PRESETS.get(spec)
        if fn is not None:
            return fn
        return linear  # fallback

    if isinstance(spec, dict):
        if "cubic_bezier" in spec:
            args = spec["cubic_bezier"]
            return cubic_bezier(*args)
        if "spring" in spec:
            params = spec["spring"] if isinstance(spec["spring"], dict) else {}
            return spring_timing(**params)
        if "steps" in spec:
            params = spec["steps"] if isinstance(spec["steps"], dict) else {}
            return step_timing(**params)

    return linear  # fallback


# Initialise presets on import
_register_presets()
