"""GLSL 1.20 shader for point cloud layers — view-Z fade effect.

Used optionally by ``BaseGLViewport._draw_point_layer`` to fade points
that lie *between the camera and the look-at target*.  The shader is
intentionally minimal: it relies on the OpenGL 2.1 compatibility built-ins
(``gl_Vertex``, ``gl_Color``, ``gl_ModelViewMatrix``, ``gl_FragColor``)
so the existing fixed-function VBO upload path (``glVertexPointer`` /
``glColorPointer``) continues to work without modification.

GLSL ``#version 120`` is the version that ships with OpenGL 2.1 — the
profile this app already requests via ``QSurfaceFormat.setVersion(2, 1)``.
That makes the shader portable across the same set of platforms the
rest of the renderer already supports (Windows / Linux / macOS).
"""

from __future__ import annotations

import logging

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUseProgram,
)

log = logging.getLogger(__name__)


_VERT_SRC = """\
#version 120
// Pass view-space position to the fragment shader. We need the full
// vec3 (not just Z) because the fade test is a cone, not a plane.
varying vec3 vViewPos;
varying vec4 vColor;
void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    vViewPos = viewPos.xyz;
    vColor   = gl_Color;
    gl_Position = gl_ProjectionMatrix * viewPos;
}
"""

# The fragment shader fades only points that lie INSIDE the cone swept
# from the camera (view-space origin) through the bounding sphere of
# the area of interest, AND in front of the sphere's near edge.
# Everything outside the cone or behind the AoI keeps its base alpha.
#
# Geometry (view space, camera at origin, looking down -Z):
#
#     camera                    AoI sphere (center C, radius R)
#       o ─────axis n──────────── O ─────────►
#        \\                     /
#         \\__cone half-angle __/
#         half_angle = asin(R / |C|)
#
# A vertex V is inside the cone iff
#     angle(V, n) <= half_angle
# i.e.
#     dot(normalize(V), n) >= cos(half_angle)
# It's "in front of" the AoI iff
#     |V| < |C| - R    (closer to the camera than the near edge)
_FRAG_SRC = """\
#version 120
uniform vec3  uAoiCenterView;  // AoI sphere center in view space
uniform float uAoiRadius;      // AoI sphere radius (world units)
uniform float uFadeBand;       // smoothstep softness (world units)
uniform float uFadeMul;        // alpha multiplier for occluders
uniform float uLayerAlpha;     // base layer opacity
uniform int   uFadeEnable;     // 0 = pass-through

varying vec3 vViewPos;
varying vec4 vColor;

void main() {
    float a = vColor.a * uLayerAlpha;

    if (uFadeEnable == 1 && uAoiRadius > 0.0) {
        float distC = length(uAoiCenterView);
        // Guard: if camera is inside the AoI sphere there's no
        // meaningful "in front of" test — skip fading.
        if (distC > uAoiRadius + 1e-3) {
            // Cone axis (camera→AoI center) and its half-angle.
            vec3  axis     = uAoiCenterView / distC;
            float cosHalf  = sqrt(max(0.0, 1.0 - (uAoiRadius * uAoiRadius)
                                                / (distC * distC)));

            float distV = length(vViewPos);
            // Avoid div-by-zero for vertices coincident with camera.
            if (distV > 1e-4) {
                vec3  vDir   = vViewPos / distV;
                float cosVA  = dot(vDir, axis);

                // 1) Inside the cone, with a small angular smoothstep
                //    so the edge isn't a hard step.
                float angBand = uAoiRadius * 0.5 / distC;  // ≈ rad
                float coneT   = smoothstep(cosHalf - angBand,
                                           cosHalf + angBand,
                                           cosVA);

                // 2) In front of the AoI's near edge (with smoothstep
                //    over a fadeBand-thick shell).
                float nearEdge = distC - uAoiRadius;
                float depthT   = 1.0 - smoothstep(nearEdge - uFadeBand,
                                                  nearEdge + uFadeBand,
                                                  distV);

                float t = coneT * depthT;
                a *= mix(1.0, uFadeMul, t);
            }
        }
    }

    gl_FragColor = vec4(vColor.rgb, a);
}
"""


class PointFadeShader:
    """Lazily-compiled GLSL 1.20 program for point cloud rendering.

    Compilation is deferred until ``ensure_compiled()`` is called inside
    a current GL context.  If compilation or linking fails the object
    enters a permanently disabled state and ``bind()`` becomes a no-op,
    so the caller can transparently fall back to fixed-function drawing.
    """

    def __init__(self) -> None:
        self._program: int = 0
        self._tried = False
        self._failed = False
        self._loc: dict[str, int] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    def ensure_compiled(self) -> bool:
        if self._tried:
            return not self._failed
        self._tried = True
        try:
            vs = self._compile(GL_VERTEX_SHADER, _VERT_SRC)
            fs = self._compile(GL_FRAGMENT_SHADER, _FRAG_SRC)
            prog = glCreateProgram()
            glAttachShader(prog, vs)
            glAttachShader(prog, fs)
            glLinkProgram(prog)
            if not glGetProgramiv(prog, GL_LINK_STATUS):
                err = glGetProgramInfoLog(prog).decode("utf-8", "replace")
                glDeleteProgram(prog)
                raise RuntimeError(f"link failed: {err}")
            glDeleteShader(vs)
            glDeleteShader(fs)
            self._program = prog
            for name in ("uAoiCenterView", "uAoiRadius", "uFadeBand",
                         "uFadeMul", "uLayerAlpha", "uFadeEnable"):
                self._loc[name] = glGetUniformLocation(prog, name)
            log.info("PointFadeShader compiled (program=%d)", prog)
            return True
        except Exception as exc:
            log.warning("PointFadeShader unavailable: %s", exc)
            self._failed = True
            self._program = 0
            return False

    def _compile(self, kind: int, src: str) -> int:
        sh = glCreateShader(kind)
        glShaderSource(sh, src)
        glCompileShader(sh)
        if not glGetShaderiv(sh, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(sh).decode("utf-8", "replace")
            glDeleteShader(sh)
            raise RuntimeError(f"compile failed: {err}")
        return sh

    @property
    def available(self) -> bool:
        return self._program != 0

    # ── Use ──────────────────────────────────────────────────────────

    def bind(self) -> bool:
        if not self.available:
            return False
        glUseProgram(self._program)
        return True

    def unbind(self) -> None:
        glUseProgram(0)

    def set_uniforms(
        self,
        *,
        aoi_center_view: tuple[float, float, float],
        aoi_radius: float,
        fade_band: float,
        fade_mul: float,
        layer_alpha: float,
        fade_enable: bool,
    ) -> None:
        if not self.available:
            return
        cx, cy, cz = aoi_center_view
        glUniform3f(self._loc["uAoiCenterView"], cx, cy, cz)
        glUniform1f(self._loc["uAoiRadius"], float(aoi_radius))
        glUniform1f(self._loc["uFadeBand"], float(fade_band))
        glUniform1f(self._loc["uFadeMul"], float(fade_mul))
        glUniform1f(self._loc["uLayerAlpha"], float(layer_alpha))
        glUniform1i(self._loc["uFadeEnable"], 1 if fade_enable else 0)
