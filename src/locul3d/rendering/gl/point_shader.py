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
    glUniform3fv,
    glUseProgram,
)

log = logging.getLogger(__name__)


_VERT_SRC = """\
#version 120
// Pass view-space position AND clip-space position to the fragment
// shader.  The box-aligned fade test uses clip-space (post-projection)
// coordinates to check whether a fragment falls inside the AoI bbox's
// screen-space silhouette, and view-space Z to check whether it lies
// in front of the bbox's nearest face.
//
// Also writes gl_PointSize from a uniform.  When a shader program is
// bound, some drivers (notably macOS Metal) ignore the fixed-function
// glPointSize() and fall back to 1.0 unless the vertex shader writes
// gl_PointSize explicitly — which would otherwise make points appear
// 1 pixel wide and look like "lower density" when fade is enabled.
uniform float uPointSize;

varying vec3 vViewPos;
varying vec4 vClipPos;
varying vec4 vColor;
void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    vViewPos = viewPos.xyz;
    gl_Position = gl_ProjectionMatrix * viewPos;
    vClipPos = gl_Position;
    vColor = gl_Color;
    gl_PointSize = uPointSize;
}
"""

# Box-aligned fade with a mirror back-fade.  Per draw call, Python
# projects the AoI bbox's 8 corners through the live MVP and computes:
#   * uAoiNdcMin / uAoiNdcMax — screen-space AABB of the projection
#   * uAoiNearViewZ          — least-negative view-space Z (the face
#                              of the bbox closest to the camera)
#   * uAoiFarViewZ           — most-negative view-space Z  (the face
#                              of the bbox farthest from the camera)
#
# Three regions for any fragment whose screen-space position is
# inside the bbox silhouette rectangle:
#
#   FRONT  — vertex is closer to the camera than the bbox front face.
#            These are the actual occluders.  Faded by `fade_mul`.
#
#   INSIDE — vertex lies within the bbox depth range (between the
#            front and back faces). These are AoI vertices we want
#            to keep at full opacity.  No fade applied.
#
#   BACK   — vertex is farther from the camera than the bbox back
#            face.  Dense backgrounds visible through the bbox can
#            create the illusion of unfaded "ceiling" behind it.
#            Faded by a softer mid-strength so the AoI still pops
#            without the back wall blending into the foreground.
#            Strength = mix(1, fade_mul, 0.5) — the midway alpha
#            multiplier between "no fade" and "full fade".
#
# All three tests use the same NDC silhouette so the fade region
# always exactly tracks the bbox as the camera orbits.  No 3D shadow
# geometry is rendered anywhere.
_FRAG_SRC = """\
#version 120
// Convex hull of the 8 projected bbox corners (in NDC).  Each edge
// is a line equation `a*x + b*y + c = 0` with the CCW convention
// that "inside the polygon" means a*x + b*y + c >= 0.  Unused
// slots are set to (0, 0, 1) → always inside, no contribution.
#define HULL_MAX 8
uniform vec3  uHullEdges[HULL_MAX];
uniform int   uHullEdgeCount;
uniform float uAoiNearViewZ;  // least-negative view-space Z (front face)
uniform float uAoiFarViewZ;   // most-negative view-space Z  (back face)
uniform float uFadeBand;      // smoothstep softness (NDC units)
uniform float uFadeMul;       // alpha multiplier for front occluders
uniform float uLayerAlpha;    // base layer opacity
uniform int   uFadeEnable;    // 0 = pass-through, 1 = apply fade
uniform int   uAoiValid;      // 0 = invalid (camera inside bbox etc.)

varying vec3 vViewPos;
varying vec4 vClipPos;
varying vec4 vColor;

void main() {
    float a = vColor.a * uLayerAlpha;

    if (uFadeEnable == 1 && uAoiValid == 1 && vClipPos.w > 1e-5) {
        vec2 ndc = vClipPos.xy / vClipPos.w;

        // Inside-hull test: for each edge compute the signed distance
        // to the half-plane.  The minimum across all edges is the
        // signed distance to the boundary; positive = inside.
        // GLSL 1.20 requires constant loop bounds, so we walk the
        // full HULL_MAX and skip unused slots (which are set to a
        // tautological "always inside" line equation).
        float minDist = 1.0;
        for (int i = 0; i < HULL_MAX; i++) {
            if (i >= uHullEdgeCount) break;
            float d = uHullEdges[i].x * ndc.x
                    + uHullEdges[i].y * ndc.y
                    + uHullEdges[i].z;
            minDist = min(minDist, d);
        }
        // Smoothstep across the boundary so the edge isn't a hard
        // step. uFadeBand is the half-band in NDC units.
        float silhouette = smoothstep(-uFadeBand, uFadeBand, minDist);

        // FRONT: vertex closer to camera than bbox front face.
        float front = step(uAoiNearViewZ, vViewPos.z);
        // BACK: vertex deeper than bbox back face.
        float back  = step(vViewPos.z, uAoiFarViewZ);

        // Mid-strength multiplier for the back region: midway
        // between "no fade" (1.0) and "full fade" (uFadeMul).
        float backMul = mix(1.0, uFadeMul, 0.5);

        float frontFactor = silhouette * front;
        float backFactor  = silhouette * back;

        float fadedAlpha = a;
        fadedAlpha = mix(fadedAlpha, fadedAlpha * uFadeMul, frontFactor);
        fadedAlpha = mix(fadedAlpha, fadedAlpha * backMul,  backFactor);
        a = fadedAlpha;
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
            for name in ("uHullEdges", "uHullEdgeCount",
                         "uAoiNearViewZ", "uAoiFarViewZ",
                         "uFadeBand", "uFadeMul", "uLayerAlpha",
                         "uFadeEnable", "uAoiValid", "uPointSize"):
                self._loc[name] = glGetUniformLocation(prog, name)
            # Each array element gets its own location too — query
            # the first slot's location and assume contiguous storage
            # (which the GLSL spec guarantees for plain arrays).
            base = glGetUniformLocation(prog, "uHullEdges[0]")
            if base != -1:
                self._loc["uHullEdges"] = base
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

    HULL_MAX = 8

    def set_uniforms(
        self,
        *,
        hull_edges,            # iterable of (a, b, c) line equations
        aoi_near_view_z: float,
        aoi_far_view_z: float,
        aoi_valid: bool,
        fade_band: float,
        fade_mul: float,
        layer_alpha: float,
        fade_enable: bool,
        point_size: float,
    ) -> None:
        if not self.available:
            return
        # Pad the hull edges out to HULL_MAX with a tautological
        # "always inside" line equation (a=0, b=0, c=1 means any
        # (x,y) yields 1 ≥ 0).  This lets the shader run a fixed
        # loop count which GLSL 1.20 requires.
        edges = list(hull_edges)[: self.HULL_MAX]
        count = len(edges)
        while len(edges) < self.HULL_MAX:
            edges.append((0.0, 0.0, 1.0))
        flat = []
        for a, b, c in edges:
            flat.extend((float(a), float(b), float(c)))
        loc = self._loc.get("uHullEdges", -1)
        if loc != -1:
            glUniform3fv(loc, self.HULL_MAX, flat)
        glUniform1i(self._loc["uHullEdgeCount"], int(count))
        glUniform1f(self._loc["uAoiNearViewZ"], float(aoi_near_view_z))
        glUniform1f(self._loc["uAoiFarViewZ"], float(aoi_far_view_z))
        glUniform1f(self._loc["uFadeBand"], float(fade_band))
        glUniform1f(self._loc["uFadeMul"], float(fade_mul))
        glUniform1f(self._loc["uLayerAlpha"], float(layer_alpha))
        glUniform1i(self._loc["uFadeEnable"], 1 if fade_enable else 0)
        glUniform1i(self._loc["uAoiValid"], 1 if aoi_valid else 0)
        loc_size = self._loc.get("uPointSize", -1)
        if loc_size != -1:
            glUniform1f(loc_size, float(point_size))
