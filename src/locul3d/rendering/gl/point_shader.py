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
from typing import cast

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
    glUniformMatrix4fv,
    glUseProgram,
)

log = logging.getLogger(__name__)


_VERT_SRC = """\
#version 120
// Point fade shader — TRUE 3D culling.
//
// Vertex shader forwards:
//   * vWorldPos — corrected-world position of the vertex.  The
//                 fragment shader uses this + the camera world
//                 position + the bbox (all in corrected world
//                 space) to run a proper ray-AABB test for the
//                 "is this vertex between camera and target"
//                 culling condition.  No screen-space projection
//                 approximation — all geometry survives rotation.
//
// gl_Vertex is in PRE-correction world space (the raw point cloud
// data).  uCorrectionMat transforms it into the corrected world
// space the AoI bbox lives in so the ray-AABB test has a consistent
// frame of reference.
uniform float uPointSize;
uniform mat4  uCorrectionMat;

varying vec3 vWorldPos;
varying vec4 vColor;

void main() {
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
    vec4 corrected = uCorrectionMat * gl_Vertex;
    vWorldPos = corrected.xyz;
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
// TRUE 3D pyramid culling via ray-AABB in corrected-world space.
//
// For each fragment, cast a ray from the camera through the vertex
// and test intersection with the target bbox.  The vertex is at
// parameter t=1 on its own ray.
//
// Region table (v1, the simple ray-AABB semantic):
//   tEnter > 1         → STRICTLY in front of bbox (occluder) → CULL
//   tEnter <= 1 <= tExit → INSIDE the bbox (AoI itself)       → KEEP
//   tExit  < 1         → STRICTLY past bbox (behind target)   → BACK
//   ray misses bbox    → NOT in the cone                       → KEEP
//
// CRUCIAL: the WHOLE bbox interior is preserved automatically
// because inside-bbox vertices have tEnter < 1, so the front-cone
// check `tEnter > 1` never fires for them.  No separate "core"
// bbox needed — the outer bbox IS the AoI.
//
// Float-precision safety band around tEnter=1 (asymmetric smoothstep)
// keeps bbox-surface vertices stable across frames even under
// worst-case rounding drift — see the eps_low/eps_high values.
uniform vec3  uCameraW;
uniform vec3  uAoiMinW;
uniform vec3  uAoiMaxW;
uniform float uFadeMul;
uniform float uLayerAlpha;
uniform int   uFadeEnable;
uniform int   uAoiValid;
uniform int   uDiscardCulled;

varying vec3 vWorldPos;
varying vec4 vColor;

void main() {
    float a = vColor.a * uLayerAlpha;

    if (uFadeEnable == 1 && uAoiValid == 1) {
        vec3 dir = vWorldPos - uCameraW;
        // Guard against zero components to avoid inf/NaN in division.
        vec3 safeDir;
        safeDir.x = (abs(dir.x) < 1e-20) ? 1e-20 : dir.x;
        safeDir.y = (abs(dir.y) < 1e-20) ? 1e-20 : dir.y;
        safeDir.z = (abs(dir.z) < 1e-20) ? 1e-20 : dir.z;
        vec3 invDir = 1.0 / safeDir;

        vec3 t1 = (uAoiMinW - uCameraW) * invDir;
        vec3 t2 = (uAoiMaxW - uCameraW) * invDir;
        vec3 tMin3 = min(t1, t2);
        vec3 tMax3 = max(t1, t2);
        float tEnter = max(max(tMin3.x, tMin3.y), tMin3.z);
        float tExit  = min(min(tMax3.x, tMax3.y), tMax3.z);

        // Ray actually intersects the forward half-line.
        float hit = step(tEnter, tExit) * step(0.0, tExit);

        // FLOAT-PRECISION SAFETY BAND — do NOT narrow without testing.
        //
        // Hard `step(1 + eps, tEnter)` causes bbox-surface vertices
        // (tEnter ~ 1.0 with float drift) to flicker in and out of
        // the culled set as the camera rotates.  In discard mode
        // this flicker is a visible "wipe ring" formed by bbox
        // edges sweeping across the scene — exactly the bug we
        // keep hitting on the recording path where MSAA on the
        // widget hides the flicker but the FBO/video captures it.
        //
        // The smoothstep band must be:
        //   - wide enough to absorb float drift (>= 1e-2)
        //   - start well past 1.0 so bbox near-face vertices
        //     (tEnter ~ 1.0) are never culled
        //   - end before realistic occluder tEnter values
        //
        // eps_low  = 5e-3   (10x float drift, start of transition)
        // eps_high = 5e-2   (wide enough for stable video frames)
        float eps_low  = 5e-3;
        float eps_high = 5e-2;

        // Cull vertices STRICTLY in front of the bbox's near face.
        // Vertices inside the bbox (tEnter < 1) or on the near face
        // (tEnter ≈ 1) have inFrontCone = 0 → fully preserved as AoI.
        float inFrontCone = hit * smoothstep(1.0 + eps_low,
                                             1.0 + eps_high, tEnter);
        // Back fade: vertices STRICTLY past the bbox far face.
        float behindBbox  = hit * (1.0 - smoothstep(1.0 - eps_high,
                                                    1.0 - eps_low, tExit));

        if (uDiscardCulled == 1 && inFrontCone > 0.5) {
            discard;
        }

        float backMul = mix(1.0, uFadeMul, 0.5);
        float fadedAlpha = a;
        fadedAlpha = mix(fadedAlpha, fadedAlpha * uFadeMul, inFrontCone);
        fadedAlpha = mix(fadedAlpha, fadedAlpha * backMul,  behindBbox);
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
            prog = cast(int, glCreateProgram())
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
            for name in ("uCameraW",
                         "uAoiMinW", "uAoiMaxW",
                         "uCorrectionMat",
                         "uFadeMul", "uLayerAlpha",
                         "uFadeEnable", "uAoiValid", "uPointSize",
                         "uDiscardCulled"):
                self._loc[name] = glGetUniformLocation(prog, name)
            log.info("PointFadeShader compiled (program=%d)", prog)
            return True
        except Exception as exc:
            log.warning("PointFadeShader unavailable: %s", exc)
            self._failed = True
            self._program = 0
            return False

    def _compile(self, kind: int, src: str) -> int:
        sh = cast(int, glCreateShader(kind))
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
        camera_world,          # (3,) camera position in corrected world space
        aoi_min_world,         # (3,) target bbox min in corrected world space
        aoi_max_world,         # (3,) target bbox max in corrected world space
        correction_matrix,     # 4x4 row-major numpy; identity if no correction
        aoi_valid: bool,
        fade_mul: float,
        layer_alpha: float,
        fade_enable: bool,
        point_size: float,
        discard_culled: bool = False,
    ) -> None:
        if not self.available:
            return
        glUniform3f(self._loc["uCameraW"],
                    float(camera_world[0]),
                    float(camera_world[1]),
                    float(camera_world[2]))
        glUniform3f(self._loc["uAoiMinW"],
                    float(aoi_min_world[0]),
                    float(aoi_min_world[1]),
                    float(aoi_min_world[2]))
        glUniform3f(self._loc["uAoiMaxW"],
                    float(aoi_max_world[0]),
                    float(aoi_max_world[1]),
                    float(aoi_max_world[2]))
        # Correction matrix: upload with transpose=GL_TRUE so the
        # row-major numpy matrix is converted to column-major on
        # the GPU automatically.
        loc_corr = self._loc.get("uCorrectionMat", -1)
        if loc_corr is not None and loc_corr != -1:
            import numpy as _np
            from OpenGL.GL import GL_TRUE
            cm = _np.ascontiguousarray(correction_matrix, dtype=_np.float32)
            if cm.shape != (4, 4):
                cm = cm.reshape(4, 4)
            glUniformMatrix4fv(loc_corr, 1, GL_TRUE, cm)
        glUniform1f(self._loc["uFadeMul"], float(fade_mul))
        glUniform1f(self._loc["uLayerAlpha"], float(layer_alpha))
        glUniform1i(self._loc["uFadeEnable"], 1 if fade_enable else 0)
        glUniform1i(self._loc["uAoiValid"], 1 if aoi_valid else 0)
        loc_disc = self._loc.get("uDiscardCulled", -1)
        if loc_disc != -1:
            glUniform1i(loc_disc, 1 if discard_culled else 0)
        loc_size = self._loc.get("uPointSize", -1)
        if loc_size != -1:
            glUniform1f(loc_size, float(point_size))
