"""UV sphere mesh generation for panorama rendering."""

import math
import numpy as np


def build_sphere(n_lat: int = 64, n_lon: int = 128):
    """Build an inside-out UV sphere suitable for equirectangular texturing.

    Returns (verts, uvs, tris) as contiguous float32/uint32 arrays.
    Inside-out winding means the texture is visible when viewed from the
    centre of the sphere — exactly what immersive panorama needs.

    Parameters
    ----------
    n_lat : int
        Number of latitude subdivisions (poles to equator).
    n_lon : int
        Number of longitude subdivisions (full 360°).

    Returns
    -------
    verts : ndarray, shape ((n_lat+1)*(n_lon+1), 3), dtype float32
    uvs   : ndarray, shape ((n_lat+1)*(n_lon+1), 2), dtype float32
    tris  : ndarray, shape (n_lat*n_lon*2, 3), dtype uint32
    """
    verts = []
    uvs = []
    tris = []

    for lat in range(n_lat + 1):
        theta = math.pi * lat / n_lat  # 0..pi (north pole → south pole)
        for lon in range(n_lon + 1):
            phi = 2 * math.pi * lon / n_lon  # 0..2pi
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            verts.append([x, y, z])
            uvs.append([lon / n_lon, lat / n_lat])

    for lat in range(n_lat):
        for lon in range(n_lon):
            i0 = lat * (n_lon + 1) + lon
            i1 = i0 + 1
            i2 = i0 + (n_lon + 1)
            i3 = i2 + 1
            # Inside-out winding (CW when viewed from outside)
            tris.append([i0, i2, i1])
            tris.append([i1, i2, i3])

    return (
        np.array(verts, dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(tris, dtype=np.uint32),
    )
