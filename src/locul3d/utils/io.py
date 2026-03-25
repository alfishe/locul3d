"""Geometry loading utilities — single loader for PLY/OBJ/STL files.

This is the canonical geometry loader. All other import paths delegate here.
Includes a fast binary PLY parser (~3 GB/s, SSD-bound) that bypasses Open3D's
slow PLY reader (~100 MB/s).  Falls back to Open3D for ASCII PLY, meshes,
wireframes, OBJ, and STL.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fast binary PLY parser
# ---------------------------------------------------------------------------

# PLY property type → (numpy dtype, byte size)
_PLY_DTYPES = {
    'float':   ('f4', 4), 'float32':  ('f4', 4), 'double': ('f8', 8),
    'float64': ('f8', 8), 'uchar':    ('u1', 1), 'uint8':  ('u1', 1),
    'char':    ('i1', 1), 'int8':     ('i1', 1), 'ushort': ('u2', 2),
    'uint16':  ('u2', 2), 'short':    ('i2', 2), 'int16':  ('i2', 2),
    'uint':    ('u4', 4), 'uint32':   ('u4', 4), 'int':    ('i4', 4),
    'int32':   ('i4', 4),
}


def _try_fast_ply(path: str, layer: 'LayerData') -> bool:
    """Try to load a binary PLY using numpy.  Returns True on success.

    Handles binary_little_endian vertex-only PLYs (point clouds).
    Falls back (returns False) for ASCII, big-endian, list properties
    (face indices), or unsupported property types.
    """
    with open(path, 'rb') as f:
        # --- Parse header --------------------------------------------------
        magic = f.readline()
        if not magic.startswith(b'ply'):
            return False

        is_binary_le = False
        n_verts = 0
        has_faces = False
        props = []          # [(name, numpy_dtype, byte_size), ...]
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                return False
            line = line.strip()
            if line == b'end_header':
                break

            parts = line.split()
            if not parts:
                continue

            if parts[0] == b'format':
                if parts[1] == b'binary_little_endian':
                    is_binary_le = True
                else:
                    return False  # ASCII or big-endian → fallback

            elif parts[0] == b'element':
                if parts[1] == b'vertex':
                    n_verts = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False
                    if parts[1] == b'face':
                        has_faces = (int(parts[2]) > 0) if len(parts) > 2 else False
                    elif parts[1] == b'edge':
                        return False  # wireframe → use O3D line_set reader

            elif parts[0] == b'property' and in_vertex:
                if parts[1] == b'list':
                    return False  # list properties → complex, use O3D
                type_name = parts[1].decode('ascii', errors='replace')
                prop_name = parts[2].decode('ascii', errors='replace')
                if type_name not in _PLY_DTYPES:
                    return False
                np_dtype, byte_sz = _PLY_DTYPES[type_name]
                props.append((prop_name, np_dtype, byte_sz))

        if not is_binary_le or n_verts == 0 or not props:
            return False
        if has_faces:
            return False  # meshes need face parsing → use O3D

        # --- Compute vertex layout ----------------------------------------
        vertex_size = sum(p[2] for p in props)
        dtype_list = [(p[0], p[1]) for p in props]
        vertex_dtype = np.dtype(dtype_list)

        # Verify no padding was inserted by numpy
        if vertex_dtype.itemsize != vertex_size:
            return False

        # --- Read binary vertex data in one shot --------------------------
        header_end = f.tell()

    # Read from disk: single sequential I/O at full SSD speed
    raw = np.fromfile(path, dtype=np.uint8)
    data_bytes = raw[header_end : header_end + n_verts * vertex_size]

    if len(data_bytes) != n_verts * vertex_size:
        return False

    # --- Extract fields via row-reshape (faster than structured dtype) ----
    prop_names = [p[0] for p in props]
    prop_offsets = {}  # name → byte offset within vertex
    off = 0
    for name, np_dt, sz in props:
        prop_offsets[name] = (off, np_dt, sz)
        off += sz

    if not all(n in prop_offsets for n in ('x', 'y', 'z')):
        return False

    # Reshape to (n_verts, vertex_size) for fast column slicing
    rows = data_bytes.reshape(n_verts, vertex_size)

    # XYZ — contiguous float32 slice if x,y,z are consecutive float32
    x_off = prop_offsets['x'][0]
    xyz_end = x_off + 12  # 3 × float32
    if (prop_offsets['y'][0] == x_off + 4
            and prop_offsets['z'][0] == x_off + 8
            and prop_offsets['x'][1] == 'f4'):
        # Fast path: xyz are consecutive float32
        layer.points = rows[:, x_off:xyz_end].copy().view(np.float32).reshape(
            n_verts, 3)
    else:
        # General path: extract each coordinate separately
        xyz = np.empty((n_verts, 3), dtype=np.float32)
        for i, name in enumerate(('x', 'y', 'z')):
            o, dt, sz = prop_offsets[name]
            col = rows[:, o:o + sz].copy().view(dt).ravel()
            xyz[:, i] = col if dt == 'f4' else col.astype(np.float32)
        layer.points = xyz

    # RGB colors
    if all(n in prop_offsets for n in ('red', 'green', 'blue')):
        r_off, r_dt, r_sz = prop_offsets['red']
        g_off = prop_offsets['green'][0]
        b_off = prop_offsets['blue'][0]
        if (r_dt == 'u1' and g_off == r_off + 1 and b_off == r_off + 2):
            # Fast path: consecutive uint8 RGB — zero-copy slice
            layer.colors_u8 = rows[:, r_off:r_off + 3].copy()
        elif r_dt == 'u1':
            rgb = np.empty((n_verts, 3), dtype=np.uint8)
            rgb[:, 0] = rows[:, r_off]
            rgb[:, 1] = rows[:, g_off]
            rgb[:, 2] = rows[:, b_off]
            layer.colors_u8 = rgb
        else:
            rgb = np.empty((n_verts, 3), dtype=np.float32)
            for i, name in enumerate(('red', 'green', 'blue')):
                o, dt, sz = prop_offsets[name]
                col = rows[:, o:o + sz].copy().view(dt).ravel()
                rgb[:, i] = col if dt == 'f4' else col.astype(np.float32)
            layer.colors = rgb

    # Normals
    if all(n in prop_offsets for n in ('nx', 'ny', 'nz')):
        nx_off = prop_offsets['nx'][0]
        if (prop_offsets['ny'][0] == nx_off + 4
                and prop_offsets['nz'][0] == nx_off + 8
                and prop_offsets['nx'][1] == 'f4'):
            layer.normals = rows[:, nx_off:nx_off + 12].copy().view(
                np.float32).reshape(n_verts, 3)
        else:
            normals = np.empty((n_verts, 3), dtype=np.float32)
            for i, name in enumerate(('nx', 'ny', 'nz')):
                o, dt, sz = prop_offsets[name]
                col = rows[:, o:o + sz].copy().view(dt).ravel()
                normals[:, i] = col if dt == 'f4' else col.astype(np.float32)
            layer.normals = normals

    layer.point_count = n_verts
    layer.loaded = True

    del raw, data_bytes, rows
    import gc; gc.collect()
    return True


# ---------------------------------------------------------------------------
# Open3D fallback loader
# ---------------------------------------------------------------------------

def _load_with_open3d(path: str, layer: 'LayerData'):
    """Load geometry via Open3D (meshes, wireframes, ASCII PLY, OBJ, STL)."""
    import open3d as o3d
    # Suppress "appears to be a PointCloud" and similar warnings
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # For PLY/OBJ point clouds that reached the fallback, skip the mesh
    # probe (it reads the entire file just to discover no triangles).
    is_ply_or_obj = path.lower().endswith(('.ply', '.obj', '.stl'))

    # --- Try line set (wireframe) first ---
    try:
        line_set = o3d.io.read_line_set(path)
        if (line_set.has_points() and len(line_set.points) > 0
                and len(line_set.lines) > 0):
            pts = np.asarray(line_set.points, dtype=np.float32)
            lines = np.asarray(line_set.lines, dtype=np.uint32)
            line_colors = (np.asarray(line_set.colors, dtype=np.float32)
                           if line_set.has_colors() else None)

            # Expand lines to consecutive point pairs
            idx = lines.ravel()
            layer.line_points = pts[idx]
            layer.points = pts
            layer.point_count = len(pts)

            if line_colors is not None and len(line_colors) > 0:
                # Each line gets its color duplicated for both endpoints
                layer.colors = np.repeat(line_colors, 2, axis=0).astype(
                    np.float32)
            else:
                pc = o3d.io.read_point_cloud(path)
                if pc.has_colors() and len(pc.colors) == len(pts):
                    vc = np.asarray(pc.colors, dtype=np.float32)
                    layer.colors = vc[idx]
                else:
                    layer.colors = np.full(
                        (len(layer.line_points), 3), [0.0, 1.0, 0.5],
                        dtype=np.float32)

            layer.layer_type = "wireframe"
            layer.loaded = True
            return
    except Exception:
        pass

    # --- Try mesh ---
    # Only probe for mesh if type is "mesh" (OBJ) or file was NOT already
    # attempted by the fast PLY parser (which returns False for face PLYs,
    # so they arrive here with is_ply_or_obj=True).
    if layer.layer_type == "mesh" or is_ply_or_obj:
        # Suppress O3D's C++ stderr output (prints "appears to be a
        # PointCloud" at Error level even with Error verbosity set)
        import io, contextlib, sys as _sys
        _stderr = _sys.stderr
        _sys.stderr = io.StringIO()
        try:
            mesh = o3d.io.read_triangle_mesh(path)
        finally:
            _sys.stderr = _stderr
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            layer.points = np.asarray(mesh.vertices, dtype=np.float32)
            layer.triangles = np.asarray(mesh.triangles, dtype=np.uint32)
            if mesh.has_vertex_colors():
                layer.colors = np.asarray(mesh.vertex_colors,
                                          dtype=np.float32)
            if mesh.has_vertex_normals():
                layer.normals = np.asarray(mesh.vertex_normals,
                                           dtype=np.float32)
            layer.layer_type = "mesh"
            layer.tri_count = len(layer.triangles)
            layer.point_count = len(layer.points)
            layer.loaded = True
            return

    # --- Try point cloud ---
    pcd = o3d.io.read_point_cloud(path)
    if pcd.has_points() and len(pcd.points) > 0:
        layer.points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            layer.colors = np.asarray(pcd.colors, dtype=np.float32)
        if pcd.has_normals():
            layer.normals = np.asarray(pcd.normals, dtype=np.float32)
        layer.point_count = len(layer.points)
        layer.loaded = True
        return

    raise ValueError(f"No valid geometry data in {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_geometry(path: str, layer: 'LayerData'):
    """Load PLY/OBJ/STL as point cloud, mesh, or wireframe into a LayerData.

    For binary little-endian PLY point clouds, uses a fast numpy-based parser
    (~3 GB/s, 30× faster than Open3D).  Falls back to Open3D for everything
    else (ASCII PLY, meshes, OBJ, STL, wireframes).
    """
    if path.lower().endswith('.ply'):
        try:
            if _try_fast_ply(path, layer):
                return
        except Exception:
            pass  # fall through to Open3D

    _load_with_open3d(path, layer)
