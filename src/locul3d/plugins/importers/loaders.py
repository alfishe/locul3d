"""Geometry loaders — Open3D-based loading for PLY/OBJ/STL files."""

import numpy as np
import open3d as o3d

# Mute Open3D warnings (e.g. "appears to be a PointCloud", "number of edges <= 0")
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def load_geometry(path: str, layer):
    """Load PLY/OBJ/STL as point cloud, mesh, or wireframe into a LayerData.

    Tries loading in order: line set → mesh → point cloud.
    Populates layer.points, .colors, .normals, .triangles, .line_points etc.
    """
    # --- Try line set (wireframe) first ---
    try:
        line_set = o3d.io.read_line_set(path)
        if (line_set.has_points() and len(line_set.points) > 0
                and len(line_set.lines) > 0):
            pts = np.asarray(line_set.points, dtype=np.float32)
            lines = np.asarray(line_set.lines, dtype=np.uint32)
            line_colors = (np.asarray(line_set.colors, dtype=np.float32)
                           if line_set.has_colors() else None)

            # Build line_points: each line = 2 consecutive points
            line_pts = []
            for idx0, idx1 in lines:
                line_pts.append(pts[idx0])
                line_pts.append(pts[idx1])
            layer.line_points = np.array(line_pts, dtype=np.float32)
            layer.points = pts
            layer.point_count = len(pts)

            if line_colors is not None and len(line_colors) > 0:
                expanded = []
                for i in range(len(lines)):
                    c = line_colors[i] if i < len(line_colors) else [1.0, 0.0, 0.0]
                    expanded.append(c)
                    expanded.append(c)
                layer.colors = np.array(expanded, dtype=np.float32)
            else:
                pc = o3d.io.read_point_cloud(path)
                if pc.has_colors() and len(pc.colors) == len(pts):
                    vc = np.asarray(pc.colors, dtype=np.float32)
                    expanded = []
                    for idx0, idx1 in lines:
                        expanded.append(vc[idx0])
                        expanded.append(vc[idx1])
                    layer.colors = np.array(expanded, dtype=np.float32)
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
    if layer.layer_type == "mesh":
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            layer.points = np.asarray(mesh.vertices, dtype=np.float32)
            layer.triangles = np.asarray(mesh.triangles, dtype=np.uint32)
            if mesh.has_vertex_colors():
                layer.colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if mesh.has_vertex_normals():
                layer.normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
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
