"""Geometry loading utilities."""

import numpy as np


def load_geometry(path: str, layer: 'LayerData'):
    """Load geometry from file into a LayerData object."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required. Install: pip install open3d")

    # First, try loading as line set (check for edges)
    try:
        line_set = o3d.io.read_line_set(path)
        if line_set.has_points() and len(line_set.points) > 0 and len(line_set.lines) > 0:
            pts = np.asarray(line_set.points, dtype=np.float32)
            lines = np.asarray(line_set.lines, dtype=np.uint32)
            line_colors = np.asarray(line_set.colors, dtype=np.float32) if line_set.has_colors() else None
            
            # Build line_points array (each line becomes 2 consecutive points)
            line_pts = []
            for idx0, idx1 in lines:
                line_pts.append(pts[idx0])
                line_pts.append(pts[idx1])
            
            layer.line_points = np.array(line_pts, dtype=np.float32)
            layer.points = pts  # Store original points for bounding box
            layer.point_count = len(pts)
            
            # Store line colors: per-edge from LineSet, or per-vertex from PLY, or default
            if line_colors is not None and len(line_colors) > 0:
                # Per-edge colors → duplicate for both endpoints
                expanded_colors = []
                for i in range(len(lines)):
                    c = line_colors[i] if i < len(line_colors) else [1.0, 0.0, 0.0]
                    expanded_colors.append(c)
                    expanded_colors.append(c)
                layer.colors = np.array(expanded_colors, dtype=np.float32)
            else:
                # Try per-vertex colors from the point cloud
                pc = o3d.io.read_point_cloud(path)
                if pc.has_colors() and len(pc.colors) == len(pts):
                    vc = np.asarray(pc.colors, dtype=np.float32)
                    expanded_colors = []
                    for idx0, idx1 in lines:
                        expanded_colors.append(vc[idx0])
                        expanded_colors.append(vc[idx1])
                    layer.colors = np.array(expanded_colors, dtype=np.float32)
                else:
                    # Default green for wireframes
                    layer.colors = np.full((len(layer.line_points), 3), [0.0, 1.0, 0.5], dtype=np.float32)
            
            # Override layer type to wireframe
            layer.layer_type = "wireframe"
            return
    except Exception:
        pass  # Not a line set, try other formats
    
    # Try loading as mesh
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
            return
    
    # Try loading as point cloud
    pcd = o3d.io.read_point_cloud(path)
    if pcd.has_points() and len(pcd.points) > 0:
        layer.points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            layer.colors = np.asarray(pcd.colors, dtype=np.float32)
        if pcd.has_normals():
            layer.normals = np.asarray(pcd.normals, dtype=np.float32)
        layer.point_count = len(layer.points)
        return
    
    # Nothing worked
    raise ValueError(f"No valid geometry data in {path}")
