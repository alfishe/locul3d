"""Viewer entry point — launches the Locul3D viewer."""

import sys
import os
import argparse

# Suppress Open3D warning spam
try:
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
except ImportError:
    pass


def main():
    """Launch the Locul3D viewer."""
    parser = argparse.ArgumentParser(
        prog="locul3d",
        description="Locul3D — 3D point cloud and mesh viewer",
    )
    parser.add_argument("files", nargs="*", help="Files or folders to open")
    parser.add_argument("--rotate-x", type=float, default=0.0,
                        help="Scene correction rotation around X axis (degrees)")
    parser.add_argument("--rotate-y", type=float, default=0.0,
                        help="Scene correction rotation around Y axis (degrees)")
    parser.add_argument("--rotate-z", type=float, default=0.0,
                        help="Scene correction rotation around Z axis (degrees)")
    parser.add_argument("--shift-x", type=float, default=0.0,
                        help="Scene correction shift along X axis (scene units)")
    parser.add_argument("--shift-y", type=float, default=0.0,
                        help="Scene correction shift along Y axis (scene units)")
    parser.add_argument("--shift-z", type=float, default=0.0,
                        help="Scene correction shift along Z axis (scene units)")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication
    from .window import ViewerWindow

    app = QApplication.instance() or QApplication(sys.argv)

    correction = {
        'rotate_x': args.rotate_x,
        'rotate_y': args.rotate_y,
        'rotate_z': args.rotate_z,
        'shift_x': args.shift_x,
        'shift_y': args.shift_y,
        'shift_z': args.shift_z,
    }
    # Only pass correction if any value is non-zero
    has_correction = any(v != 0.0 for v in correction.values())

    window = ViewerWindow(
        files=args.files or None,
        correction_angles=correction if has_correction else None,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

