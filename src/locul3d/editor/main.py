"""Locul3D Editor entry point."""

import sys
import argparse
from pathlib import Path


def main():
    """Launch the Locul3D annotation editor."""
    from PySide6.QtWidgets import QApplication
    from .window import EditorWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Locul3D Editor")
    app.setStyle("Fusion")

    parser = argparse.ArgumentParser(
        prog="locul3d editor",
        description="Locul3D — annotation editor",
    )
    parser.add_argument("files", nargs="*", help="Files to open")
    parser.add_argument("-a", "--annotations", dest="annotations",
                        help="Path to annotations YAML")
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
    # Remote API options
    parser.add_argument("--api-port", type=int, default=8350,
                        help="Remote API server port (default: 8350)")
    parser.add_argument("--no-api", action="store_true",
                        help="Disable the Remote API server")
    parser.add_argument("--api-host", type=str, default="127.0.0.1",
                        help="Remote API bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    correction = {
        'rotate_x': args.rotate_x,
        'rotate_y': args.rotate_y,
        'rotate_z': args.rotate_z,
        'shift_x': args.shift_x,
        'shift_y': args.shift_y,
        'shift_z': args.shift_z,
    }
    has_correction = any(v != 0.0 for v in correction.values())

    window = EditorWindow(
        files=args.files or None,
        annotations_path=args.annotations,
        correction_angles=correction if has_correction else None,
    )
    window.show()

    # Start Remote API server (unless disabled)
    if not args.no_api:
        try:
            from locul3d.remote import start_server
            window._remote_server = start_server(
                window=window,
                port=args.api_port,
                host=args.api_host,
            )
        except ImportError:
            pass  # remote package not installed

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

