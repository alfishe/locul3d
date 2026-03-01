"""Entry point for `python -m locul3d`.

Launches the Viewer by default.
Use `python -m locul3d editor` to launch the annotation editor.
"""

import sys


def main():
    """Dispatch to viewer or editor based on CLI args."""
    if len(sys.argv) > 1 and sys.argv[1] == "editor":
        sys.argv.pop(1)
        from locul3d.editor.main import main as editor_main
        editor_main()
    else:
        from locul3d.viewer.main import main as viewer_main
        viewer_main()


if __name__ == "__main__":
    main()
