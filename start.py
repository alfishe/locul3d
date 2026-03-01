#!/usr/bin/env python3
"""Locul3D launcher.

Usage:
    python start.py                # launch viewer (default)
    python start.py editor         # launch editor
    python start.py scan.ply       # launch viewer with file
"""

import sys
import os

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Suppress Open3D warning spam (probing PLY as mesh/lineset before correct format)
try:
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
except ImportError:
    pass

if len(sys.argv) > 1 and sys.argv[1] == "editor":
    sys.argv.pop(1)
    from locul3d.editor.main import main
else:
    from locul3d.viewer.main import main

main()
