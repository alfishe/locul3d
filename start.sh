#!/bin/bash
# Locul3D Quick Start
# Usage:
#   ./start.sh                        # launch viewer (default)
#   ./start.sh editor                 # launch editor
#   ./start.sh scan.ply               # launch viewer with file

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Clear Python bytecode caches to ensure fresh code is always loaded
find "${SCRIPT_DIR}/src" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "${SCRIPT_DIR}/src" -name "*.pyc" -delete 2>/dev/null

if [ "$1" = "editor" ]; then
    shift
    exec python3 -m locul3d.editor.main "$@"
else
    exec python3 -m locul3d.viewer.main "$@"
fi
