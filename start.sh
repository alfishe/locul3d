#!/bin/bash
# Locul3D Quick Start
# Usage:
#   ./start.sh                        # launch viewer (default)
#   ./start.sh editor                 # launch editor
#   ./start.sh scan.ply               # launch viewer with file

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

if [ "$1" = "editor" ]; then
    shift
    exec python3 -m locul3d.editor.main "$@"
else
    exec python3 -m locul3d.viewer.main "$@"
fi
