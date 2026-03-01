#!/bin/bash
# Locul3D — Install dependencies
# Run once before first use, or after pulling updates.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Locul3D Install ==="
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required but not found."
    exit 1
fi

echo "Python: $(python3 --version)"
echo

# Install core dependencies
echo "Installing dependencies from requirements.txt..."
pip3 install -r "${SCRIPT_DIR}/requirements.txt"
echo

# Install package in editable mode
echo "Installing locul3d package..."
pip3 install -e "${SCRIPT_DIR}"
echo

# Install optional E57 panorama dependencies
echo "Installing optional E57/panorama dependencies..."
pip3 install pye57 Pillow 2>/dev/null || echo "  (some optional packages unavailable — panorama/E57 may be limited)"
echo

# Verify
echo "=== Verifying ==="
python3 -c "
import sys
ok = True

def check(name, pkg):
    global ok
    try:
        __import__(pkg)
        print(f'  ✓ {name}')
    except Exception as e:
        print(f'  ✗ {name}: {e}')
        ok = False

check('PySide6',   'PySide6')
check('PyOpenGL',  'OpenGL')
check('numpy',     'numpy')
check('open3d',    'open3d')
check('scipy',     'scipy')
check('pye57',     'pye57')
check('libe57',    'libe57')
check('Pillow',    'PIL')

print()
if ok:
    print('All dependencies installed.')
else:
    print('Some dependencies missing — see above.')
"

echo
echo "Done. Launch with:  ./start.sh  or  ./start.sh editor"
