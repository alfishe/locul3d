@echo off
REM Locul3D — Install dependencies
REM Run once before first use, or after pulling updates.

echo === Locul3D Install ===
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is required but not found.
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo Python: %%v
echo.

REM Install core dependencies
echo Installing dependencies from requirements.txt...
pip install -r "%~dp0requirements.txt"
echo.

REM Install package in editable mode
echo Installing locul3d package...
pip install -e "%~dp0."
echo.

REM Install optional E57 panorama dependencies
echo Installing optional E57/panorama dependencies...
pip install pye57 Pillow 2>nul
echo.

REM Verify
echo === Verifying ===
python -c "import sys; ok=True; exec(\"\"\"
def check(name, pkg):
    global ok
    try:
        __import__(pkg)
        print(f'  + {name}')
    except Exception as e:
        print(f'  - {name}: {e}')
        ok = False
check('PySide6',  'PySide6')
check('PyOpenGL', 'OpenGL')
check('numpy',    'numpy')
check('open3d',   'open3d')
check('scipy',    'scipy')
check('pye57',    'pye57')
check('libe57',   'libe57')
check('Pillow',   'PIL')
print()
print('All dependencies installed.' if ok else 'Some dependencies missing.')
\"\"\")"

echo.
echo Done. Launch with:  start.bat  or  start.bat editor
pause
