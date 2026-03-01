@echo off
REM Locul3D Quick Start (Windows)
REM Usage:
REM   start.bat                        # launch viewer (default)
REM   start.bat editor                 # launch editor
REM   start.bat scan.ply               # launch viewer with file

set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%

if "%1"=="editor" (
    shift
    python -m locul3d.editor.main %*
) else (
    python -m locul3d.viewer.main %*
)
