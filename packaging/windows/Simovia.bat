@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Simovia

if not exist ".venv\Scripts\python.exe" (
    echo Simovia 还没有配置运行环境，请先双击本目录里的 install-windows.bat
    echo.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" -m simovia gui
if errorlevel 1 (
    echo.
    echo [提示] Simovia 异常退出。请把上面的报错反馈给维护者。
    pause
)