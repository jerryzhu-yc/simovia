@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Simovia 桌面 GUI 快速启动（Windows 双击入口）
rem Python 选择优先级：SIMOVIA_PYTHON 环境变量 > 仓库内 .venv > PATH 里的 python > py 启动器
rem 解释器路径可能含空格（C:\Program Files\...）或中文，调用时统一给变量加双引号；
rem 额外参数（如 py 启动器的 -3）单独放 PYARGS，避免把 "py -3" 整体当成一个路径。

set "PYEXE=%SIMOVIA_PYTHON%"
set "PYARGS="
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=%CD%\.venv\Scripts\python.exe"
if not defined PYEXE (
    where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    where py >nul 2>nul && (set "PYEXE=py" & set "PYARGS=-3")
)
if not defined PYEXE (
    echo [错误] 未找到 Python。请安装 Python 3.11+ 并勾选 "Add python.exe to PATH"，
    echo        或设置环境变量 SIMOVIA_PYTHON 指向 python.exe 的完整路径。
    pause
    exit /b 1
)

echo 用 [%PYEXE% %PYARGS%] 启动 Simovia GUI ...
"%PYEXE%" %PYARGS% -m simovia gui
if errorlevel 1 (
    echo.
    echo [提示] GUI 异常退出。若上方报 ModuleNotFoundError，请先安装依赖：
    echo        pip install -r requirements.txt
    pause
)
