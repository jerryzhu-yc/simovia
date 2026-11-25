@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Simovia 安装

echo ============================================
echo   Simovia 一键配置（Windows）
echo   安装目录：%CD%
echo ============================================
echo.

rem ---- 1. 查找 64 位 Python 3.11+ ----
set "PYEXE="
set "PYARGS="

where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.13 3.12 3.11) do (
        if not defined PYEXE (
            py -%%V -c "import sys" >nul 2>nul
            if not errorlevel 1 (
                set "PYEXE=py"
                set "PYARGS=-%%V"
            )
        )
    )
    if not defined PYEXE (
        py -3 -c "import sys; raise SystemExit(sys.hexversion < 0x030b0000)" >nul 2>nul
        if not errorlevel 1 (
            set "PYEXE=py"
            set "PYARGS=-3"
        )
    )
)

if not defined PYEXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYEXE (
            echo %%P | findstr /i "WindowsApps" >nul
            if errorlevel 1 (
                "%%P" -c "import sys; raise SystemExit(sys.hexversion < 0x030b0000)" >nul 2>nul
                if not errorlevel 1 set "PYEXE=%%P"
            )
        )
    )
)

if not defined PYEXE (
    echo [错误] 没有找到 Python 3.11 或更高版本。
    echo.
    echo 请从 https://www.python.org/downloads/windows/ 安装 64 位 Python，
    echo 安装时勾选 "Add python.exe to PATH"，然后重新双击本安装程序。
    echo.
    pause
    exit /b 1
)

echo [1/4] Python：%PYEXE% %PYARGS%
"%PYEXE%" %PYARGS% -c "import sys; print('        版本', sys.version.split()[0], '64位' if sys.maxsize > 2**32 else '32位')"

rem ---- 2. 虚拟环境 ----
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo [2/4] 现有 .venv 不可用，正在重建 ...
        rmdir /s /q ".venv"
    )
)
if exist ".venv\Scripts\python.exe" (
    echo [2/4] 复用已有虚拟环境 .venv
) else (
    echo [2/4] 创建虚拟环境 .venv ...
    "%PYEXE%" %PYARGS% -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认 Python 安装完整（需要 venv 模块）。
        pause
        exit /b 1
    )
)
set "VPY=%CD%\.venv\Scripts\python.exe"

rem ---- 3. 依赖 ----
echo [3/4] 安装依赖（首次需要联网，约几分钟）...
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check
if errorlevel 1 goto :deps_failed
"%VPY%" -m pip install --disable-pip-version-check -r requirements-core.txt -r requirements-gui.txt -r demo\fruit_sorting\requirements.txt
if errorlevel 1 goto :deps_failed
"%VPY%" -m pip check
if errorlevel 1 echo [提示] pip check 有告警，通常不影响启动。

rem ---- 4. 自检 + 快捷方式 ----
echo.
echo [4/4] 环境自检 ...
"%VPY%" -m simovia doctor
if errorlevel 1 (
    echo.
    echo [提示] 诊断有未通过项。若只是显卡/离屏渲染告警，可先启动试试；
    echo        缺少必需依赖时请把上面的输出反馈给维护者。
)

echo.
echo 创建桌面与开始菜单快捷方式 ...
set "SIMOVIA_ROOT=%CD%"
set "SIMOVIA_ICON=%CD%\simovia.ico"
set "SIMOVIA_PYW=%CD%\.venv\Scripts\pythonw.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = $env:SIMOVIA_ROOT; $target = $env:SIMOVIA_PYW; $icon = $env:SIMOVIA_ICON; $dirs = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')); foreach ($d in $dirs) { if ([string]::IsNullOrEmpty($d)) { continue }; $lnk = Join-Path $d 'Simovia.lnk'; $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut($lnk); $s.TargetPath = $target; $s.Arguments = '-m simovia gui'; $s.WorkingDirectory = $root; $s.IconLocation = $icon; $s.Description = 'Simovia 机器人仿真工作台'; $s.Save(); Write-Host ('        ' + $lnk) }"
if errorlevel 1 (
    echo [提示] 快捷方式创建失败（PowerShell 不可用或被策略拦截）。
    echo        可以直接双击本目录里的 Simovia.bat 启动。
)

echo.
echo 配置完成！双击桌面或开始菜单里的 Simovia 图标即可启动。
echo （排错时可以双击本目录里的 Simovia.bat，会保留控制台输出）
echo.
set /p STARTNOW=现在启动 Simovia 吗？[Y/n] 
if /i "%STARTNOW%"=="n" goto :finished
start "" "%SIMOVIA_PYW%" -m simovia gui
echo 已启动，本窗口可以关闭。
timeout /t 3 >nul
exit /b 0

:finished
echo.
echo 已跳过启动。随时双击桌面上的 Simovia 图标即可。
echo.
pause
exit /b 0

:deps_failed
echo.
echo [错误] 依赖安装失败。常见原因：
echo   1. 网络无法访问 pypi.org。可配置镜像后重试，例如：
echo      "%VPY%" -m pip install -r requirements-core.txt -r requirements-gui.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo   2. Python 版本过低或不是 64 位。
echo.
pause
exit /b 1
