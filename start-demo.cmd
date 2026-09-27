@echo off
REM 一键启动 demo（Windows）
REM 用法：双击本文件，或在命令行执行 start-demo.cmd [端口]
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=8000

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set DEMO_PORT=%PORT%

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 serving.py
) else (
  python serving.py
)

echo.
echo 服务已退出。
pause
