@echo off
REM ============================================================
REM  Start the two demos (web UI) on Windows.
REM
REM  Usage: double-click this file, or run:
REM      start-demo.cmd [port]        (default port: 8000)
REM
REM  ASCII only on purpose: cmd.exe reads .cmd files using the console
REM  code page, so non-ASCII text here would be garbled on zh-CN Windows.
REM ============================================================
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=8000

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set DEMO_PORT=%PORT%

echo Starting demo server on http://127.0.0.1:%PORT%
echo Press Ctrl+C to stop.

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 serving.py
) else (
  python serving.py
)

echo.
echo Server exited.
pause
