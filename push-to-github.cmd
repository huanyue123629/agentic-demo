@echo off
REM ============================================================
REM  Push this demo folder to GitHub (Windows one-liner).
REM
REM  Usage (inside the demo folder):
REM      push-to-github.cmd https://github.com/<user>/<repo>.git
REM
REM  Notes:
REM   1) Create an EMPTY repository on GitHub first (no README),
REM      or pass a token URL: https://<TOKEN>@github.com/<user>/<repo>.git
REM   2) This machine has a global git mirror rule
REM      (url.https://ghfast.top/https://github.com/.insteadOf) which is
REM      not guaranteed to support push, so this script disables that
REM      rewrite for the push only. It does NOT modify your global config.
REM   3) ASCII only on purpose: cmd.exe reads .cmd files using the console
REM      code page, so non-ASCII text here would be garbled on zh-CN Windows.
REM ============================================================
setlocal

if "%~1"=="" (
  echo.
  echo [ERROR] Missing repository URL.
  echo Usage: push-to-github.cmd https://github.com/^<user^>/^<repo^>.git
  echo.
  pause
  exit /b 1
)

cd /d "%~dp0"

echo.
echo [1/4] Commits to be pushed:
git log --format="  %%h %%an ^<%%ae^> %%s"
echo.

echo [2/4] Setting remote origin ...
git remote remove origin >nul 2>nul
git remote add origin "%~1" || (echo [ERROR] failed to add remote & pause & exit /b 1)
git remote -v
echo.

echo [3/4] Pushing (mirror proxy bypassed for this command only) ...
git -c url.https://github.com/.insteadOf= push -u origin main
if errorlevel 1 (
  echo.
  echo [FAILED] Push did not succeed. Common causes:
  echo   - Invalid credentials: use https://^<TOKEN^>@github.com/^<user^>/^<repo^>.git
  echo   - Remote repo not empty: create an empty repo, or run with --force
  echo   - No network access to github.com
  echo.
  pause
  exit /b 1
)

echo.
echo [4/4] Done. Remote tracking:
git branch -vv
echo.
echo Repository: %~1
pause
