@echo off
REM ============================================================
REM  Push this demo folder to GitHub (Windows one-liner).
REM
REM  Usage (inside the demo folder):
REM      push-to-github.cmd https://github.com/<user>/<repo>.git
REM
REM  Notes:
REM   1) Create an EMPTY repository on GitHub first (no README), otherwise
REM      push fails with "remote: Repository not found".
REM   2) If this machine cannot resolve github.com, configure a proxy first:
REM        git config http.proxy http://127.0.0.1:7897
REM      The script also passes -c url.https://github.com/.insteadOf= to skip
REM      the global ghfast.top mirror rewrite, which is read-only for git
REM      (push through it returns 401). It does NOT modify your global config;
REM      to change that permanently, edit %USERPROFILE%\.gitconfig instead.
REM   3) Credentials: log in once with Git Credential Manager
REM        "C:\Program Files\Git\mingw64\bin\git-credential-manager.exe" github login
REM   4) ASCII only on purpose: cmd.exe reads .cmd files using the console
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
