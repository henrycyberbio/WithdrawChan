@echo off
setlocal
cd /d "%~dp0"
where node.exe >nul 2>nul
if errorlevel 1 (
  echo Node.js 22.18.0 or newer with npm is required. Node.js 24 is recommended.
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm.cmd was not found. Install Node.js with npm and add it to PATH.
  pause
  exit /b 1
)
call npm.cmd run setup
if errorlevel 1 goto failed
echo Preview defaults to http://127.0.0.1:4173/ unless PORT is set.
echo Use the address printed by the server. Keep this window open.
echo Press Ctrl+C to stop. Source edits require rebuilding; no hot reload.
call npm.cmd run dev
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup, engine build, site build or preview failed. See the error above.
pause
exit /b 1
