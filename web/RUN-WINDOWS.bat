@echo off
setlocal
cd /d "%~dp0.."
call RUN-WINDOWS.bat
exit /b %errorlevel%
