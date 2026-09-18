@echo off
rem Local AI Studio — cift tikla: backend + UI acilir, tarayici gelir. Durdurmak: start-studio.bat stop
setlocal
cd /d "%~dp0"
if /i "%~1"=="stop" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-studio.ps1" -Stop
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-studio.ps1" %*
)
if errorlevel 1 pause
