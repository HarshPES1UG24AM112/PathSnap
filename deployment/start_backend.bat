@echo off
REM Laptop 1: start the PathSnap API on all network interfaces (port 8000)
cd /d %~dp0\..
if "%PATHSNAP_EVAL_DIR%"=="" set PATHSNAP_EVAL_DIR=%CD%\data\raw\student_test
echo Evaluation trips: %PATHSNAP_EVAL_DIR%
echo Laptop 1 IP addresses:
ipconfig | findstr /R /C:"IPv4"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
