@echo off
REM Laptop 2: build and serve the UI. Usage: start_frontend.bat 192.168.1.10
cd /d %~dp0\..\frontend
if "%1"=="" (set API=http://localhost:8000) else (set API=http://%1:8000)
echo VITE_API_BASE_URL=%API%> .env.local
call npm install
call npm run build
call npm run preview
