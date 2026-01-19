@echo off
echo Starting Clinical View Pro - Polyp Detection System
echo ================================================
echo.

REM Start Backend API Server
echo [1/2] Starting Backend API Server on port 8000...
start "Polyp Detection API" cmd /k "python yolo_api_server.py"
timeout /t 3 /nobreak >nul

REM Start Frontend Development Server
echo [2/2] Starting Frontend Development Server on port 5173...
start "Clinical View Frontend" cmd /k "npm run dev"

echo.
echo ================================================
echo Both servers are starting...
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo.
echo Close this window or press Ctrl+C to stop the script.
echo The servers will continue running in separate windows.
echo ================================================
pause
