@echo off
REM =========================================
REM AI Sales Copilot - Stop Script
REM =========================================

echo.
echo ========================================
echo   Stopping AI Sales Copilot...
echo ========================================
echo.

echo [1/2] Stopping Backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/2] Stopping Frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ========================================
echo   Stopped successfully!
echo ========================================
echo.
pause
