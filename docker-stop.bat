@echo off
REM =========================================
REM AI Sales Copilot - Docker Stop Script
REM =========================================

echo.
echo ========================================
echo   Stopping Docker Containers...
echo ========================================
echo.

docker ps >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop khong chay!
    pause
    exit /b 1
)

echo [1/2] Stopping containers...
docker-compose stop

echo.
echo [2/2] Removing containers...
docker-compose down

echo.
echo ========================================
echo   Stopped successfully!
echo ========================================
echo.
echo To start again: docker-start.bat
echo.
pause
