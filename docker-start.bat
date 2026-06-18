@echo off
REM =========================================
REM AI Sales Copilot - Docker Start Script
REM =========================================

echo.
echo ========================================
echo   AI Sales Copilot - Docker Mode
echo ========================================
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop chua chay!
    echo.
    echo Vui long:
    echo 1. Mo Docker Desktop
    echo 2. Doi Docker Desktop khoi dong xong
    echo 3. Chay lai script nay
    echo.
    pause
    exit /b 1
)

echo [OK] Docker Desktop dang chay!
echo.

REM Check if .env exists
if not exist .env (
    echo [ERROR] File .env khong ton tai!
    echo Vui long tao file .env voi cac API keys can thiet.
    pause
    exit /b 1
)

echo [1/3] Checking Docker images...
docker-compose images >nul 2>&1

REM Check if images exist
docker images | findstr sales-copilot >nul
if errorlevel 1 (
    echo Images chua co. Building images (lan dau se mat 5-10 phut)...
    echo.
    docker-compose build
    if errorlevel 1 (
        echo [ERROR] Build failed!
        pause
        exit /b 1
    )
) else (
    echo Images da co. Skip build.
)

echo.
echo [2/3] Starting containers...
docker-compose up -d

if errorlevel 1 (
    echo [ERROR] Khong the start containers!
    echo Xem logs: docker-compose logs
    pause
    exit /b 1
)

echo.
echo [3/3] Checking container status...
timeout /t 3 /nobreak >nul
docker-compose ps

echo.
echo ========================================
echo   System started successfully!
echo ========================================
echo.
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000
echo.
echo Commands:
echo   docker-compose logs -f    : Xem logs
echo   docker-compose ps         : Xem status
echo   docker-compose down       : Stop containers
echo   docker-stop.bat           : Stop va cleanup
echo.
echo Press any key to open browser...
pause >nul

start http://localhost:5173

echo.
echo Containers dang chay o che do background.
echo De stop: chay docker-stop.bat
echo.
