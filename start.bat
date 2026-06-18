@echo off
REM =========================================
REM AI Sales Copilot - Windows Startup Script
REM =========================================

echo.
echo ========================================
echo   AI Sales Copilot - Starting...
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo [ERROR] File .env khong ton tai!
    echo Vui long tao file .env voi cac API keys can thiet.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python chua duoc cai dat!
    echo Vui long cai dat Python 3.10+ tu https://www.python.org
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js chua duoc cai dat!
    echo Vui long cai dat Node.js tu https://nodejs.org
    pause
    exit /b 1
)

echo [1/4] Checking Python dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing Python packages...
    pip install -r requirements.txt
)

echo [2/4] Checking Node.js dependencies...
if not exist frontend\node_modules (
    echo Installing Node.js packages...
    cd frontend
    call npm install
    cd ..
)

echo [3/4] Starting Backend Server...
start "AI Sales Copilot - Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

echo [4/4] Starting Frontend Server...
start "AI Sales Copilot - Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   System started successfully!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to open browser...
pause >nul

start http://localhost:5173

echo.
echo To stop the application:
echo - Close both terminal windows
echo - Or run: stop.bat
echo.
