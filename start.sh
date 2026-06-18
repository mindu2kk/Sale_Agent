#!/bin/bash
# =========================================
# AI Sales Copilot - Linux/Mac Startup Script
# =========================================

set -e

echo ""
echo "========================================"
echo "  AI Sales Copilot - Starting..."
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "[ERROR] File .env không tồn tại!"
    echo "Vui lòng tạo file .env với các API keys cần thiết."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python chưa được cài đặt!"
    echo "Vui lòng cài đặt Python 3.10+ từ https://www.python.org"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js chưa được cài đặt!"
    echo "Vui lòng cài đặt Node.js từ https://nodejs.org"
    exit 1
fi

echo "[1/4] Checking Python dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing Python packages..."
    pip3 install -r requirements.txt
fi

echo "[2/4] Checking Node.js dependencies..."
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing Node.js packages..."
    cd frontend
    npm install
    cd ..
fi

echo "[3/4] Starting Backend Server..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

echo "[4/4] Starting Frontend Server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
cd ..

# Save PIDs for stop script
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo ""
echo "========================================"
echo "  System started successfully!"
echo "========================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "To stop: ./stop.sh or Ctrl+C"
echo ""

# Wait for user interrupt
wait
