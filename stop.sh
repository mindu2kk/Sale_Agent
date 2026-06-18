#!/bin/bash
# =========================================
# AI Sales Copilot - Stop Script
# =========================================

echo ""
echo "========================================"
echo "  Stopping AI Sales Copilot..."
echo "========================================"
echo ""

# Stop backend
if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    echo "[1/2] Stopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    rm .backend.pid
fi

# Stop frontend
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    echo "[2/2] Stopping Frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null || true
    rm .frontend.pid
fi

# Fallback: kill by port
echo "Cleaning up processes on ports 8000 and 5173..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

echo ""
echo "========================================"
echo "  Stopped successfully!"
echo "========================================"
echo ""
