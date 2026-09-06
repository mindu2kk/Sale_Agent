# AI Sales Copilot - Makefile
# Quick commands for common tasks

.PHONY: help start stop install test test-full clean docker-up docker-down docker-logs

# Default target
help:
	@echo ""
	@echo "AI Sales Copilot - Available Commands:"
	@echo "======================================"
	@echo ""
	@echo "Development:"
	@echo "  make start        - Start backend + frontend"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo ""
	@echo "Installation:"
	@echo "  make install      - Install all dependencies"
	@echo "  make install-py   - Install Python dependencies"
	@echo "  make install-js   - Install Node.js dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run push-ready smoke/regression tests"
	@echo "  make test-full    - Run full pytest suite"
	@echo "  make test-unit    - Run unit tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    - Start with Docker Compose"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make docker-logs  - View Docker logs"
	@echo "  make docker-build - Rebuild Docker images"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache files"
	@echo "  make clean-all    - Remove all generated files"
	@echo ""

# Development commands
start:
	@echo "Starting AI Sales Copilot..."
	./start.sh

stop:
	@echo "Stopping AI Sales Copilot..."
	./stop.sh

restart: stop start

# Installation
install: install-py install-js

install-py:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

install-js:
	@echo "Installing Node.js dependencies..."
	cd frontend && npm install

# Testing
test:
	@echo "Running push-ready smoke/regression tests..."
	python -m pytest tests/test_harness_runtime.py tests/test_agent_verifier.py tests/test_query_frame_display_specs.py tests/test_product_reference_resolution.py tests/test_api_contract_runtime.py tests/test_harness_preflight.py tests/test_harness_postflight.py tests/agent tests/verification/test_config_loader.py tests/test_unit.py -q

test-full:
	@echo "Running full pytest suite..."
	python -m pytest -q

test-unit:
	@echo "Running unit tests..."
	pytest tests/test_unit.py

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=. --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Docker commands
docker-up:
	@echo "Starting with Docker Compose..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

docker-build:
	@echo "Rebuilding Docker images..."
	docker-compose build --no-cache

docker-restart:
	@echo "Restarting Docker containers..."
	docker-compose restart

# Cleanup
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "Cache cleaned!"

clean-all: clean
	@echo "Removing all generated files..."
	rm -rf chroma_db/ 2>/dev/null || true
	rm -f chat.db 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true
	rm -rf frontend/node_modules/ 2>/dev/null || true
	rm -rf frontend/dist/ 2>/dev/null || true
	@echo "All generated files removed!"

# Database
db-reset:
	@echo "Resetting database..."
	rm -f chat.db
	rm -rf chroma_db/
	@echo "Database reset complete!"

# Health check
health:
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@curl -s http://localhost:8000/metrics | python -m json.tool

# Logs
logs-backend:
	@echo "Backend logs:"
	@tail -f logs/backend.log 2>/dev/null || echo "No backend logs found"

logs-frontend:
	@echo "Frontend logs:"
	@tail -f logs/frontend.log 2>/dev/null || echo "No frontend logs found"
