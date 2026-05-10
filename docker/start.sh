#!/usr/bin/env bash
# deploy/start.sh — Startup script with environment validation
# Usage: ./deploy/start.sh [--dev]
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Parse args ────────────────────────────────────────────────────────────────
DEV_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--dev" ]] && DEV_MODE=true
done

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  info "Loading environment from .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ── Required environment variables ───────────────────────────────────────────
REQUIRED_VARS=(
  "LLAMA_CLOUD_API_KEY"
)

OPTIONAL_VARS=(
  "OPENAI_API_KEY"
  "ANTHROPIC_API_KEY"
  "GOOGLE_API_KEY"
  "TAVILY_API_KEY"
)

missing=0
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    error "Required environment variable '$var' is not set."
    missing=$((missing + 1))
  fi
done

for var in "${OPTIONAL_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    warn "Optional environment variable '$var' is not set."
  fi
done

if [[ $missing -gt 0 ]]; then
  error "Aborting: $missing required variable(s) missing."
  exit 1
fi

# ── Runtime directories ───────────────────────────────────────────────────────
for dir in logs/errors logs/metrics logs/workflow data; do
  mkdir -p "$dir"
done

# ── Settings ──────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

info "Starting Verification Workflow service"
info "  Host:    $HOST:$PORT"
info "  Workers: $WORKERS"
info "  Log:     $LOG_LEVEL"
info "  Mode:    $( $DEV_MODE && echo development || echo production )"

# ── Launch ────────────────────────────────────────────────────────────────────
if $DEV_MODE; then
  exec uvicorn verification.api:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --loop asyncio \
    --log-level "$LOG_LEVEL"
else
  exec uvicorn verification.api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --loop asyncio \
    --log-level "$LOG_LEVEL"
fi
