#!/usr/bin/env bash
# deploy/healthcheck.sh — Health check script for the verification workflow
# Returns 0 (healthy/degraded) or 1 (unhealthy/unreachable)
set -euo pipefail

HOST="${HEALTH_HOST:-localhost}"
PORT="${HEALTH_PORT:-8000}"
TIMEOUT="${HEALTH_TIMEOUT:-5}"
URL="http://${HOST}:${PORT}/health"

# Attempt the health check
response=$(curl --silent --max-time "$TIMEOUT" --write-out "\n%{http_code}" "$URL" 2>/dev/null) || {
  echo "UNHEALTHY: could not reach $URL"
  exit 1
}

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [[ "$http_code" -eq 200 ]]; then
  # Parse status field from JSON body (no jq dependency)
  status=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
  case "$status" in
    healthy)
      echo "HEALTHY: $URL returned status=healthy"
      exit 0
      ;;
    degraded)
      echo "DEGRADED: $URL returned status=degraded (still serving)"
      exit 0
      ;;
    *)
      echo "UNHEALTHY: $URL returned status=$status"
      exit 1
      ;;
  esac
elif [[ "$http_code" -eq 503 ]]; then
  echo "UNHEALTHY: $URL returned HTTP 503"
  exit 1
else
  echo "UNHEALTHY: $URL returned HTTP $http_code"
  exit 1
fi
