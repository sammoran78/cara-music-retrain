#!/bin/bash
# CARA Attribution Console Launcher for macOS
# Starts the FastAPI backend, Vite frontend, and opens the browser.
# Kills any previously running backend/frontend processes first.

set -euo pipefail

# --- Configuration ---
BACKEND_PORT=8000
FRONTEND_PORT=5173
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

# Determine project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DASHBOARD_VENV="$SCRIPT_DIR/.venv-dashboard"
if [ -x "$DASHBOARD_VENV/bin/python" ]; then
  BACKEND_PYTHON="$DASHBOARD_VENV/bin/python"
else
  BACKEND_PYTHON="$(command -v python3)"
fi

# --- Kill existing processes ---
echo "[CARA] Cleaning up any running backend/frontend processes..."

# Kill uvicorn processes serving our backend module
pkill -f "uvicorn gui.backend.main:app.*--port ${BACKEND_PORT}" 2>/dev/null || true

# Kill vite/npm dev servers on our frontend port
# We look for vite running on port 5173 or npm run dev in our frontend dir
pkill -f "vite.*--port ${FRONTEND_PORT}" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

release_port() {
  local port="$1"
  local label="$2"
  local pids

  pids="$(lsof -ti tcp:${port} 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return
  fi

  echo "[CARA] Releasing ${label} port ${port}..."
  kill $pids 2>/dev/null || true

  for _ in 1 2 3 4 5; do
    sleep 0.5
    pids="$(lsof -ti tcp:${port} 2>/dev/null || true)"
    if [ -z "$pids" ]; then
      return
    fi
  done

  echo "[CARA] Force-releasing ${label} port ${port}..."
  kill -9 $pids 2>/dev/null || true

  for _ in 1 2 3 4 5; do
    sleep 0.5
    pids="$(lsof -ti tcp:${port} 2>/dev/null || true)"
    if [ -z "$pids" ]; then
      return
    fi
  done

  echo "[CARA] ERROR: Could not release ${label} port ${port}. Still held by:"
  lsof -nP -iTCP:${port} -sTCP:LISTEN || true
  exit 1
}

# Also clear any process still bound to the expected ports. This catches uvicorn
# reload children whose command line no longer matches the parent pattern.
release_port "${BACKEND_PORT}" "backend"
release_port "${FRONTEND_PORT}" "frontend"

# Give processes a moment to release ports
sleep 1

# --- Log files ---
mkdir -p "$SCRIPT_DIR/logs"
BACKEND_LOG="$SCRIPT_DIR/logs/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/logs/frontend.log"

echo "[CARA] Logs will be written to:"
echo "  Backend: $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"

# --- Start Backend ---
echo "[CARA] Starting FastAPI backend on ${BACKEND_PORT}..."
echo "[CARA] Backend Python: ${BACKEND_PYTHON}"
nohup "$BACKEND_PYTHON" -m uvicorn gui.backend.main:app \
  --host 127.0.0.1 \
  --port ${BACKEND_PORT} \
  --reload \
  > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# --- Start Frontend ---
echo "[CARA] Starting Vite frontend on ${FRONTEND_PORT}..."
cd "$SCRIPT_DIR/gui/frontend"
nohup npm run dev \
  > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

cd "$SCRIPT_DIR"

# --- Wait for services ---
echo "[CARA] Waiting for services to be ready..."
MAX_WAIT=15
WAITED=0

backend_ready=false
frontend_ready=false

while [ $WAITED -lt $MAX_WAIT ]; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null && ! $backend_ready; then
    echo "[CARA] ERROR: Backend process exited before becoming ready. Check ${BACKEND_LOG}"
    break
  fi

  if ! kill -0 "$FRONTEND_PID" 2>/dev/null && ! $frontend_ready; then
    echo "[CARA] ERROR: Frontend process exited before becoming ready. Check ${FRONTEND_LOG}"
    break
  fi

  # Check backend
  if ! $backend_ready && curl -s "${BACKEND_URL}/docs" > /dev/null 2>&1; then
    backend_ready=true
    echo "[CARA] Backend ready at ${BACKEND_URL}"
  fi

  # Check frontend
  if ! $frontend_ready && curl -s "${FRONTEND_URL}" > /dev/null 2>&1; then
    frontend_ready=true
    echo "[CARA] Frontend ready at ${FRONTEND_URL}"
  fi

  if $backend_ready && $frontend_ready; then
    break
  fi

  sleep 1
  WAITED=$((WAITED + 1))
done

if ! $backend_ready; then
  echo "[CARA] WARNING: Backend did not become ready within ${MAX_WAIT}s. Check ${BACKEND_LOG}"
fi

if ! $frontend_ready; then
  echo "[CARA] WARNING: Frontend did not become ready within ${MAX_WAIT}s. Check ${FRONTEND_LOG}"
fi

# --- Open Browser ---
echo "[CARA] Opening browser..."
open "$FRONTEND_URL" 2>/dev/null || true

echo ""
echo "[CARA] Console is running!"
echo "  Frontend: ${FRONTEND_URL}"
echo "  Backend API: ${BACKEND_URL}"
echo "  Backend Docs: ${BACKEND_URL}/docs"
echo ""
echo "  Backend PID: ${BACKEND_PID}"
echo "  Frontend PID: ${FRONTEND_PID}"
echo ""
echo "[CARA] Press Ctrl+C to stop both services."
echo ""

# --- Cleanup on exit ---
cleanup() {
  echo ""
  echo "[CARA] Shutting down..."
  kill $FRONTEND_PID 2>/dev/null || true
  kill $BACKEND_PID 2>/dev/null || true
  wait $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID 2>/dev/null || true
  echo "[CARA] Done."
}
trap cleanup INT TERM EXIT

# Keep script running so Ctrl+C triggers cleanup
wait $BACKEND_PID
