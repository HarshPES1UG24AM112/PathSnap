#!/usr/bin/env bash
# Laptop 1: start the PathSnap API on all network interfaces (port 8000)
cd "$(dirname "$0")/.."
export PATHSNAP_EVAL_DIR="${PATHSNAP_EVAL_DIR:-$PWD/data/raw/student_test}"
echo "Evaluation trips: $PATHSNAP_EVAL_DIR"
(hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null) | sed 's/^/Laptop 1 IP: /'
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
