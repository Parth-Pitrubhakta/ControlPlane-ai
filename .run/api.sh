#!/usr/bin/env bash
set -a; . "$(dirname "$0")/../.env.native"; set +a
cd "$(dirname "$0")/.."
echo $$ > "$PWD/.run/api.pid"
PYTHONPATH="$PWD" exec /DATA2/home/parth/.conda/envs/cp-api/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 --port "${API_PORT:-8080}"
