#!/usr/bin/env bash
# Kill by recorded pid. See det-stop.sh for why this does not pattern-match.
R="$(cd "$(dirname "$0")" && pwd)"
f="$R/api.pid"
[ -f "$f" ] || exit 0
pid=$(cat "$f" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
fi
rm -f "$f"
exit 0
