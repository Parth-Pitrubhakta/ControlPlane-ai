#!/usr/bin/env bash
# Kill by recorded pid only. Never pattern-match on the process table here: a
# -f pattern also matches whatever shell happens to have this script's text on
# its own command line, which is a memorable way to kill your own session.
R="$(cd "$(dirname "$0")" && pwd)"
for f in "$R"/det-*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f" 2>/dev/null) || continue
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done
exit 0
