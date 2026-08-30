#!/usr/bin/env bash
# Dashboard dev server. Node 20 comes from the cp-ui conda env; the system node
# is v12 and Vite will not start on it.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"     # resolve before cd, $0 is relative
echo $$ > "$R/ui.pid"
cd "$R/../ui"
export PATH=/DATA2/home/parth/.conda/envs/cp-ui/bin:$PATH
exec npm run dev
