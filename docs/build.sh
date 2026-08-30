#!/usr/bin/env bash
# Rebuild ControlPlane-ai.pdf from ARCHITECTURE.md.
#
# GitHub renders the mermaid blocks in the markdown natively, but a PDF cannot,
# so the diagrams are rendered to PNG first and swapped in. Typst is the PDF
# engine because it is a single binary; LaTeX would need a full distribution.
set -euo pipefail
cd "$(dirname "$0")"
DOC=/DATA2/home/parth/.conda/envs/cp-doc/bin
UI=/DATA2/home/parth/.conda/envs/cp-ui/bin
export PUPPETEER_CACHE_DIR=/DATA2/home/parth/.cache/puppeteer

python3 - <<'PY'
import pathlib, re
src = pathlib.Path("ARCHITECTURE.md").read_text()
names = ["architecture", "lifecycle", "tier2", "recalibration"]
caps  = ["System architecture: two deployables, one repository",
         "The request lifecycle",
         "Tier 2: claim decomposition and per-claim retrieval",
         "Recalibration from human feedback"]
pathlib.Path("img").mkdir(exist_ok=True)
for i, b in enumerate(re.findall(r"```mermaid\n(.*?)```", src, re.S)):
    pathlib.Path(f"img/{names[i]}.mmd").write_text(b)
i = 0
def sub(m):
    global i
    n, c = names[i], caps[i]; i += 1
    return f"![{c}](img/{n}.png)\n"
out = re.sub(r"```mermaid\n.*?```\n", sub, src, flags=re.S)
out = re.sub(r"## Table of contents\n\n(?:\d+\.\s+\[.*\n)+\n---\n\n", "", out)
out = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", out)   # typst has no anchors
pathlib.Path(".pdf.md").write_text(out)
PY

echo '{"args":["--no-sandbox","--disable-dev-shm-usage","--disable-gpu"]}' > .pptr.json
for d in architecture lifecycle tier2 recalibration; do
  "$UI/../../../Hackathons/Accenture_Innovation_Challenge/controlplane/ui/node_modules/.bin/mmdc" \
    -i "img/$d.mmd" -o "img/$d.png" -b white -w 1200 -s 2 -p .pptr.json >/dev/null 2>&1 \
    || echo "  warn: could not render $d"
done

PATH="$DOC:$PATH" pandoc .pdf.md -o ControlPlane-ai.pdf \
  --pdf-engine=typst --toc --toc-depth=2 \
  -V papersize=a4 -V margin-x=2cm -V margin-y=2cm -V fontsize=10pt \
  -M title="ControlPlane.ai — System Documentation" \
  -M subtitle="Team NavRatnas · Accenture Innovation Challenge 2026 · Track 1"

rm -f .pdf.md .pptr.json img/*.mmd
echo "built docs/ControlPlane-ai.pdf"
