#!/bin/bash
# usage: scripts/run_astra.sh <blind-file> <out-dir>
# One blind prompt to GPT-6 Astra through the Codex CLI with every tool off, as in the post; run from the repo root.
# For the exp3 task (code allowed), run it from a directory holding weights.npz and drop the shell_tool=false flag.
f=$1; out=$2; id=$(basename "$f" .txt); mkdir -p "$out"
codex exec --skip-git-repo-check --json -m gpt-6-astra -s read-only \
  -c 'model_reasoning_effort="high"' -c 'web_search="disabled"' \
  -c "model_instructions_file=\"$PWD/prompts/astra_instructions.md\"" \
  -c features.shell_tool=false -c features.unified_exec=false -c features.apps=false -c features.plugins=false \
  -c features.view_image=false -c features.multi_agent=false \
  -o "$out/$id.md" - < "$f" > "$out/$id.events.jsonl" 2> "$out/$id.err"
echo "$id rc=$?"
