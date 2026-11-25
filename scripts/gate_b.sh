#!/usr/bin/env bash
# Gate B of plans/PLAN_PRUNE_KIT.md §7: `prune-kit build` writes K 208 of the full Qwen with the rank of the record,
# and each file of the result is equal to the folder of the prune by hand, byte for byte.
# The run writes about 19 GiB into runs/gate-b and removes them at the end. It needs no GPU and stops no server.
# Start it on the jobs page:
#   tmux new -d -s gate-b job run --steps 3 --log runs/gate-b.log \
#     --desc "prune-kit gate B: st_moe builds Qwen K 208 and compares it with the prune by hand" \
#     "prune-kit-gate-b" -- scripts/gate_b.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RECORD=${PRUNE_KIT_RECORD:-$HOME/Projects/dsv4-specialize}
MODEL=${PRUNE_KIT_QWEN:-unsloth/Qwen3.6-35B-A3B-NVFP4-Fast}
OUT=runs/gate-b/qwen-k208
n=0
step() { echo "== $1"; job step "$n" "$1" 2>/dev/null || true; n=$((n + 1)); }

rm -rf runs/gate-b
step "build K 208"
uv run prune-kit build "$MODEL" --rank "$RECORD/results/qwen-rank.json" --k 208 --draft keep --out "$OUT"
step "compare with the prune by hand"
fail=0
for f in "$RECORD"/data/qwen-k208/*.safetensors "$RECORD"/data/qwen-k208/{config.json,model.safetensors.index.json}; do
    if cmp -s "$f" "$OUT/$(basename "$f")"; then echo "same      $(basename "$f")"; else echo "DIFFERENT $(basename "$f")"; fail=1; fi
done
step "remove the output"
rm -rf runs/gate-b
[ "$fail" = 0 ] && echo "gate B: pass" || { echo "gate B: FAIL"; exit 1; }
