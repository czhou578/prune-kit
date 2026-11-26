#!/usr/bin/env bash
# Gate C of plans/PLAN_PRUNE_MOE_KIT.md §7: `prune-moe-kit gate --score` scores the 60 recorded turns of the Qwen prune
# on the live Qwen (K 208), and the NLL is equal to the record inside the noise of one replay.
# The noise is large for one turn (2% to 5% between two runs) and small for the total (0.56% in the record). So the
# script also runs the scorer of the record on the same server, as the control.
# It sends requests to the live server at an idle time. It stops no server. It takes about 20 minutes.
#   tmux new -d -s gate-c job run --steps 3 --log runs/gate-c/gate-c.log \
#     --desc "prune-moe-kit gate C: the new replay scorer against the scorer of the record, on the live Qwen" \
#     "prune-moe-kit-gate-c" -- scripts/gate_c.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RECORD=${PRUNE_MOE_KIT_RECORD:-$HOME/Projects/dsv4-specialize}
URL=${PRUNE_MOE_KIT_QWEN_URL:-http://127.0.0.1:8889}
OUT=$PWD/runs/gate-c
n=0
step() { echo "== $1"; job step "$n" "$1" 2>/dev/null || true; n=$((n + 1)); }

busy=$(curl -fsS "$URL/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {s += $NF} END {print s + 0}')
[ "$busy" = 0 ] || { echo "the server has $busy requests: start the gate at an idle time"; exit 2; }
mkdir -p "$OUT" && rm -f "$OUT"/new.jsonl "$OUT"/old.jsonl
step "score with prune-moe-kit"
uv run prune-moe-kit gate --score --data "$RECORD/data/qwen-data" --base-url "$URL" \
    --recorded "$RECORD/results/qwen-recorded.jsonl" --out "$OUT/new.jsonl"
step "score with the scorer of the record"
(cd "$RECORD" && "$OLDPWD/.venv/bin/python" tools/qwen_gate.py --score --base-url "$URL" --out "$OUT/old.jsonl")
step "compare"
for pair in "$RECORD/results/qwen-fit-g208-score.jsonl new" "$OUT/old.jsonl new" "$RECORD/results/qwen-fit-g208-score.jsonl old"; do
    set -- $pair
    echo "-- reference: $(basename "$1"); new: $2.jsonl"
    uv run prune-moe-kit gate --compare "$1" "$OUT/$2.jsonl" || true
done
