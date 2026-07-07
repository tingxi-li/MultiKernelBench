#!/bin/bash
# timed_bench.sh — measurement wrapper for the 6-op cross-DSL convergence redo.
#
# Wraps a cell's own scripts/bench.sh, times ONLY compile+bench (compute_s; the
# agent's thinking sits outside this timer), parses the machine block, and appends
# ONE row to <cell>/convergence.csv. The agent's only logging duty is <variant_desc>.
# See CONVERGENCE_PROTOCOL.md. Method roams above the bench call; this is the fixed
# yardstick at/below it.
#
# usage: timed_bench.sh <cell_dir> "<variant_desc>" [--csv PATH] [--ncu-key STR]
#                       [--agent-s N] [--label LBL]
#   <cell_dir>      e.g. ako_runs/layer_norm/cuda_noptx
#   <variant_desc>  one-line label, e.g. "L2-resident 2-pass" (commas -> ';')
#   --csv PATH      override log path (default <cell>/convergence.csv); use a
#                   scratch path for throwaway scaffold checks
#   --ncu-key STR   the one steering number if this variant was profiled ("passes=3.04")
#   --agent-s N     optional agent think-seconds, for the record only (not compared)
#   --label LBL     trajectory-dir label (default: sanitized variant_desc)
set -eo pipefail

usage() { echo "usage: timed_bench.sh <cell_dir> \"<variant_desc>\" [--csv PATH] [--ncu-key STR] [--agent-s N] [--label LBL]" >&2; exit 2; }

CSV=""; NCUKEY=""; AGENTS=""; LABEL=""; GPUPIN=""; SERIAL3=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --csv) CSV="$2"; shift 2;;
    --ncu-key) NCUKEY="$2"; shift 2;;
    --agent-s) AGENTS="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --gpu) GPUPIN="$2"; shift 2;;
    # memory-bound cells: pin GPU3 AND flock a shared lock so concurrent agents'
    # benches serialize (Discipline 4: overlapping memory-bound benches corrupt the
    # speedup ratio). The compute-timer sits INSIDE the lock so lock-wait never
    # inflates compute_s.
    --gpu3-serial) GPUPIN=3; SERIAL3=1; shift 1;;
    # --serialize: acquire the shared bench lock WITHOUT forcing GPU3 — combine with
    # --gpu N to keep a memory-bound op on its own dedicated card while still preventing
    # its timed bench from overlapping another lane's (cross-GPU memory-P-state contamination).
    --serialize) SERIAL3=1; shift 1;;
    -h|--help) usage;;
    *) POS+=("$1"); shift;;
  esac
done
CELL="${POS[0]:-}"; DESC="${POS[1]:-}"
[ -z "$CELL" ] && usage
[ -z "$DESC" ] && usage

CELL="$(cd "$CELL" && pwd)"
BENCH="$CELL/scripts/bench.sh"
[ -f "$BENCH" ] || { echo "no bench.sh at $BENCH" >&2; exit 1; }
[ -z "$CSV" ] && CSV="$CELL/convergence.csv"

# sanitize free-text fields so they can't break the CSV
DESC_CLEAN="$(printf '%s' "$DESC" | tr ',\n\r' ';  ' | tr -d '"')"
NCUKEY_CLEAN="$(printf '%s' "$NCUKEY" | tr ',\n\r' ';  ' | tr -d '"')"
[ -z "$LABEL" ] && LABEL="$(printf '%s' "$DESC" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-40)"

# header on first write
if [ ! -f "$CSV" ]; then
  printf 'iter,cum_compute_s,variant_desc,runtime_ms,speedup,ncu_key,kept,agent_s\n' > "$CSV"
fi

# prior state: iter count, last cum_compute_s, best speedup so far
read PREV_ITER PREV_CUM PREV_BEST < <(awk -F',' 'NR>1{n=$1; cum=$2; s=$5+0; if(s>best)best=s} END{printf "%d %s %s\n", (n?n:0), (cum==""?0:cum), (best==""?0:best)}' "$CSV") || true
: "${PREV_ITER:=0}" "${PREV_CUM:=0}" "${PREV_BEST:=0}"
ITER=$((PREV_ITER+1))

echo "== timed_bench: $CELL  iter=$ITER  variant='$DESC_CLEAN'${GPUPIN:+  gpu=$GPUPIN}${SERIAL3:+ (serial-lock)} =="
TMPOUT="$(mktemp)"
trap 'rm -f "$TMPOUT"' EXIT

# serialize across concurrent agents on GPU3, if requested (blocks; wait is OUTSIDE the timer)
if [ -n "$SERIAL3" ]; then
  # FIXED absolute path (not $TMPDIR-relative) so all concurrent agents share one lock
  LOCKFILE="/home/lxt230026/MultiKernelBench/ako_runs/tools/.gpu3_bench.lock"
  exec 9>"$LOCKFILE"
  echo "-- waiting for GPU3 bench lock ($LOCKFILE) ..."
  flock 9
fi

START=$(date +%s.%N)
set +e
if [ -n "$GPUPIN" ]; then
  CUDA_VISIBLE_DEVICES="$GPUPIN" bash "$BENCH" "$LABEL" > "$TMPOUT" 2>&1
else
  bash "$BENCH" "$LABEL" > "$TMPOUT" 2>&1
fi
BXIT=$?
set -e
END=$(date +%s.%N)
[ -n "$SERIAL3" ] && flock -u 9 2>/dev/null || true
cat "$TMPOUT"
COMPUTE_S=$(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.2f", b-a}')

# parse region: a failed variant emits no COMPILED/SPEEDUP line, so greps may
# return nonzero — don't let errexit/pipefail abort before we log the row.
set +eo pipefail
COMPILED="$(grep -E '^COMPILED: ' "$TMPOUT" | tail -1 | sed 's/^COMPILED: //')"
CORRECT="$(grep -E '^CORRECT: '  "$TMPOUT" | tail -1 | sed 's/^CORRECT: //')"
RUNTIME="$(grep -E '^RUNTIME: '   "$TMPOUT" | tail -1 | sed 's/^RUNTIME: //')"
SPEEDUP="$(grep -E '^SPEEDUP: '   "$TMPOUT" | tail -1 | sed 's/^SPEEDUP: //; s/x$//')"

KEPT=0
if [ "$COMPILED" = "True" ] && [ "$CORRECT" = "True" ] && [ -n "$SPEEDUP" ] && [ "$SPEEDUP" != "-1" ]; then
  KEPT=$(awk -v s="$SPEEDUP" -v b="$PREV_BEST" 'BEGIN{print (s>b)?1:0}')
else
  SPEEDUP=""; RUNTIME=""   # blank on compile/correctness failure
fi
CUM=$(awk -v p="$PREV_CUM" -v c="$COMPUTE_S" 'BEGIN{printf "%.2f", p+c}')

printf '%d,%s,%s,%s,%s,%s,%d,%s\n' "$ITER" "$CUM" "$DESC_CLEAN" "$RUNTIME" "$SPEEDUP" "$NCUKEY_CLEAN" "$KEPT" "$AGENTS" >> "$CSV"
echo "-- logged: iter=$ITER speedup=${SPEEDUP:-FAIL} runtime_ms=${RUNTIME:--} compute_s=${COMPUTE_S}s cum=${CUM}s kept=$KEPT (prev_best=$PREV_BEST) compiled=$COMPILED correct=$CORRECT -> $CSV"
exit $BXIT
