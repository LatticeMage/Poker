#!/usr/bin/env bash
# End-to-end check of a relay tool: an interrupted run, restarted by gpu-run.sh, gives the same result as an
# uninterrupted one. Not a GPU job itself (start it with setsid nohup); the runs inside go through gpu-run.sh.
#
#   e2e-evict.sh term|kill WORKDIR RELAY_FILE RESULT_FILE[,RESULT_FILE...] -- command {D} ...
#
# {D} in the command (and in RELAY_FILE, RESULT_FILEs) stands for the directory of one run. The script
#   1. runs the command to the end under gpu-run.sh in WORKDIR/ref (the reference),
#   2. runs it again in WORKDIR/term (or WORKDIR/kill) under gpu-run.sh, waits until the relay file exists and the
#      process has run E2E_DELAY seconds more (default 8), then sends SIGTERM (what gpu-cooler.sh sends: the tool
#      finishes its batch, saves, exits with 75) or SIGKILL (nothing runs: the last saved batch is all there is) to the
#      CUDA process; gpu-run.sh restarts it once the card is cool and the restarted run carries on from the file,
#   3. compares the result files byte for byte with the reference.
# Prints PASS (exit 0) when the files are identical and the process was restarted at least once, else FAIL.
set -u
MODE=$1; WORK=$2; RELAY=$3; RESULTS=$4; shift 4
[ "${1:-}" = "--" ] && shift
GPURUN=${GPURUN:-/home/ubuntu/app/Github/LatticeMage/Poker/bin/gpu-run.sh}
DELAY=${E2E_DELAY:-8}
stamp() { date '+%F %T'; }
sub() { local s=$1; shift; local out=(); for a in "$@"; do out+=("${a//\{D\}/$s}"); done; printf '%s\n' "${out[@]}"; }
mkdir -p "$WORK"

echo "$(stamp) [e2e] reference run"
rm -rf "$WORK/ref"; mkdir -p "$WORK/ref"
mapfile -t CMD < <(sub "$WORK/ref" "$@")
bash "$GPURUN" -- "${CMD[@]}" > "$WORK/ref.log" 2>&1 || { echo "FAIL: the reference run failed (see $WORK/ref.log)"; exit 1; }

echo "$(stamp) [e2e] $MODE run"
rm -rf "$WORK/$MODE"; mkdir -p "$WORK/$MODE"
mapfile -t CMD < <(sub "$WORK/$MODE" "$@")
REL=${RELAY//\{D\}/$WORK/$MODE}
bash "$GPURUN" -- "${CMD[@]}" > "$WORK/$MODE.log" 2>&1 &
RP=$!
for _ in $(seq 1 7200); do [ -s "$REL" ] && break; sleep 1; done  # up to 2 h: other GPU tasks may be queued ahead
[ -s "$REL" ] || { echo "FAIL: no relay file appeared"; kill "$RP" 2>/dev/null; exit 1; }
sleep "$DELAY"
CP=$(pgrep -P "$RP" | head -1)
if [ -z "$CP" ]; then echo "FAIL: no running tool process (finished too early? raise the work or lower E2E_DELAY)"; wait "$RP"; exit 1; fi
if [ "$MODE" = term ]; then SIG=TERM; else SIG=KILL; fi
echo "$(stamp) [e2e] SIG$SIG to pid $CP; relay file at that moment holds $(stat -c %s "$REL") bytes"
kill -"$SIG" "$CP"
wait "$RP"
RC=$?
RUNS=$(grep -c "\[gpu-run\] run " "$WORK/$MODE.log")
echo "$(stamp) [e2e] gpu-run finished with status $RC after $RUNS run(s)"
OK=1
[ "$RC" -eq 0 ] && [ "$RUNS" -ge 2 ] || OK=0
IFS=, read -ra FILES <<< "$RESULTS"
for f in "${FILES[@]}"; do
	if cmp -s "${f//\{D\}/$WORK/ref}" "${f//\{D\}/$WORK/$MODE}"; then echo "identical: $f"; else echo "DIFFERENT: $f"; OK=0; fi
done
if [ "$OK" -eq 1 ]; then echo "PASS ($MODE)"; exit 0; fi
echo "FAIL ($MODE)"; exit 1
