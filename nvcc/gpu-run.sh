#!/usr/bin/env bash
# Runs a GPU command as a relay: whenever it is evicted (it exits with 75 after saving its state, or it is killed
# by gpu-cooler.sh), wait for the GPU to cool down and run the same command again. Because the command's state is
# in its relay file, each run carries on from the last saved batch.
#
#   gpu-run.sh [-n max_runs=200] -- command [args...]
#   GPU_RESUME_C=53 sets the temperature the GPU must be at or below before every (re)start; default 53.
#   GPU_FLOOR_S=120: start anyway once the idle temperature has not moved by more than 1 C for that long.
#   The command runs with GPU_RUN=1 and GPU_START_OK=1 (the start check is done here, not again inside): if the GPU is too hot when it wants to start or go on, thermal.cuh makes it leave with
#   status 75 at once (its state is already in its relay file), so the GPU can empty and cool outside the process.
#
# Exit status: that of the command when it finishes (0 = done), or 1 after max_runs evictions.
MAX=200
if [ "${1:-}" = "-n" ]; then MAX=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ $# -eq 0 ] && { echo "usage: gpu-run.sh [-n max_runs] -- command [args...]" >&2; exit 2; }
START_C=${GPU_RESUME_C:-53}
stamp() { date '+%F %T'; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }

# The GPU is cool enough to start when it is at or below START_C, or when it has reached its floor: the fan stops at low
# temperature, so an idle card settles at a temperature that can be above START_C (46 to 53 C seen on an RTX 4060 Ti) and
# waiting longer would never end. The floor counts when the temperature has not moved by more than 1 C for FLOOR_S
# seconds and is still well under the eviction temperature (GPU_PAUSE_C, default 63, minus 5).
FLOOR_S=${GPU_FLOOR_S:-120}
PAUSE_C=${GPU_PAUSE_C:-63}
wait_cool() {
	local t note=0 lo=999 hi=0 since=0 now
	while true; do
		t=$(temp)
		{ [ -z "$t" ] || [ "$t" -le "$START_C" ]; } && return
		now=$(date +%s)
		if [ "$since" -eq 0 ] || [ "$t" -lt $((lo - 1)) ] || [ "$t" -gt $((hi + 1)) ]; then lo=$t; hi=$t; since=$now; fi
		[ "$t" -lt "$lo" ] && lo=$t
		[ "$t" -gt "$hi" ] && hi=$t
		if [ $((now - since)) -ge "$FLOOR_S" ] && [ "$t" -le $((PAUSE_C - 5)) ]; then
			echo "$(stamp) [gpu-run] the GPU has settled at ${t}C (floor, above ${START_C}C): starting anyway"
			return
		fi
		if [ "$note" -eq 0 ]; then echo "$(stamp) [gpu-run] ${t}C > ${START_C}C: waiting for the GPU to cool down (or to settle)"; note=1; fi
		sleep 10
	done
}

runs=0
while [ "$runs" -lt "$MAX" ]; do
	wait_cool
	runs=$((runs + 1))
	echo "$(stamp) [gpu-run] run $runs: $*"
	GPU_RUN=1 GPU_START_OK=1 "$@"
	rc=$?
	case "$rc" in
		75|137|143) echo "$(stamp) [gpu-run] evicted (status $rc): rerunning after the GPU has cooled down" ;;
		*) echo "$(stamp) [gpu-run] finished with status $rc after $runs run(s)"; exit "$rc" ;;
	esac
	sleep 5
done
echo "$(stamp) [gpu-run] gave up after $MAX runs" >&2
exit 1
