#!/usr/bin/env bash
# Runs a GPU command as a relay: whenever it is evicted (it exits with 75 after saving its state, or it is killed
# by gpu-cooler.sh), wait for the GPU to cool down and run the same command again. Because the command's state is
# in its relay file, each run carries on from the last saved batch.
#
#   gpu-run.sh [-n max_runs=200] -- command [args...]
#   GPU_RESUME_C=45 sets the temperature the GPU must be at or below before every (re)start; default 45.
#   The command runs with GPU_RUN=1: if the GPU is too hot when it wants to start or go on, thermal.cuh makes it leave with
#   status 75 at once (its state is already in its relay file), so the GPU can empty and cool outside the process.
#
# Exit status: that of the command when it finishes (0 = done), or 1 after max_runs evictions.
MAX=200
if [ "${1:-}" = "-n" ]; then MAX=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ $# -eq 0 ] && { echo "usage: gpu-run.sh [-n max_runs] -- command [args...]" >&2; exit 2; }
START_C=${GPU_RESUME_C:-45}
stamp() { date '+%F %T'; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_cool() {
	local t note=0
	while true; do
		t=$(temp)
		{ [ -z "$t" ] || [ "$t" -le "$START_C" ]; } && return
		if [ "$note" -eq 0 ]; then echo "$(stamp) [gpu-run] ${t}C > ${START_C}C: waiting for the GPU to cool down"; note=1; fi
		sleep 5
	done
}

runs=0
while [ "$runs" -lt "$MAX" ]; do
	wait_cool
	runs=$((runs + 1))
	echo "$(stamp) [gpu-run] run $runs: $*"
	GPU_RUN=1 "$@"
	rc=$?
	case "$rc" in
		75|137|143) echo "$(stamp) [gpu-run] evicted (status $rc): rerunning after the GPU has cooled down" ;;
		*) echo "$(stamp) [gpu-run] finished with status $rc after $runs run(s)"; exit "$rc" ;;
	esac
	sleep 5
done
echo "$(stamp) [gpu-run] gave up after $MAX runs" >&2
exit 1
