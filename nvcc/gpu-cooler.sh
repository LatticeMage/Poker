#!/usr/bin/env bash
# Keeps the GPU under a temperature ceiling without root or clock control: when the GPU is too hot it EVICTS the
# CUDA processes of this user. It does not pause them: a paused process keeps its GPU context, the card stays in
# a high power state (about 23 W instead of 10 W) and does not cool down.
#
#   gpu-cooler.sh [evict_c=63] [interval_s=2] [grace_s=30]
#
# Eviction = SIGTERM, then SIGKILL for whatever is still there after grace_s seconds. dispatch() (dispatch.cuh)
# catches SIGTERM, finishes its batch, saves its state to its relay file and exits with status 75, so nothing is
# lost: the files are the truth and the GPU is only a workspace. Run the task through gpu-run.sh and it is started
# again once the card has cooled down; the rerun reloads its state from the file and carries on. A process that
# does not use dispatch() and a relay loses its progress.
#
# Needed because clocks and power limit cannot be changed from inside the container (no CAP_SYS_ADMIN).
set -u
EVICT=${1:-63}
EVERY=${2:-2}
GRACE=${3:-30}
ME=$$
USER_NAME=$(id -un)

stamp() { date '+%F %T'; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }

# Processes of this user that have libcuda mapped, not this script.
cuda_pids() {
	local d p
	for d in /proc/[0-9]*; do
		p=${d#/proc/}
		[ "$p" = "$ME" ] && continue
		[ "$(stat -c %U "$d" 2>/dev/null)" = "$USER_NAME" ] || continue
		grep -q libcuda "$d/maps" 2>/dev/null || continue
		echo "$p"
	done
}

trap 'echo "$(stamp) stopped"; exit 0' TERM INT HUP
echo "$(stamp) gpu-cooler: evict above ${EVICT}C, every ${EVERY}s, grace ${GRACE}s"
last_beat=0
while true; do
	t=$(temp)
	if [ -n "$t" ] && [ "$t" -ge "$EVICT" ]; then
		pids=$(cuda_pids)
		if [ -n "$pids" ]; then
			echo "$(stamp) ${t}C >= ${EVICT}C: asking $(echo $pids | wc -w) process(es) to save and leave: $(echo $pids)"
			for p in $pids; do kill -CONT "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
			waited=0
			while [ "$waited" -lt "$GRACE" ] && [ -n "$(cuda_pids)" ]; do sleep 1; waited=$((waited + 1)); done
			left=$(cuda_pids)
			if [ -n "$left" ]; then
				echo "$(stamp) still there after ${GRACE}s, killing: $(echo $left)"
				for p in $left; do kill -KILL "$p" 2>/dev/null; done
			fi
			echo "$(stamp) evicted; the GPU is empty now ($(temp)C)"
		fi
	fi
	now=$(date +%s)
	if [ $((now - last_beat)) -ge 60 ]; then
		echo "$(stamp) ${t:-?}C"
		last_beat=$now
	fi
	sleep "$EVERY"
done
