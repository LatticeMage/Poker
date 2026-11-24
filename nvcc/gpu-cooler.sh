#!/usr/bin/env bash
# Keeps the GPU under a temperature ceiling without root or clock control: when the GPU gets too hot it
# pauses (SIGSTOP) every process of this user that has the CUDA driver mapped, and resumes (SIGCONT) exactly
# those once it has cooled down. Work in flight on the GPU finishes, then the card idles and cools.
#
#   gpu-cooler.sh [pause_c=70] [resume_c=62] [interval_s=2]
#
# Needed because clocks and power limit cannot be changed from inside the container (no CAP_SYS_ADMIN).
# Stop it with SIGTERM or Ctrl-C: it resumes what it paused. If it is killed with SIGKILL, run
# `pkill -CONT -u "$USER"` to resume the paused processes.
set -u
PAUSE=${1:-70}
RESUME=${2:-62}
EVERY=${3:-2}
ME=$$
USER_NAME=$(id -un)
paused=()

stamp() { date '+%F %T'; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }

# Processes of this user that have libcuda mapped, not stopped already, not this script.
cuda_pids() {
	local d p state
	for d in /proc/[0-9]*; do
		p=${d#/proc/}
		[ "$p" = "$ME" ] && continue
		[ "$(stat -c %U "$d" 2>/dev/null)" = "$USER_NAME" ] || continue
		grep -q libcuda "$d/maps" 2>/dev/null || continue
		state=$(awk '/^State:/ {print $2}' "$d/status" 2>/dev/null)
		[ "$state" = "T" ] && continue
		echo "$p"
	done
}

resume_all() {
	local p
	for p in "${paused[@]:-}"; do [ -n "$p" ] && kill -CONT "$p" 2>/dev/null; done
	paused=()
}
trap 'resume_all; echo "$(stamp) stopped, resumed everything"; exit 0' TERM INT HUP

echo "$(stamp) gpu-cooler: pause above ${PAUSE}C, resume below ${RESUME}C, every ${EVERY}s"
last_beat=0
while true; do
	t=$(temp)
	[ -z "$t" ] && { sleep "$EVERY"; continue; }
	if [ "${#paused[@]}" -eq 0 ]; then
		if [ "$t" -ge "$PAUSE" ]; then
			for p in $(cuda_pids); do kill -STOP "$p" 2>/dev/null && paused+=("$p"); done
			echo "$(stamp) ${t}C >= ${PAUSE}C: paused ${#paused[@]} process(es): ${paused[*]:-none}"
		fi
	else
		# a process started while we are paused would add heat: stop it too
		for p in $(cuda_pids); do kill -STOP "$p" 2>/dev/null && paused+=("$p"); done
		if [ "$t" -le "$RESUME" ]; then
			echo "$(stamp) ${t}C <= ${RESUME}C: resuming ${paused[*]}"
			resume_all
		fi
	fi
	now=$(date +%s)
	if [ $((now - last_beat)) -ge 60 ]; then
		echo "$(stamp) ${t}C paused=${#paused[@]}"
		last_beat=$now
	fi
	sleep "$EVERY"
done
