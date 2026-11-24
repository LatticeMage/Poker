// Thermal gate. Tools do not call it directly: dispatch() in dispatch.cuh calls it before every batch of GPU
// work. If the GPU is hotter than the pause
// limit it waits until the GPU has cooled down to the resume limit (a hysteresis, so it does not stop and go
// around one temperature). It needs no root and no NVML linking: it reads the temperature from nvidia-smi.
//
// Two rules. A task may only START (the first batch of a dispatch) when the GPU is at or below GPU_RESUME_C.
// Once running, the next batches go on until the GPU reaches GPU_PAUSE_C; it then waits until it is back at or
// below GPU_RESUME_C.
//
// Settings (environment variables, defaults in brackets):
//   GPU_PAUSE_C  [63]   while running, wait when the GPU is at or above this temperature; 0 turns the gate off
//   GPU_RESUME_C [45]   start a task only at or below this, and after a wait keep waiting until at or below this
//   GPU_POLL_S   [2]    seconds between readings while waiting
//
// Waiting inside a process is a poor way to cool: the process keeps its GPU context and the card stays at a higher
// floor (53 to 56 C instead of 46 C). So when the program was started by gpu-run.sh (it sets GPU_RUN=1) the gate does
// not wait: it leaves with status 75 (EXIT_EVICTED) and gpu-run.sh waits outside, with the GPU empty, and starts it
// again. That loses nothing: dispatch() has written its state to the relay file after every finished batch.
//
// The GPU is read as nvidia-smi index 0 (one card). If nvidia-smi cannot be run the gate lets the work
// through and says so once. The reading and the sleep can be replaced (see ThermalHooks), which is how
// test_thermal.cu checks the logic without waiting for a real GPU to cool.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

struct ThermalHooks {
	int (*read_c)();            // temperature in C, or -1 if unavailable
	void (*sleep_s)(double);    // wait this many seconds
	double min_interval_s;      // a cool reading is trusted for this long (batches can be milliseconds apart)
};

static inline int thermal_read_nvidia_smi() {
	FILE* p = popen("nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
	if (!p) return -1;
	int t = -1;
	if (fscanf(p, "%d", &t) != 1) t = -1;
	pclose(p);
	return t;
}

static inline void thermal_real_sleep(double s) {
	std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

static inline bool hooks_is_real(const ThermalHooks& h) { return h.read_c == thermal_read_nvidia_smi; }  // tests inject fakes

static inline ThermalHooks& thermal_hooks() {
	static ThermalHooks h = {thermal_read_nvidia_smi, thermal_real_sleep, 2.0};
	return h;
}

static inline int thermal_env(const char* name, int fallback) {
	const char* v = getenv(name);
	return (v && *v) ? atoi(v) : fallback;
}

// Waits while the GPU is too hot. Returns the seconds spent waiting (0 if it did not wait).
// pause_c / resume_c / poll_s < 0 means: take it from the environment (or the default).
// starting = true is the rule for the first batch of a task: wait until the GPU is at or below resume_c.
static inline double thermal_gate(int pause_c = -1, int resume_c = -1, double poll_s = -1, bool starting = false) {
	if (pause_c < 0) pause_c = thermal_env("GPU_PAUSE_C", 63);
	if (resume_c < 0) resume_c = thermal_env("GPU_RESUME_C", 45);
	if (poll_s < 0) poll_s = thermal_env("GPU_POLL_S", 2);
	if (pause_c == 0) return 0;
	ThermalHooks& h = thermal_hooks();
	static bool warned = false;
	static std::chrono::steady_clock::time_point last_ok;
	static bool have_last = false;
	const auto now = std::chrono::steady_clock::now();
	if (!starting && have_last && h.min_interval_s > 0 && std::chrono::duration<double>(now - last_ok).count() < h.min_interval_s)
		return 0;  // read a cool temperature a moment ago
	int t = h.read_c();
	if (t < 0) {
		if (!warned) fprintf(stderr, "[thermal] cannot read the GPU temperature, not gating\n");
		warned = true;
		return 0;
	}
	if (starting ? t <= resume_c : t < pause_c) {
		last_ok = now;
		have_last = true;
		return 0;
	}
	have_last = false;
	if (starting) fprintf(stderr, "[thermal] %d C > %d C: not cool enough to start\n", t, resume_c);
	else fprintf(stderr, "[thermal] %d C >= %d C: too hot\n", t, pause_c);
	if (getenv("GPU_RUN") && hooks_is_real(h)) {
		fprintf(stderr, "[thermal] leaving (status 75) so that the GPU can empty and cool; gpu-run.sh starts me again\n");
		exit(75);
	}
	double waited = 0, next_note = 60;
	while (t > resume_c) {
		h.sleep_s(poll_s);
		waited += poll_s;
		t = h.read_c();
		if (t < 0) break;  // lost the reading: do not hang forever
		if (waited >= next_note) {
			fprintf(stderr, "[thermal] still %d C after %.0f s\n", t, waited);
			next_note += 60;
		}
	}
	fprintf(stderr, "[thermal] %d C: continuing after %.0f s\n", t, waited);
	last_ok = std::chrono::steady_clock::now();
	have_last = t >= 0;
	return waited;
}
