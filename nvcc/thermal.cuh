// Thermal gate. Tools do not call it directly: dispatch() in dispatch.cuh calls it before every batch of GPU
// work. If the GPU is hotter than the pause
// limit it waits until the GPU has cooled down to the resume limit (a hysteresis, so it does not stop and go
// around one temperature). It needs no root and no NVML linking: it reads the temperature from nvidia-smi.
//
// Settings (environment variables, defaults in brackets):
//   GPU_PAUSE_C  [60]   wait when the GPU is at or above this temperature; 0 turns the gate off
//   GPU_RESUME_C [52]   keep waiting until the GPU is at or below this
//   GPU_POLL_S   [2]    seconds between readings while waiting
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

static inline ThermalHooks& thermal_hooks() {
	static ThermalHooks h = {thermal_read_nvidia_smi, thermal_real_sleep};
	return h;
}

static inline int thermal_env(const char* name, int fallback) {
	const char* v = getenv(name);
	return (v && *v) ? atoi(v) : fallback;
}

// Waits while the GPU is too hot. Returns the seconds spent waiting (0 if it did not wait).
// pause_c / resume_c / poll_s < 0 means: take it from the environment (or the default).
static inline double thermal_gate(int pause_c = -1, int resume_c = -1, double poll_s = -1) {
	if (pause_c < 0) pause_c = thermal_env("GPU_PAUSE_C", 60);
	if (resume_c < 0) resume_c = thermal_env("GPU_RESUME_C", 52);
	if (poll_s < 0) poll_s = thermal_env("GPU_POLL_S", 2);
	if (pause_c == 0) return 0;
	ThermalHooks& h = thermal_hooks();
	static bool warned = false;
	int t = h.read_c();
	if (t < 0) {
		if (!warned) fprintf(stderr, "[thermal] cannot read the GPU temperature, not gating\n");
		warned = true;
		return 0;
	}
	if (t < pause_c) return 0;
	fprintf(stderr, "[thermal] %d C >= %d C: waiting until <= %d C\n", t, pause_c, resume_c);
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
	return waited;
}
