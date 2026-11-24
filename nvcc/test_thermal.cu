// Checks for thermal.cuh: the gate logic with a fake thermometer and clock, and the real reading.
//
//   nvcc -O3 -arch=sm_89 -o test_thermal test_thermal.cu && ./test_thermal

#include "thermal.cuh"

#include <vector>

static std::vector<int> g_seq;  // temperatures the fake thermometer returns, in order
static size_t g_pos = 0;
static double g_slept = 0;
static int fake_read() { return g_pos < g_seq.size() ? g_seq[g_pos++] : g_seq.back(); }
static void fake_sleep(double s) { g_slept += s; }

// Runs the gate against a sequence; returns {seconds waited, readings used}.
static void run(const std::vector<int>& seq, int pause_c, int resume_c, double* waited, size_t* reads) {
	g_seq = seq;
	g_pos = 0;
	g_slept = 0;
	thermal_hooks() = {fake_read, fake_sleep, 0};
	*waited = thermal_gate(pause_c, resume_c, 2);
	*reads = g_pos;
}

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok " : "BAD", what);
	failures += !ok;
}

int main() {
	double w;
	size_t r;

	// starting a task: only at or below the resume limit
	g_seq = {55, 54, 52};
	g_pos = 0;
	g_slept = 0;
	thermal_hooks() = {fake_read, fake_sleep, 0};
	w = thermal_gate(60, 52, 2, true);
	check(w == 4 && g_pos == 3, "starting at 55 C waits until 52 C (a running task would have gone on at 55)");
	g_seq = {52};
	g_pos = 0;
	w = thermal_gate(60, 52, 2, true);
	check(w == 0 && g_pos == 1, "starting at exactly the resume limit does not wait");
	g_seq = {53};
	g_pos = 0;
	w = thermal_gate(60, 52, 2, false);
	check(w == 0, "a task already running goes on at 53 C");

	run({55}, 60, 52, &w, &r);
	check(w == 0 && r == 1, "below the pause limit: no waiting, one reading");

	run({60, 59, 56, 53, 52}, 60, 52, &w, &r);
	check(w == 8 && r == 5, "at the pause limit it waits and stops only at the resume limit (not at 59, 56, 53)");

	run({75, 70, 65, 58, 51}, 60, 52, &w, &r);
	check(w == 8 && r == 5, "a hot start waits until it is at or below the resume limit");

	run({58, 45}, 60, 52, &w, &r);
	check(w == 0 && r == 1, "between resume and pause, not previously hot: no waiting (hysteresis only after pausing)");

	run({90}, 0, 52, &w, &r);
	check(w == 0 && r == 0, "pause limit 0 turns the gate off without even reading");

	run({-1}, 60, 52, &w, &r);
	check(w == 0, "no reading available: the gate lets the work through");

	run({70, -1}, 60, 52, &w, &r);
	check(r == 2, "a reading lost while waiting ends the wait instead of hanging");

	// the real thermometer (nvidia-smi on this machine)
	thermal_hooks() = {thermal_read_nvidia_smi, thermal_real_sleep, 0};
	int t = thermal_read_nvidia_smi();
	if (t < 0) printf("  --  real reading unavailable here (no nvidia-smi): skipped\n");
	else {
		printf("  real reading: %d C\n", t);
		check(t > 0 && t < 120, "real reading is a plausible temperature");
		double ww = thermal_gate(200, 190, 1);
		check(ww == 0, "gate with a very high limit does not wait on the real GPU");
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}
