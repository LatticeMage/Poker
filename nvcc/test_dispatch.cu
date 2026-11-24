// Checks for dispatch.cuh: every item is covered exactly once, in order, in batches of the requested size,
// and the thermal gate runs before every batch (counted with a fake thermometer).
//
//   nvcc -O3 -arch=sm_89 -o test_dispatch test_dispatch.cu && ./test_dispatch

#include "dispatch.cuh"

#include <vector>

static int g_reads = 0;
static int g_temp = 40;
static int fake_read() { g_reads++; return g_temp; }
static void fake_sleep(double) { g_temp -= 5; }  // each poll the card cools a little

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok " : "BAD", what);
	failures += !ok;
}

int main() {
	thermal_hooks() = {fake_read, fake_sleep, 0};

	// 1. coverage: 10 items in batches of 4 -> [0,4) [4,8) [8,10)
	std::vector<std::pair<uint64_t, uint64_t>> got;
	uint64_t n = dispatch(10, 4, [&](uint64_t first, uint64_t count) { got.push_back({first, count}); });
	check(n == 3 && got.size() == 3 && got[0] == std::make_pair<uint64_t, uint64_t>(0, 4) &&
	          got[1] == std::make_pair<uint64_t, uint64_t>(4, 4) && got[2] == std::make_pair<uint64_t, uint64_t>(8, 2),
	      "10 items in batches of 4: three batches [0,4) [4,8) [8,10)");

	// 2. exact multiple, a single batch, nothing to do, a zero batch size
	got.clear();
	n = dispatch(8, 4, [&](uint64_t f, uint64_t c) { got.push_back({f, c}); });
	check(n == 2 && got.back().first + got.back().second == 8, "8 items in batches of 4: two batches, covering 8");
	got.clear();
	n = dispatch(3, 100, [&](uint64_t f, uint64_t c) { got.push_back({f, c}); });
	check(n == 1 && got[0].second == 3, "a batch larger than the work: one batch with everything");
	n = dispatch(0, 4, [&](uint64_t, uint64_t) { got.push_back({0, 0}); });
	check(n == 0, "no work: no batch");
	got.clear();
	n = dispatch(3, 0, [&](uint64_t f, uint64_t c) { got.push_back({f, c}); });
	check(n == 3, "a batch size of 0 is treated as 1");

	// 3. the thermal gate runs once before each batch when the card is cool
	g_reads = 0;
	g_temp = 40;
	n = dispatch(10, 2, [&](uint64_t, uint64_t) {});
	check(n == 5 && g_reads == 5, "cool card: one temperature reading before each of the 5 batches");

	// 4. a hot card makes dispatch wait before the batch, and the batch runs only after it cooled down
	g_reads = 0;
	g_temp = 70;
	int temp_at_launch = 0;
	n = dispatch(1, 1, [&](uint64_t, uint64_t) { temp_at_launch = g_temp; });
	check(n == 1 && temp_at_launch <= 52, "hot card (70 C): the batch starts only once it is at or below 52 C");

	// 4b. a task starts only when cool, but once running it goes on until 60 C
	g_reads = 0;
	g_temp = 56;  // between 52 and 60: not cool enough to start
	int temps_seen[3] = {0, 0, 0};
	int calls = 0;
	auto reheat = [&](uint64_t, uint64_t) {
		temps_seen[calls++] = g_temp;
		g_temp = 56;  // the card heats back up to 56 C during every batch
	};
	n = dispatch(3, 1, reheat);
	check(n == 3 && temps_seen[0] <= 52, "starting at 56 C: the first batch waits until 52 C");
	check(temps_seen[1] == 56 && temps_seen[2] == 56, "later batches of the running task go on at 56 C (below the 60 C limit)");

	// 5. the work of a batch sees the order of the batches (nothing is launched out of order)
	std::vector<uint64_t> firsts;
	dispatch(100, 7, [&](uint64_t f, uint64_t) { firsts.push_back(f); });
	bool ordered = true;
	for (size_t i = 1; i < firsts.size(); i++) ordered = ordered && firsts[i] == firsts[i - 1] + 7;
	check(ordered && firsts.size() == 15, "100 items in batches of 7: 15 batches in order");

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}
