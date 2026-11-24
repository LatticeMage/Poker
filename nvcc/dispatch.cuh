// The one way GPU work is launched: split the work into batches and hand the batches to dispatch().
// Before every batch it checks the GPU temperature (thermal.cuh) and waits while the card is too hot; it then
// runs the batch, waits for it to finish, and reports progress. Tools do not call thermal_gate() themselves.
//
//   dispatch(total_items, items_per_batch, [&](uint64_t first, uint64_t count) {
//       kernel<<<grid, block>>>(..., first, count);
//   }, "label", verbose);
//
// The batch size is the only knob: a batch is the unit that can be postponed, so it should take seconds, not
// minutes. A launch that fails (CK) stops the program, as everywhere else.
#pragma once

#include "poker.cuh"
#include "thermal.cuh"

// Calls launch(first, count) for consecutive batches covering [0, total). Returns the number of batches run.
// With `verbose` it prints "label done/total  seconds  eta" after every batch.
template <class Launch>
static inline uint64_t dispatch(uint64_t total, uint64_t per_batch, Launch&& launch, const char* label = "",
                                bool verbose = false) {
	if (per_batch == 0) per_batch = 1;
	uint64_t batches = 0;
	auto t0 = std::chrono::steady_clock::now();
	double waited = 0;
	for (uint64_t first = 0; first < total; first += per_batch) {
		const uint64_t count = std::min<uint64_t>(per_batch, total - first);
		waited += thermal_gate();
		launch(first, count);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		batches++;
		if (verbose) {
			const uint64_t done = first + count;
			const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			const double work = secs - waited;  // the estimate ignores time spent cooling
			printf("%s %llu/%llu  %.1fs  eta %.0fs\n", label, (unsigned long long)done, (unsigned long long)total, secs,
			       work / done * (total - done));
			fflush(stdout);
		}
	}
	return batches;
}
