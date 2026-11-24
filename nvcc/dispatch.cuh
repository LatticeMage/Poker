// The one way GPU work is launched: split the work into batches and hand the batches to dispatch().
// Before the first batch it waits until the GPU is cool enough to start (<= 53 C), and before every later batch
// it waits while the card is too hot (>= 63 C, back to <= 53 C); thermal.cuh has the rules. It then runs the
// batch, waits for it to finish, and reports progress. Tools do not call thermal_gate() themselves.
//
//   dispatch(total_items, items_per_batch, [&](uint64_t first, uint64_t count) {
//       kernel<<<grid, block>>>(..., first, count);
//   }, "label", verbose, &relay);
//
// Relay: a long task is a relay, not a run from scratch. The files are the truth and the GPU is only a workspace:
// with a Relay, dispatch writes the task's state back to its file after EVERY batch (atomically, and flushed to
// disk), so whatever happens (eviction, SIGKILL, a crash, a reboot) at most the batch that was running is lost.
// When the task is started again with the same problem it restores that state and carries on from the first
// batch that was not finished. The caller supplies two functions that write and read
// its own state (copy device accumulators to the host in save_state, back in load_state). The file also holds
// `tag`, a hash of everything the answer depends on (sizes, seeds, inputs): a file with another tag, another total
// or another batch size is ignored. A finished task keeps its file with done == total, so running it again
// restores the answer without launching anything; delete the file to recompute.
//
// Eviction: when the GPU gets too hot, gpu-cooler.sh sends SIGTERM to the CUDA processes (it does not pause
// them: a paused process keeps its GPU context and the card stays hot). dispatch() catches SIGTERM/SIGINT, finishes
// the batch it is running, saves the relay state, and exits with status 75 (EXIT_EVICTED). gpu-run.sh then waits
// for the card to cool down and runs the same command again, which carries on from the saved batch.
//
// The batch size is the only knob: a batch is the unit that can be postponed or lost, so it should take seconds,
// not minutes. A launch that fails (CK) stops the program, as everywhere else.
#pragma once

#include "poker.cuh"
#include "thermal.cuh"

#include <csignal>
#include <unistd.h>
#include <functional>
#include <string>

static const int EXIT_EVICTED = 75;  // "stopped because asked to; run me again" (EX_TEMPFAIL)
static volatile sig_atomic_t g_stop_requested = 0;
static void dispatch_on_signal(int) { g_stop_requested = 1; }
static inline void dispatch_install_signal_handlers() {
	static bool done = false;
	if (done) return;
	done = true;
	struct sigaction sa;
	memset(&sa, 0, sizeof sa);
	sa.sa_handler = dispatch_on_signal;
	sigaction(SIGTERM, &sa, nullptr);
	sigaction(SIGINT, &sa, nullptr);
}
static inline void dispatch_request_stop() { g_stop_requested = 1; }  // for tests
static inline void dispatch_clear_stop() { g_stop_requested = 0; }

struct Relay {
	std::string path;                         // checkpoint file; empty = no relay
	uint64_t tag = 0;                         // hash of the problem; a file with another tag is not used
	double every_s = 0;                       // 0 = save after every batch (the rule); a larger value only for tiny batches
	std::function<void(FILE*)> save_state;    // write the caller's state
	std::function<bool(FILE*)> load_state;    // restore it; false = unusable, start from scratch
	uint64_t stop_after = 0;                  // for tests: stop cleanly after this many batches in this call
	bool return_on_stop = false;              // for tests: on SIGTERM return instead of exiting with EXIT_EVICTED
	uint64_t done = 0;                        // out: items finished so far (restored ones included)
};

// FNV-1a, for building a Relay tag from the numbers a result depends on.
static inline uint64_t relay_hash(uint64_t h, const void* data, size_t n) {
	const unsigned char* p = (const unsigned char*)data;
	for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
	return h;
}
static inline uint64_t relay_hash_u64(uint64_t h, uint64_t v) { return relay_hash(h, &v, sizeof v); }

static const char RELAY_MAGIC[4] = {'R', 'L', 'Y', '1'};

static inline bool relay_save(Relay& r, uint64_t total, uint64_t per_batch, uint64_t done) {
	const std::string tmp = r.path + ".tmp";
	FILE* f = fopen(tmp.c_str(), "wb");
	if (!f) return false;
	const uint64_t head[4] = {r.tag, total, per_batch, done};
	bool ok = fwrite(RELAY_MAGIC, 1, 4, f) == 4 && fwrite(head, sizeof head, 1, f) == 1;
	if (ok) r.save_state(f);
	ok = ok && fflush(f) == 0 && !ferror(f) && fsync(fileno(f)) == 0;  // on disk before it replaces the old file
	fclose(f);
	return ok && rename(tmp.c_str(), r.path.c_str()) == 0;  // a half-written file never replaces a good one
}

// The number of finished items a relay file says it holds (0 if there is no usable file); for tools and tests.
static inline uint64_t relay_peek_done(const std::string& path) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) return 0;
	char magic[4];
	uint64_t head[4];
	uint64_t done = 0;
	if (fread(magic, 1, 4, f) == 4 && memcmp(magic, RELAY_MAGIC, 4) == 0 && fread(head, sizeof head, 1, f) == 1) done = head[3];
	fclose(f);
	return done;
}

// Returns the number of finished items found in the file, or 0 if there is none, or it is for another problem.
static inline uint64_t relay_load(Relay& r, uint64_t total, uint64_t per_batch) {
	FILE* f = fopen(r.path.c_str(), "rb");
	if (!f) return 0;
	char magic[4];
	uint64_t head[4];
	uint64_t done = 0;
	if (fread(magic, 1, 4, f) == 4 && memcmp(magic, RELAY_MAGIC, 4) == 0 && fread(head, sizeof head, 1, f) == 1 &&
	    head[0] == r.tag && head[1] == total && head[2] == per_batch && head[3] <= total &&
	    (head[3] % per_batch == 0 || head[3] == total)) {
		if (r.load_state(f)) done = head[3];
		else fprintf(stderr, "[relay] %s is unusable, starting from scratch\n", r.path.c_str());
	} else {
		fprintf(stderr, "[relay] %s is for another problem, starting from scratch\n", r.path.c_str());
	}
	fclose(f);
	return done;
}

// Calls launch(first, count) for consecutive batches covering [0, total). Returns the number of batches run.
// With `verbose` it prints "label done/total  seconds  eta" after every batch.
template <class Launch>
static inline uint64_t dispatch(uint64_t total, uint64_t per_batch, Launch&& launch, const char* label = "",
                                bool verbose = false, Relay* relay = nullptr) {
	if (per_batch == 0) per_batch = 1;
	dispatch_install_signal_handlers();
	const bool relaying = relay && !relay->path.empty();
	uint64_t start = 0;
	if (relaying) {
		start = relay_load(*relay, total, per_batch);
		if (start > 0)
			fprintf(stderr, "[relay] %s: restored %llu/%llu from %s\n", label, (unsigned long long)start,
			        (unsigned long long)total, relay->path.c_str());
		relay->done = start;
	}
	uint64_t batches = 0;
	auto t0 = std::chrono::steady_clock::now();
	auto last_save = t0;
	double waited = 0;
	for (uint64_t first = start; first < total; first += per_batch) {
		const uint64_t count = std::min<uint64_t>(per_batch, total - first);
		waited += thermal_gate(-1, -1, -1, first == start);  // the first batch of this call starts a task
		if (g_stop_requested) {  // asked to stop while waiting: nothing is running, the saved state is current
			fprintf(stderr, "[evicted] %s: stopping before batch at %llu/%llu\n", label, (unsigned long long)first, (unsigned long long)total);
			if (relaying && relay->return_on_stop) return batches;
			exit(EXIT_EVICTED);
		}
		launch(first, count);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		batches++;
		const uint64_t done = first + count;
		if (verbose) {
			const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			const double work = secs - waited;  // the estimate ignores time spent cooling
			printf("%s %llu/%llu  %.1fs  eta %.0fs\n", label, (unsigned long long)done, (unsigned long long)total, secs,
			       work / (done - start) * (total - done));
			fflush(stdout);
		}
		if (relaying) {
			relay->done = done;
			const double since = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_save).count();
			const bool stopping = relay->stop_after && batches >= relay->stop_after;
			if (relay->every_s <= 0 || since >= relay->every_s || done == total || stopping) {
				if (!relay_save(*relay, total, per_batch, done)) fprintf(stderr, "[relay] cannot write %s\n", relay->path.c_str());
				last_save = std::chrono::steady_clock::now();
			}
			if (stopping && done < total) return batches;
		}
		if (g_stop_requested && done < total) {  // evicted: the batch is finished, keep its result and leave
			if (relaying) {
				relay->done = done;
				if (!relay_save(*relay, total, per_batch, done)) fprintf(stderr, "[relay] cannot write %s\n", relay->path.c_str());
			}
			fprintf(stderr, "[evicted] %s: stopped after %llu/%llu%s\n", label, (unsigned long long)done, (unsigned long long)total,
			        relaying ? "; run it again to carry on" : " (no relay: progress is lost)");
			if (relaying && relay->return_on_stop) return batches;
			exit(EXIT_EVICTED);
		}
	}
	return batches;
}
