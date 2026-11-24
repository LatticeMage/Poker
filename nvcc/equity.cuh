// The equity computation: kernels and host drivers for the 169x169 hand-class matrix. Used by equity.cu
// (the command line) and test_equity.cu (the checks).
//
// Sampled: a random combo of each class (skipped if they share a card), a random board from the other
// 48 cards, both hands scored by eval7 (poker.cuh). Equity = (wins + ties/2) / valid.
// Exact: every non-conflicting pair of combos on every one of the C(48,5) boards.
#pragma once

#include "dispatch.cuh"

__global__ void matchup_kernel(const int2* pairs, int per_thread, uint64_t seed, unsigned long long* wins2,
                               unsigned long long* valid) {
	const int pi = blockIdx.y;
	const int a = pairs[pi].x, b = pairs[pi].y;
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	Rng r;
	r.s = mix64(seed ^ mix64(((uint64_t)a * NCLASS + b) * 0xD1B54A32D192ED03ull) ^ mix64(tid + 0x1234567ull));
	const uint32_t na = c_ncombo[a], nb = c_ncombo[b];
	unsigned long long w2 = 0, v = 0;
	for (int s = 0; s < per_thread; s++) {
		const uint32_t ia = r.below(na), ib = r.below(nb);
		const int a0 = c_combos[a][ia][0], a1 = c_combos[a][ia][1];
		const int b0 = c_combos[b][ib][0], b1 = c_combos[b][ib][1];
		if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
		uint64_t used = (1ull << a0) | (1ull << a1) | (1ull << b0) | (1ull << b1);
		int ha[7], hb[7];
		ha[0] = a0; ha[1] = a1; hb[0] = b0; hb[1] = b1;
#pragma unroll
		for (int k = 0; k < 5; k++) {
			int c;
			do { c = r.below(52); } while ((used >> c) & 1);
			used |= 1ull << c;
			ha[2 + k] = c;
			hb[2 + k] = c;
		}
		const uint32_t sa = eval7(ha), sb = eval7(hb);
		w2 += sa > sb ? 2 : (sa == sb ? 1 : 0);
		v++;
	}
	for (int off = 16; off > 0; off >>= 1) {
		w2 += __shfl_down_sync(0xffffffffu, w2, off);
		v += __shfl_down_sync(0xffffffffu, v, off);
	}
	if ((threadIdx.x & 31) == 0) {
		atomicAdd(&wins2[pi], w2);
		atomicAdd(&valid[pi], v);
	}
}

// -- exact enumeration --------------------------------------------------------------------------
// Every non-conflicting (combo of a, combo of b) pair is a job; each job is split into CHUNKS slices of the
// C(48,5) boards, one slice per thread. Board k is the k-th 5-subset of the 48 free cards in lexicographic order.
static const int CHUNKS = 512;
static const uint32_t TOTAL_BOARDS = 1712304;  // C(48,5)
__constant__ uint32_t c_binom[49][6];

__global__ void exact_kernel(const uint32_t* jobs, const int2* pairs, uint64_t first, uint64_t njobs,
                             unsigned long long* wins2) {
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= njobs * CHUNKS) return;
	const uint32_t job = jobs[first + tid / CHUNKS];
	const uint32_t chunk = (uint32_t)(tid % CHUNKS);
	const int pi = job >> 8, ia = (job >> 4) & 15, ib = job & 15;
	const int a = pairs[pi].x, b = pairs[pi].y;
	const int a0 = c_combos[a][ia][0], a1 = c_combos[a][ia][1];
	const int b0 = c_combos[b][ib][0], b1 = c_combos[b][ib][1];
	const uint64_t used = (1ull << a0) | (1ull << a1) | (1ull << b0) | (1ull << b1);
	uint8_t rem[48];
	int n = 0;
	for (int c = 0; c < 52; c++)
		if (!((used >> c) & 1)) rem[n++] = c;
	const uint32_t start = (uint32_t)((uint64_t)chunk * TOTAL_BOARDS / CHUNKS);
	const uint32_t end = (uint32_t)((uint64_t)(chunk + 1) * TOTAL_BOARDS / CHUNKS);
	// unrank `start`
	int idx[5];
	uint32_t r = start;
	int x = 0;
	for (int i = 0; i < 5; i++) {
		int v = x;
		for (;; v++) {
			uint32_t cnt = c_binom[47 - v][4 - i];
			if (r < cnt) break;
			r -= cnt;
		}
		idx[i] = v;
		x = v + 1;
	}
	int ha[7], hb[7];
	ha[0] = a0; ha[1] = a1; hb[0] = b0; hb[1] = b1;
	unsigned long long w2 = 0;
	for (uint32_t k = start; k < end; k++) {
#pragma unroll
		for (int j = 0; j < 5; j++) {
			ha[2 + j] = rem[idx[j]];
			hb[2 + j] = ha[2 + j];
		}
		const uint32_t sa = eval7(ha), sb = eval7(hb);
		w2 += sa > sb ? 2 : (sa == sb ? 1 : 0);
		int i = 4;
		while (i >= 0 && idx[i] == 43 + i) i--;
		if (i < 0) break;
		idx[i]++;
		for (int j = i + 1; j < 5; j++) idx[j] = idx[j - 1] + 1;
	}
	atomicAdd(&wins2[pi], w2);
}

static const int BLOCK = 256;
static const int BLOCKS_PER_PAIR = 64;

// Runs `pairs` for about `samples` samples each (in batches of `pairs_per_batch` pairs); returns equity per pair and
// valid counts. Every launch goes through dispatch(), which checks the GPU temperature first.
static inline void run_pairs(const std::vector<int2>& pairs, long long samples, uint64_t seed, std::vector<double>& eq,
                             std::vector<unsigned long long>& valid_out, bool verbose = false,
                             uint64_t pairs_per_batch = 256) {
	const int P = (int)pairs.size();
	const int per_thread = (int)((samples + (long long)BLOCKS_PER_PAIR * BLOCK - 1) / ((long long)BLOCKS_PER_PAIR * BLOCK));
	int2* d_pairs;
	unsigned long long *d_w, *d_v;
	CK(cudaMalloc(&d_pairs, P * sizeof(int2)));
	CK(cudaMalloc(&d_w, P * sizeof(unsigned long long)));
	CK(cudaMalloc(&d_v, P * sizeof(unsigned long long)));
	CK(cudaMemcpy(d_pairs, pairs.data(), P * sizeof(int2), cudaMemcpyHostToDevice));
	CK(cudaMemset(d_w, 0, P * sizeof(unsigned long long)));
	CK(cudaMemset(d_v, 0, P * sizeof(unsigned long long)));
	dispatch(P, pairs_per_batch, [&](uint64_t first, uint64_t count) {
		dim3 grid(BLOCKS_PER_PAIR, (unsigned)count);
		matchup_kernel<<<grid, BLOCK>>>(d_pairs + first, per_thread, seed, d_w + first, d_v + first);
	}, "pairs", verbose);
	std::vector<unsigned long long> w(P), v(P);
	CK(cudaMemcpy(w.data(), d_w, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(v.data(), d_v, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	cudaFree(d_pairs); cudaFree(d_w); cudaFree(d_v);
	eq.resize(P);
	valid_out = v;
	for (int i = 0; i < P; i++) eq[i] = v[i] ? (double)w[i] / (2.0 * (double)v[i]) : 0.5;
}

// All class pairs i < j, in row-major order.
static inline std::vector<int2> all_pairs() {
	std::vector<int2> p;
	for (int i = 0; i < NCLASS; i++)
		for (int j = i + 1; j < NCLASS; j++) p.push_back(make_int2(i, j));
	return p;
}

// Exact equity of each pair: every non-conflicting (combo of a, combo of b) on all C(48,5) boards.
// njob_out[p] is the number of combo pairs that went into pair p.
static inline void exact_pairs(const std::vector<int2>& pairs, std::vector<double>& eq,
                               std::vector<unsigned long long>& njob_out, bool verbose) {
	uint32_t binom[49][6] = {};
	for (int nn = 0; nn <= 48; nn++) {
		binom[nn][0] = 1;
		for (int k = 1; k <= 5 && k <= nn; k++) binom[nn][k] = binom[nn - 1][k - 1] + (k <= nn - 1 ? binom[nn - 1][k] : 0);
	}
	CK(cudaMemcpyToSymbol(c_binom, binom, sizeof(binom)));
	std::vector<uint32_t> jobs;
	std::vector<unsigned long long> njob(pairs.size(), 0);
	for (size_t p = 0; p < pairs.size(); p++) {
		int a = pairs[p].x, b = pairs[p].y;
		for (int ia = 0; ia < g_ncombo[a]; ia++)
			for (int ib = 0; ib < g_ncombo[b]; ib++) {
				int a0 = g_combos[a][ia][0], a1 = g_combos[a][ia][1], b0 = g_combos[b][ib][0], b1 = g_combos[b][ib][1];
				if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
				jobs.push_back((uint32_t)p << 8 | ia << 4 | ib);
				njob[p]++;
			}
	}
	if (verbose) printf("%zu pairs, %zu jobs, %.3e boards\n", pairs.size(), jobs.size(), (double)jobs.size() * TOTAL_BOARDS);
	uint32_t* d_jobs;
	int2* d_pairs;
	unsigned long long* d_w;
	CK(cudaMalloc(&d_jobs, jobs.size() * 4));
	CK(cudaMalloc(&d_pairs, pairs.size() * sizeof(int2)));
	CK(cudaMalloc(&d_w, pairs.size() * 8));
	CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(int2), cudaMemcpyHostToDevice));
	CK(cudaMemset(d_w, 0, pairs.size() * 8));
	dispatch(jobs.size(), 32768, [&](uint64_t first, uint64_t nj) {
		uint64_t threads = nj * CHUNKS;
		exact_kernel<<<(unsigned)((threads + 255) / 256), 256>>>(d_jobs, d_pairs, first, nj, d_w);
	}, "jobs", verbose);
	std::vector<unsigned long long> w(pairs.size());
	CK(cudaMemcpy(w.data(), d_w, pairs.size() * 8, cudaMemcpyDeviceToHost));
	cudaFree(d_jobs); cudaFree(d_pairs); cudaFree(d_w);
	eq.assign(pairs.size(), 0.5);
	for (size_t p = 0; p < pairs.size(); p++)
		if (njob[p]) eq[p] = (double)w[p] / (2.0 * (double)njob[p] * (double)TOTAL_BOARDS);
	njob_out = njob;
}
