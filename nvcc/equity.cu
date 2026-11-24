// Hand-class equity matrix on the GPU: sampled or exact. Checked by test_equity.cu.
//
//   nvcc -O3 -arch=sm_89 -o equity equity.cu
//   ./equity pair AKs QQ 8000000          equity of one class pair
//   ./equity matrix 1000000 out.bin       all 14,196 pairs; out.bin = 169*169 float64, row-major
//   ./equity exact out.bin [A B]          every combo pair on every board: no sampling noise
//
// Each sample: a random combo of each class (skipped if they share a card), a random board from the other
// 48 cards, both hands scored by eval7 (poker.cuh). Equity = (wins + ties/2) / valid.

#include "poker.cuh"

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

// Runs `pairs` for about `samples` samples each; returns equity per pair and valid counts.
static void run_pairs(const std::vector<int2>& pairs, long long samples, uint64_t seed, std::vector<double>& eq,
                      std::vector<unsigned long long>& valid_out) {
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
	dim3 grid(BLOCKS_PER_PAIR, P);
	matchup_kernel<<<grid, BLOCK>>>(d_pairs, per_thread, seed, d_w, d_v);
	CK(cudaGetLastError());
	CK(cudaDeviceSynchronize());
	std::vector<unsigned long long> w(P), v(P);
	CK(cudaMemcpy(w.data(), d_w, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(v.data(), d_v, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	cudaFree(d_pairs); cudaFree(d_w); cudaFree(d_v);
	eq.resize(P);
	valid_out = v;
	for (int i = 0; i < P; i++) eq[i] = v[i] ? (double)w[i] / (2.0 * (double)v[i]) : 0.5;
}

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: equity pair|matrix|test ...\n"); return 1; }
	setup_classes();
	std::string mode = argv[1];
	if (mode == "pair" && argc >= 5) {
		int a = class_index(argv[2]), b = class_index(argv[3]);
		long long samples = atoll(argv[4]);
		std::vector<double> eq;
		std::vector<unsigned long long> v;
		auto t0 = std::chrono::steady_clock::now();
		run_pairs({make_int2(a, b)}, samples, 12345, eq, v);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("%s vs %s: %.5f  (%llu valid samples, %.2fs)\n", argv[2], argv[3], eq[0], v[0], secs);
	} else if (mode == "matrix" && argc >= 4) {
		long long samples = atoll(argv[2]);
		std::vector<double> m(NCLASS * NCLASS, 0.5);
		std::vector<int2> all;
		for (int i = 0; i < NCLASS; i++)
			for (int j = i + 1; j < NCLASS; j++) all.push_back(make_int2(i, j));
		const int PB = 256;
		auto t0 = std::chrono::steady_clock::now();
		for (size_t s = 0; s < all.size(); s += PB) {
			std::vector<int2> chunk(all.begin() + s, all.begin() + std::min(all.size(), s + PB));
			std::vector<double> eq;
			std::vector<unsigned long long> v;
			run_pairs(chunk, samples, 20261124, eq, v);
			for (size_t k = 0; k < chunk.size(); k++) {
				m[chunk[k].x * NCLASS + chunk[k].y] = eq[k];
				m[chunk[k].y * NCLASS + chunk[k].x] = 1.0 - eq[k];
			}
			double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			size_t done = s + chunk.size();
			printf("%zu/%zu pairs  %.1fs  eta %.1fs\n", done, all.size(), secs, secs / done * (all.size() - done));
			fflush(stdout);
		}
		FILE* f = fopen(argv[3], "wb");
		fwrite(m.data(), sizeof(double), m.size(), f);
		fclose(f);
		printf("wrote %s\n", argv[3]);
	} else if (mode == "exact" && argc >= 3) {
		// ./equity exact out.bin [A B]   (the optional pair restricts the run to that one matchup)
		uint32_t binom[49][6] = {};
		for (int nn = 0; nn <= 48; nn++) {
			binom[nn][0] = 1;
			for (int k = 1; k <= 5 && k <= nn; k++) binom[nn][k] = binom[nn - 1][k - 1] + (k <= nn - 1 ? binom[nn - 1][k] : 0);
		}
		CK(cudaMemcpyToSymbol(c_binom, binom, sizeof(binom)));
		std::vector<int2> pairs;
		if (argc >= 5) pairs.push_back(make_int2(class_index(argv[3]), class_index(argv[4])));
		else
			for (int i = 0; i < NCLASS; i++)
				for (int j = i + 1; j < NCLASS; j++) pairs.push_back(make_int2(i, j));
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
		printf("%zu pairs, %zu jobs, %.3e boards\n", pairs.size(), jobs.size(), (double)jobs.size() * TOTAL_BOARDS);
		uint32_t* d_jobs;
		int2* d_pairs;
		unsigned long long* d_w;
		CK(cudaMalloc(&d_jobs, jobs.size() * 4));
		CK(cudaMalloc(&d_pairs, pairs.size() * sizeof(int2)));
		CK(cudaMalloc(&d_w, pairs.size() * 8));
		CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(int2), cudaMemcpyHostToDevice));
		CK(cudaMemset(d_w, 0, pairs.size() * 8));
		const uint64_t BATCH = 32768;  // jobs per launch
		auto t0 = std::chrono::steady_clock::now();
		for (uint64_t first = 0; first < jobs.size(); first += BATCH) {
			uint64_t nj = std::min<uint64_t>(BATCH, jobs.size() - first);
			uint64_t threads = nj * CHUNKS;
			exact_kernel<<<(unsigned)((threads + 255) / 256), 256>>>(d_jobs, d_pairs, first, nj, d_w);
			CK(cudaGetLastError());
			CK(cudaDeviceSynchronize());
			double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			uint64_t done = first + nj;
			printf("%llu/%zu jobs  %.1fs  eta %.0fs\n", (unsigned long long)done, jobs.size(), secs, secs / done * (jobs.size() - done));
			fflush(stdout);
		}
		std::vector<unsigned long long> w(pairs.size());
		CK(cudaMemcpy(w.data(), d_w, pairs.size() * 8, cudaMemcpyDeviceToHost));
		std::vector<double> m(NCLASS * NCLASS, 0.5);
		for (size_t p = 0; p < pairs.size(); p++) {
			double e = (double)w[p] / (2.0 * (double)njob[p] * (double)TOTAL_BOARDS);
			m[pairs[p].x * NCLASS + pairs[p].y] = e;
			m[pairs[p].y * NCLASS + pairs[p].x] = 1.0 - e;
			if (pairs.size() == 1) printf("%s vs %s: %.6f exact (%llu combo pairs)\n", argv[3], argv[4], e, njob[p]);
		}
		FILE* f = fopen(argv[2], "wb");
		fwrite(m.data(), sizeof(double), m.size(), f);
		fclose(f);
		printf("wrote %s\n", argv[2]);
	} else {
		fprintf(stderr, "bad arguments\n");
		return 1;
	}
	return 0;
}
